"""Balanced classical additions, split safety and task-aware genre sampling."""
from collections import Counter
from dataclasses import astuple
import hashlib
import json
import random
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from improvnet.model import melody_config as cfg
from improvnet.scripts.melody_add_classical import add_classical
from improvnet.scripts.melody_prepare import make_record, read_job
from improvnet.utils.melody import MelodyTokenizer, Note, TASKS, corrupt_melody, write_midi_notes
from improvnet.utils.melody_data import MelodyDataset, prepare_example


def settings(**updates):
    return SimpleNamespace(**dict({k: v for k, v in vars(cfg).items() if k.isupper()}, **updates))


def write_corpus(path, records):
    rows = []
    with path.open('wb') as handle:
        for i, record in enumerate(records):
            rows.append(dict(offset=handle.tell(), source=record['source'], genre=record['genre'],
                             has_harmony=record.get('has_harmony', True), has_drums=record.get('has_drums', False),
                             instruments=record.get('instruments', ['Acoustic Piano', 'drum'] if record.get('has_drums') else ['Acoustic Piano']),
                             has_multi_instrument=record.get('has_multi_instrument', bool(record.get('has_drums'))),
                             sha256=hashlib.sha256(str(i).encode()).hexdigest(), group=str(i),
                             split=record.get('split', 'train')))
            handle.write((json.dumps(record) + '\n').encode())
    path.with_suffix('.index.json').write_text(json.dumps(dict(
        version=3, tokenizer=MelodyTokenizer().signature(), skyline_tolerance_ms=50, rows=rows)))
    return rows


def test_classical_subset_matches_base_size_preserves_splits_and_is_reproducible(tmp_path):
    base = tmp_path / 'base.jsonl'
    base_records, base_hashes = [], []
    for i in range(3):
        path = tmp_path / f'base{i}.mid'
        notes = [Note('Acoustic Piano', 60 + j, 90, j * (700 + i * 10), 70) for j in range(5)]
        write_midi_notes(notes, path)
        job, digest, parsed, error = read_job((path, 'jazz' if i != 1 else 'blues', 'fixture', '', False))
        assert error is None
        record, _ = make_record(job, parsed)
        record['split'] = ('train', 'validation', 'test')[i]
        base_records.append(record)
        base_hashes.append(digest)
    rows = write_corpus(base, base_records)
    for row, digest in zip(rows, base_hashes):
        row['sha256'] = digest
    index = json.loads(base.with_suffix('.index.json').read_text())
    index['rows'] = rows
    base.with_suffix('.index.json').write_text(json.dumps(index))
    midi_index = tmp_path / 'index.sqlite'
    with sqlite3.connect(midi_index) as db:
        db.execute('CREATE TABLE files(path TEXT, collection TEXT, genre TEXT, group_id TEXT)')
        for i in range(3):
            path = tmp_path / f'classical{i}.mid'
            notes = [Note('Acoustic Piano', 72 + j, 90, j * (200 + i * 50), 100 + i * 10) for j in range(5)]
            if i == 1:
                notes.append(Note('drum', 42, 80, 0, 20))
            write_midi_notes(notes, path)
            db.execute('INSERT INTO files VALUES(?,?,?,?)',
                       (str(path), 'aria', 'classical', 'shared-recording' if i < 2 else 'other-recording'))
        duplicate = tmp_path / 'duplicate.mid'
        duplicate.write_bytes((tmp_path / 'classical0.mid').read_bytes())
        db.execute('INSERT INTO files VALUES(?,?,?,?)', (str(duplicate), 'aria', 'classical', 'duplicate'))
        transposed = tmp_path / 'base-copy.mid'
        write_midi_notes([Note('Acoustic Piano', 62 + j, 90, j * 700, 70) for j in range(5)], transposed)
        db.execute('INSERT INTO files VALUES(?,?,?,?)', (str(transposed), 'maestro', 'classical', 'base-copy'))
        db.execute('INSERT INTO files VALUES(?,?,?,?)', ('missing.mid', 'aria', 'jazz', 'excluded-label'))
    output, repeat = tmp_path / 'mixed.jsonl', tmp_path / 'repeat.jsonl'
    report = add_classical(base, midi_index, output, workers=0)
    add_classical(base, midi_index, repeat, workers=0)
    assert report['classical'] == report['jazz_blues'] == 3
    assert output.read_bytes() == repeat.read_bytes()
    merged = json.loads(output.with_suffix('.index.json').read_text())
    assert merged['rows'][:3] == rows
    assert output.read_bytes().startswith(base.read_bytes())
    assert len({r['sha256'] for r in merged['rows']}) == 6
    new = merged['rows'][3:]
    shared = [r for r in new if r['source_group'] == 'shared-recording']
    assert len(shared) == 2 and len({r['split'] for r in shared}) == 1
    assert sum(r['has_drums'] for r in new) == 1
    with output.open('rb') as handle:
        for row in merged['rows']:
            handle.seek(row['offset'])
            record = json.loads(handle.readline())
            assert record['genre'] == row['genre']
    with pytest.raises(FileExistsError):
        add_classical(base, midi_index, output, workers=0)


@pytest.mark.parametrize('task', TASKS)
def test_classical_sampling_matches_combined_other_genres_per_eligible_task(tmp_path, monkeypatch, task):
    data = tmp_path / 'data.jsonl'
    write_corpus(data, [dict(source='fixture', genre=g, has_drums=g != 'blues')
                        for g in ('classical', 'jazz', 'blues')])
    monkeypatch.setattr('improvnet.utils.melody_data.prepare_example', lambda record, *a, **kw: record['genre'])
    weights = tuple(int(t == task) for t in TASKS)
    dataset = MelodyDataset(data, 'train', settings(TASK_WEIGHTS=weights))
    counts = Counter(dataset[i] for i in range(2000))
    assert .62 < counts['classical'] / 2000 < .71
    if task == 'stem':
        assert counts['blues'] == 0 and .29 < counts['jazz'] / 2000 < .38
    else:
        assert all(.13 < counts[g] / 2000 < .20 for g in ('jazz', 'blues'))


def test_drum_sampling_falls_back_to_only_genres_with_real_drums(tmp_path, monkeypatch):
    data = tmp_path / 'data.jsonl'
    write_corpus(data, [dict(source='fixture', genre=g, has_drums=g == 'jazz')
                        for g in ('classical', 'jazz', 'blues')])
    monkeypatch.setattr('improvnet.utils.melody_data.prepare_example', lambda record, *a, **kw: record['genre'])
    dataset = MelodyDataset(data, 'train', settings(TASK_WEIGHTS=(0, 0, 0, 1)))
    assert {dataset[i] for i in range(100)} == {'jazz'}


def test_dense_chord_corruption_falls_back_to_masking_without_dropping_target_notes(monkeypatch):
    from improvnet.inference.melody_inference import MelodyInference
    notes = [Note('Acoustic Piano', p, 90, 0, 1000) for p in range(108, 0, -1)]
    record = dict(notes=[astuple(n) for n in notes], melody_indices=[0],
                  has_harmony=True, drums=[], genre='classical')
    def grow(core, strength, rng, tok, *args, **kwargs):
        return corrupt_melody(core, strength, rng, tok, names=kwargs.get('names', ['note_modification']))
    monkeypatch.setattr('improvnet.utils.melody_data.corrupt_melody', grow)
    monkeypatch.setattr('improvnet.inference.melody_inference.corrupt_melody', grow)
    x = prepare_example(record, random.Random(1), settings(INPUT_MONO_PROB=0, HISTORY_PROB=0), 'refine', 1)
    tok = MelodyTokenizer()
    assert x['source_length'] <= 512 and x['response_length'] == 4 * len(notes) + 1
    assert x['input'].tolist().count(tok.tok_to_id['<MASK>']) == 4 * len(notes)
    inference = MelodyInference.__new__(MelodyInference)
    inference.tokenizer = tok
    inference.meta = dict(seq_len=8192, chunk_tokens=512, history_tokens=2048,
                          harmony_expansion=3., max_corruptions=3)
    inference.generate = lambda *args, **kwargs: notes
    result, stats = inference.transform(notes, 'refine', 'jazz', 1, random.Random(1), monophonic=False)
    assert result == notes and len(stats) == 1
    assert stats[0]['bounded_corruption'] and stats[0]['source_tokens'] <= 512
    assert stats[0]['response_tokens'] == 4 * len(notes) + 1
