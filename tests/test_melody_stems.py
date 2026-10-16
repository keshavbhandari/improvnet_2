"""Missing-instrument supervision, sampling and full-mix inference contracts."""
from collections import Counter
from dataclasses import astuple
import hashlib
import json
import random
import sqlite3
from types import SimpleNamespace

import pytest
import torch

from improvnet.model import melody_config as cfg
from improvnet.utils.ar_utils import ProcessData
from improvnet.utils.melody import (MelodyTokenizer, Note, TASKS, make_prompt,
                                   notes_to_tokens, tokens_to_notes, ordered, write_midi_notes)
from improvnet.utils.melody_data import (MelodyDataset, NoStemWindow, prepare_example,
                                       stem_source, collate_examples)


def settings(**updates):
    return SimpleNamespace(**dict({k: v for k, v in vars(cfg).items() if k.isupper()}, **updates))


def mix(count=100):
    notes = [Note(inst, pitch + i % 3, 90, i * 200, 180)
             for i in range(count) for inst, pitch in (('Acoustic Piano', 72), ('Acoustic Bass', 40))]
    drums = [Note('drum', 42, 80, i * 200, 20) for i in range(count)]
    return dict(notes=[astuple(n) for n in notes], drums=[astuple(n) for n in drums],
                melody_indices=list(range(0, len(notes), 2)), has_harmony=True, genre='jazz')


def streams(example):
    tok = MelodyTokenizer()
    prompt = [tok.id_to_tok[i] for i in example['input'].tolist()]
    start = prompt.index('<SEP>') + 1
    source = tokens_to_notes(prompt[start:start + example['source_length']], tok)
    target = tokens_to_notes([tok.id_to_tok[i] for i in example['target'].tolist() if i != -100], tok)
    return prompt, source, target


@pytest.mark.parametrize('instrument', ('Acoustic Piano', 'Acoustic Bass', 'drum'))
def test_remove_entire_selected_instrument_and_restore_complete_mix(instrument):
    record = mix()
    x = prepare_example(record, random.Random(6), settings(HISTORY_PROB=1), 'stem',
                        target_instrument=instrument)
    prompt, source, target = streams(x)
    assert prompt[:2] == ['<STEM>', ('instrument', instrument)]
    assert not x['input_mono'] and not x['history_mono']
    assert source and all(n.instrument != instrument for n in source)
    expected = [Note(n[0], n[1], n[2], n[3] - x['start_ms'], n[4])
                for n in record['notes'] + record['drums'] if x['start_ms'] <= n[3] < x['end_ms']]
    assert Counter(target) == Counter(expected)
    assert Counter(source) == Counter(n for n in expected if n.instrument != instrument)
    history = tokens_to_notes(prompt[2:2 + x['history_length']])
    assert {n.instrument for n in history} == {'Acoustic Piano', 'Acoustic Bass', 'drum'}
    assert x['stem_instrument'] == ProcessData().INSTRUMENT_CLASSES.index(instrument)
    # The diagnostics cover every attribute of the missing notes, no copied notes.
    assert x['stem_mask'].sum() == 4 * sum(n.instrument == instrument for n in target)
    assert not (x['stem_mask'] & (x['target'] == -100)).any()
    batch = collate_examples([x, prepare_example(record, random.Random(7), task='rhythm')])
    assert batch['stem_mask'][0].sum() == x['stem_mask'].sum()
    assert batch['stem_mask'][1].sum() == 0 and batch['stem_instrument'][1] == -1
    assert len(x['input']) + 3 <= cfg.SEQ_LEN


def test_solo_piece_cannot_create_fake_missing_stem():
    record = mix()
    record['notes'] = record['notes'][::2]
    record['melody_indices'] = list(range(len(record['notes'])))
    record['drums'] = []
    with pytest.raises(NoStemWindow, match='multi-instrument'):
        prepare_example(record, random.Random(1), task='stem')


def test_drum_weight_is_twice_each_other_instrument():
    record = mix(3)
    notes = ordered([Note(*n) for n in record['notes'] + record['drums']])
    counts = Counter(stem_source(notes, random.Random(i), cfg)[0] for i in range(2000))
    assert .46 < counts['drum'] / 2000 < .54
    assert all(.21 < counts[i] / 2000 < .29 for i in ('Acoustic Piano', 'Acoustic Bass'))


def test_stem_prompt_rejects_unspecified_or_unsupported_target():
    for instrument in (None, 'not an instrument'):
        with pytest.raises(ValueError):
            make_prompt('stem', ['<S>', '<E>'], target_instrument=instrument)


def test_inference_adds_only_requested_instrument_and_keeps_input_drums():
    from improvnet.inference.melody_inference import MelodyInference
    infer = MelodyInference.__new__(MelodyInference)
    infer.tokenizer = tok = MelodyTokenizer()
    infer.meta = dict(seq_len=512, chunk_tokens=64, history_tokens=100,
                      harmony_expansion=3, max_corruptions=3)
    record = mix(40)
    anchors = ordered([Note(*n) for n in record['notes'][::2] + record['drums']])
    prompts = []
    def generate(prompt, genre, strength, instruments, window_ms, *args, **kwargs):
        prompts.append(prompt)
        assert set(instruments) == {'Acoustic Piano', 'Acoustic Bass', 'drum'}
        return [Note('Acoustic Bass', 40, 80, 0, 100), Note('drum', 36, 80, 0, 20),
                Note('Violin', 65, 80, 0, 100)]
    infer.generate = generate
    result, stats = infer.transform(anchors, 'stem', 'jazz', 1, random.Random(4),
                                    target_instrument='Acoustic Bass', history_view='mono', monophonic=True)
    assert [n for n in result if n.instrument != 'Acoustic Bass'] == anchors
    assert all(n.instrument != 'Violin' for n in result)
    assert len(stats) > 1 and stats[0]['history_tokens'] == 0
    assert all(s['history_view'] == 'poly' and not s['input_mono'] for s in stats)
    assert all(p[:2] == ['<STEM>', ('instrument', 'Acoustic Bass')] for p in prompts)
    assert any(('instrument', 'drum') in p[2:p.index('<SEP>')] for p in prompts[1:])
    assert all(s['total_token_budget'] <= 512 for s in stats)
    assert all(s['response_tokens'] <= s['response_token_budget'] for s in stats)
    with pytest.raises(ValueError, match='already present'):
        infer.transform(result, 'stem', 'jazz', 1, random.Random(4), target_instrument='Acoustic Bass')


def test_classical_draws_prefer_genuine_mixtures(tmp_path, monkeypatch):
    path = tmp_path / 'data.jsonl'
    rows = []
    with path.open('wb') as f:
        for g in ('classical', 'jazz', 'blues'):
            for multi in (True, False):
                rows.append(dict(offset=f.tell(), split='train', genre=g, source='fixture',
                                 has_harmony=True, has_multi_instrument=multi))
                f.write((json.dumps(dict(genre=g, multi=multi)) + '\n').encode())
    path.with_suffix('.index.json').write_text(json.dumps(dict(tokenizer=MelodyTokenizer().signature(), rows=rows)))
    monkeypatch.setattr('improvnet.utils.melody_data.prepare_example', lambda r, *a, **kw: r)
    ds = MelodyDataset(path, 'train', settings(TASK_WEIGHTS=(1, 0, 0, 0)))
    examples = [ds[i] for i in range(4000)]
    classical = [r for r in examples if r['genre'] == 'classical']
    assert .64 < len(classical) / len(examples) < .69
    assert .72 < sum(r['multi'] for r in classical) / len(classical) < .78
    stems = MelodyDataset(path, 'train', settings(TASK_WEIGHTS=(0, 0, 0, 1)))
    assert all(stems[i]['multi'] for i in range(100))


def test_validation_reports_missing_attributes_separately_from_copied_mix():
    from improvnet.model.melody_train import evaluate_validation
    class FixedModel(torch.nn.Module):
        def forward(self, target, **kwargs):
            return torch.tensor([[[4., 0., 0.], [0., 4., 0.], [4., 0., 0.], [4., 0., 0.]]])
    batch = dict(input=torch.zeros(1, 4, dtype=torch.long), target=torch.tensor([[0, 1, 2, -100]]),
                 task=torch.tensor([TASKS.index('stem')]), genre=torch.tensor([1]),
                 multi_hot=torch.zeros(1, 41), strength=torch.ones(1),
                 style_values=torch.zeros(1, 7), style_mask=torch.zeros(1, 7),
                 stem_mask=torch.tensor([[False, False, True, False]]),
                 stem_instrument=torch.tensor([ProcessData().INSTRUMENT_CLASSES.index('drum')]))
    metrics = evaluate_validation(FixedModel(), [batch], torch.device('cpu'))
    assert metrics['stem_target/drum'] == metrics['stem_target']
    assert metrics['stem_target'] > metrics['stem'] + 2
    assert metrics['loss'] == metrics['stem']


def test_preparation_builds_double_classical_and_verified_mixture_metadata(tmp_path):
    from improvnet.scripts.melody_prepare import read_job, make_record
    from improvnet.scripts.melody_prepare_stems import prepare_stems
    def corpus(path, records):
        rows = []
        with path.open('wb') as f:
            for i, (rec, digest) in enumerate(records):
                rows.append(dict(offset=f.tell(), sha256=digest, genre=rec['genre'], source=rec['source'],
                                 group=digest, source_group=digest, split=('train', 'validation')[i % 2],
                                 has_harmony=rec['has_harmony'], has_drums=rec['has_drums']))
                f.write((json.dumps(rec) + '\n').encode())
        # Retained jazz/blues may use the older vocabulary prefix.
        path.with_suffix('.index.json').write_text(json.dumps(dict(
            tokenizer=MelodyTokenizer().signature()[:-1], skyline_tolerance_ms=50, rows=rows)))
        return rows
    base, cache, output, index = [tmp_path / n for n in ('base.jsonl', 'cache.jsonl', 'new.jsonl', 'index.sqlite')]
    base_records, raw = [], []
    for i in range(8):
        path = tmp_path / f'{i}.mid'
        notes = [Note('Acoustic Piano', 60 + j, 90, j * (200 + 70 * i), 100) for j in range(5)]
        if i >= 4:
            notes += [Note('Acoustic Bass', 36 + j, 80, j * (200 + 70 * i), 180) for j in range(5)]
        write_midi_notes(notes, path)
        job, digest, notes, error = read_job((path, 'jazz' if i < 2 else 'classical', 'fixture', '', False))
        assert error is None
        rec, _ = make_record(job, notes)
        if i < 2:
            base_records.append((rec, digest))
        else:
            raw.append((path, rec, digest))
    old_rows = corpus(base, base_records)
    corpus(cache, [(rec, d) for _, rec, d in raw[:2]])
    with sqlite3.connect(index) as db:
        db.execute('CREATE TABLE files(path TEXT, collection TEXT, genre TEXT, group_id TEXT)')
        db.executemany('INSERT INTO files VALUES(?,?,?,?)',
                       [(str(p), 'gigamidi', 'classical', f'g{i}') for i, (p, _, _) in enumerate(raw)])
    report = prepare_stems(base, cache, index, output, ratio=2, multi_fraction=.5, workers=0)
    assert report['prepared'] == 6 and report['classical'] == 4 and report['jazz_blues'] == 2
    assert report['classical_multi_instrument'] == 2
    assert output.read_bytes().startswith(base.read_bytes())
    rows = json.loads(output.with_suffix('.index.json').read_text())['rows']
    assert all(all(r[k] == old[k] for k in old) for r, old in zip(rows, old_rows))
    assert len({r['sha256'] for r in rows}) == 6
    for row in rows:
        assert row['has_multi_instrument'] == (len(row['instruments']) >= 2)
    assert sum(r['has_multi_instrument'] for r in rows) == 2
