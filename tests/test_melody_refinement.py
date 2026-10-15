"""Musical data invariants and a small CPU train/resume/inference integration."""
from collections import Counter
from dataclasses import astuple
import json
import random
import os
import socket
import time
from types import SimpleNamespace

import pytest
import torch

from improvnet.model import melody_config as cfg
from improvnet.model.ar_model import ARContextModel
from improvnet.model.melody_model import MelodyRefinementModel, initialize_from_ar
from improvnet.utils.melody import (CORRUPTIONS, MelodyTokenizer, Note, corrupt_melody,
                                   mask_timing, melody_voice, notes_to_tokens, read_midi_notes,
                                   skyline, tokens_to_notes, window, write_midi_notes)
from improvnet.utils.melody_data import MelodyDataset, collate_examples, prepare_example

torch.set_num_threads(2)


def tune(**updates):
    values = {k: v for k, v in vars(cfg).items() if k.isupper()}
    values.update(updates)
    return SimpleNamespace(**values)


def performance():
    return [Note('Acoustic Piano', p, 90, start, dur) for p, start, dur in
            [(48, 0, 400), (60, 10, 400), (67, 30, 400), (64, 700, 800),
             (52, 700, 800), (65, 1600, 700), (53, 1600, 700),
             (67, 3000, 900), (55, 3000, 900), (69, 4900, 500),
             (57, 4900, 500), (71, 5500, 400), (59, 5500, 400)]]


def record():
    notes = sorted(performance(), key=lambda n: (n.onset, -n.pitch, n.instrument, n.duration))
    melody, _ = skyline(notes)
    selected = {id(n) for n in melody}
    return dict(notes=[astuple(n) for n in notes],
                melody_indices=[i for i, n in enumerate(notes) if id(n) in selected],
                genre='jazz', source='fixture', windows=[0, 1], harmony_windows=[0, 1])


def tiny_kwargs():
    return dict(vocab_size=MelodyTokenizer().vocab_size, embed_dim=32,
                n_heads=4, n_kv_heads=2, n_layers=1, dropout=.1,
                num_genres=4, num_instruments=41)


def test_skyline_partitions_originals_and_keeps_attacks():
    notes = performance()
    melody, harmony = skyline(notes)
    assert Counter(melody) + Counter(harmony) == Counter(notes)
    assert len(melody) + len(harmony) == len(notes)
    assert [n.pitch for n in melody[:2]] == [67, 64]
    repeated = [Note('Acoustic Piano', 60, 90, t, 10) for t in (0, 20, 40)]
    assert skyline(repeated)[0] == repeated
    fast_run = [Note('Acoustic Piano', 60 + i, 90, i * 20, 10) for i in range(3)]
    assert skyline(fast_run)[0] == fast_run


def test_window_ownership_and_midi_roundtrip(tmp_path):
    held = Note('Acoustic Piano', 60, 90, 4900, 500)
    assert window([held], 0, 5000) == [held]
    assert window([held], 5000, 10000) == []
    path = tmp_path / 'held.mid'
    write_midi_notes([held], path)
    assert read_midi_notes(path) == [held]
    repeated = [Note('Acoustic Piano', 60, 90, 0, 3000),
                Note('Acoustic Piano', 60, 90, 1000, 2000)]
    write_midi_notes(repeated, path)
    decoded = read_midi_notes(path)
    assert len(decoded) == 2 and decoded[0].duration == 1000
    assert decoded[1] == repeated[1]


def test_missing_program_change_uses_canonical_piano(tmp_path):
    import mido
    path = tmp_path / 'missing_program.mid'
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.extend([
        mido.Message('program_change', channel=1, program=40, time=0),
        mido.Message('note_on', channel=3, note=60, velocity=90, time=0),
        mido.Message('note_on', channel=1, note=64, velocity=90, time=0),
        mido.Message('note_off', channel=3, note=60, velocity=0, time=480),
        mido.Message('note_off', channel=1, note=64, velocity=0, time=0),
    ])
    midi.save(path)
    tok = MelodyTokenizer()
    notes = read_midi_notes(path, tok)
    assert {n.instrument for n in notes} == {'Acoustic Piano', 'Violin'}
    assert all(t in tok.tok_to_id for t in notes_to_tokens(notes, tok))


@pytest.mark.parametrize('task', ('refine', 'rhythm', 'harmonize'))
def test_existing_prepared_piano_alias_is_normalized(task):
    canonical = record()
    canonical['_augment'] = True
    legacy = dict(canonical, notes=[['piano', *n[1:]] for n in canonical['notes']])
    expected = prepare_example(canonical, random.Random(7), task=task, strength=.7)
    actual = prepare_example(legacy, random.Random(7), task=task, strength=.7)
    for key in ('input', 'target', 'multi_hot', 'genre', 'strength'):
        assert torch.equal(actual[key], expected[key]), key
    assert actual['multi_hot'][0] == 1 and actual['multi_hot'].sum() == 1
    assert all(n[0] == 'piano' for n in legacy['notes'])  # Raw records stay intact.


def test_vocabulary_and_timing_mask_endpoints():
    tok = MelodyTokenizer()
    from improvnet.tokenizer.absolute import AbsTokenizer
    base = AbsTokenizer()
    assert tok.vocab[:base.vocab_size] == base.vocab
    assert tok.vocab_size == base.vocab_size + 3
    notes = skyline(performance())[0]
    clean = notes_to_tokens(notes, tok)
    assert mask_timing(notes, 0, random.Random(0), tok) == clean
    masked = mask_timing(notes, 1, random.Random(0), tok)
    assert masked.count('<MASK>') == 2 * len(notes)
    for a, b in zip(clean, masked):
        if not isinstance(a, tuple) or a[0] not in ('onset', 'dur'):
            assert a == b
    assert tokens_to_notes(clean, tok) == notes


def test_harmony_preserves_held_boundary_notes():
    from improvnet.inference.melody_inference import preserve_melody
    anchor = Note('Acoustic Piano', 72, 90, 4900, 500)
    extra_attack = Note('Acoustic Piano', 72, 75, 5200, 100)
    bass = Note('Acoustic Piano', 48, 75, 5200, 100)
    assert preserve_melody([extra_attack, bass], [anchor]) == [anchor, bass]


def test_quantized_solo_ties_become_one_melody_attack():
    notes = [Note('Acoustic Piano', 60, 90, 0, 100),
             Note('Acoustic Piano', 64, 90, 0, 100),
             Note('Acoustic Piano', 62, 90, 10, 100)]
    assert [n.pitch for n in melody_voice(notes)] == [64, 62]
    rec = dict(notes=[astuple(n) for n in notes], melody_indices=[0, 1, 2],
               genre='jazz', source='weimar', windows=[0], harmony_windows=[])
    x = prepare_example(rec, random.Random(1), task='rhythm', strength=0)
    tok = MelodyTokenizer()
    response = [tok.id_to_tok[i] for i in x['target'].tolist() if i != -100]
    assert [n.pitch for n in tokens_to_notes(response, tok)] == [64, 62]


@pytest.mark.parametrize('name', CORRUPTIONS)
def test_corruptions_are_valid_and_zero_is_identity(name):
    notes = skyline(performance())[0]
    tok = MelodyTokenizer()
    assert corrupt_melody(notes, 0, random.Random(3), tok, names=[name]) == notes_to_tokens(notes, tok)
    result = corrupt_melody(notes, .8, random.Random(3), tok, names=[name])
    assert all(t in tok.tok_to_id for t in result)
    assert result[0] == '<S>' and result[-1] == '<E>'
    assert '<REFINE>' not in result  # No corruption-specific label in the music.
    if '<MASK>' not in result:
        assert all(n.duration > 0 for n in tokens_to_notes(result, tok))


def test_composed_corruptions_and_response_only_labels():
    tok = MelodyTokenizer()
    notes = skyline(performance())[0]
    corrupted = corrupt_melody(notes, 1, random.Random(3), tok,
                               names=['whole_mask', 'note_modification', 'incorrect_transposition'])
    assert '<MASK>' in corrupted and all(t in tok.tok_to_id for t in corrupted)
    for task in ('refine', 'rhythm', 'harmonize'):
        example = prepare_example(record(), random.Random(1), task=task, strength=1)
        labels = example['target']
        first = (labels != -100).nonzero()[0].item()
        assert tok.id_to_tok[example['input'][first].item()] == '<S>'
        assert labels[-1].item() == tok.tok_to_id['<E>']
        assert (labels[:first] == -100).all()
        assert len(example['input']) + 2 <= cfg.SEQ_LEN
    batch = collate_examples([prepare_example(record(), random.Random(i)) for i in range(3)])
    assert batch['input'].shape == batch['target'].shape
    assert (batch['target'][batch['input'] == cfg.PAD_ID] == -100).all()


def test_ar_migration_and_cache_alignment():
    torch.manual_seed(7)
    kwargs = tiny_kwargs()
    kwargs['dropout'] = 0
    ar_kwargs = dict(kwargs, vocab_size=MelodyTokenizer().base_vocab_size)
    base = ARContextModel(**ar_kwargs).eval()
    model = MelodyRefinementModel(**kwargs).eval()
    initialize_from_ar(model, {'model_state_dict': base.state_dict()})
    tokens = torch.tensor([[0, 20, 30, 1]])
    with torch.no_grad():
        old = base(tokens)
        new = model(tokens, strength=torch.tensor([.5]))
        assert torch.allclose(old, new[:, :, :old.shape[-1]], atol=1e-6)
        prefix, cache = model(tokens[:, :3], strength=torch.tensor([.5]), use_cache=True)
        suffix, _ = model(tokens[:, 3:], past_key_values=cache, seq_offset=5, use_cache=True)
        assert torch.allclose(new[:, -1:], suffix, atol=1e-6)
        # After learning a strength projection, changing the scalar affects logits.
        model.strength_proj[-1].weight.normal_(std=.03)
        a = model(tokens, strength=torch.tensor([0.]))
        b = model(tokens, strength=torch.tensor([1.]))
        assert not torch.allclose(a, b)


def make_data(path):
    row = record()
    rows = []
    with path.open('wb') as handle:
        for split in ('train', 'validation', 'test'):
            for i in range(6):
                rows.append(dict(offset=handle.tell(), split=split, genre='jazz', source='fixture', has_harmony=True))
                handle.write((json.dumps(row) + '\n').encode())
    path.with_suffix('.index.json').write_text(json.dumps(dict(
        tokenizer=MelodyTokenizer().signature(), window_ms=cfg.WINDOW_MS, rows=rows)))


def test_stateless_epoch_resampling(tmp_path):
    path = tmp_path / 'data.jsonl'
    make_data(path)
    ds = MelodyDataset(path, 'train')
    first = ds[0]
    assert torch.equal(first['input'], ds[0]['input'])
    ds.set_epoch(1)
    other = ds[0]
    assert first['strength'] != other['strength'] or not torch.equal(first['input'], other['input'])
    val = MelodyDataset(path, 'validation')
    a = val[0]
    val.set_epoch(10)
    assert torch.equal(a['input'], val[0]['input'])
    with pytest.raises(ValueError, match='skyline tolerance'):
        MelodyDataset(path, 'train', tune(SKYLINE_TOLERANCE_MS=100))


def test_short_pilot_keeps_the_full_schedule(tmp_path):
    from improvnet.model.melody_train import train
    path = tmp_path / 'data.jsonl'
    make_data(path)
    settings = tune(DATA_FILE=path, MODEL_KWARGS=tiny_kwargs(), NUM_WORKERS=0,
                    BATCH_SIZE=2, ACCUM_STEPS=1, INIT_CHECKPOINT=None,
                    N_STEPS=4, WARMUP_STEPS=2, DECAY_STEPS=1, VAL_EVERY=4,
                    LOG_EVERY=1, CHECKPOINT_EVERY=1, SEQ_LEN=256,
                    SAVE_DIR=str(tmp_path / 'pilot'))
    train(settings, n_steps=1)
    checkpoint = torch.load(tmp_path / 'pilot/latest_checkpoint.pt', weights_only=False)
    assert checkpoint['step'] == 1
    assert checkpoint['scheduler_state_dict']['last_epoch'] == 1


def test_cpu_training_resume_and_inference(tmp_path, monkeypatch):
    from improvnet.model import melody_train
    from improvnet.inference.melody_inference import MelodyInference, preserve_melody
    path = tmp_path / 'data.jsonl'
    make_data(path)
    settings = tune(DATA_FILE=path, MODEL_KWARGS=tiny_kwargs(), NUM_WORKERS=0,
                    BATCH_SIZE=2, ACCUM_STEPS=2, INIT_CHECKPOINT=None,
                    N_STEPS=4, WARMUP_STEPS=0, DECAY_STEPS=1, VAL_EVERY=2,
                    LOG_EVERY=1, CHECKPOINT_EVERY=1, SEQ_LEN=256,
                    SAVE_DIR=str(tmp_path / 'baseline'))
    melody_train.train(settings)
    baseline = torch.load(tmp_path / 'baseline/latest_checkpoint.pt', weights_only=False)
    settings.SAVE_DIR = str(tmp_path / 'resumed')
    calls = [0]
    def interrupt(_):
        calls[0] += 1
        return calls[0] == 2
    monkeypatch.setattr(melody_train, 'checkpoint_requested', interrupt)
    melody_train.train(settings)
    stopped = torch.load(tmp_path / 'resumed/latest_checkpoint.pt', weights_only=False)
    assert stopped['step'] == 2
    monkeypatch.setattr(melody_train, 'checkpoint_requested', lambda _: False)
    melody_train.train(settings)
    resumed = torch.load(tmp_path / 'resumed/latest_checkpoint.pt', weights_only=False)
    assert resumed['step'] == 4
    for key, value in baseline['model_state_dict'].items():
        assert torch.equal(value, resumed['model_state_dict'][key]), key
    inference = MelodyInference(tmp_path / 'resumed/latest_checkpoint.pt', 'cpu')
    melody = skyline(performance())[0]
    result, log = inference.transform(melody, 'rhythm', 'jazz', 1, random.Random(2),
                                      temperature=0, keep_pitches=True)
    assert len(result) == len(melody)
    assert [(n.instrument, n.pitch, n.velocity) for n in result] == [(n.instrument, n.pitch, n.velocity) for n in melody]
    assert all(n.duration > 0 for n in result)
    restored = preserve_melody(result, melody)
    assert all(n in restored for n in melody)
    assert log
    from improvnet.inference.melody_inference import main
    import sys
    input_path, output_path = tmp_path / 'input.mid', tmp_path / 'output.mid'
    write_midi_notes(performance(), input_path)
    monkeypatch.setattr(sys, 'argv', ['melody_inference', '--model',
        str(tmp_path / 'resumed/latest_checkpoint.pt'), '--input', str(input_path),
        '--output', str(output_path), '--melody-strength', '0',
        '--rhythm-strength', '1', '--temperature', '0', '--device', 'cpu'])
    main()
    assert read_midi_notes(output_path)
    assert output_path.with_suffix('.json').exists()
    assert output_path.with_suffix('.melody.mid').exists()
    assert output_path.with_suffix('.rhythm.mid').exists()


def test_preparation_duplicate_grouping_and_musicxml(tmp_path):
    from improvnet.scripts.melody_prepare import prepare, read_score_notes
    from music21 import stream, note, tempo
    root = tmp_path / 'raw'
    folder = root / 'doug_mcenzie_jazz'
    folder.mkdir(parents=True)
    write_midi_notes(performance(), folder / 'Original.mid')
    (folder / 'Duplicate.mid').write_bytes((folder / 'Original.mid').read_bytes())
    transposed = [Note(n.instrument, n.pitch + 1, n.velocity, n.onset, n.duration) for n in performance()]
    write_midi_notes(transposed, folder / 'Transposed.mid')
    output = tmp_path / 'prepared.jsonl'
    report = prepare(root, output)
    assert report['prepared'] == 2 and report['sources']['duplicate_files'] == 1
    rows = json.loads(output.with_suffix('.index.json').read_text())['rows']
    assert rows[0]['group'] == rows[1]['group'] and rows[0]['split'] == rows[1]['split']
    score = stream.Stream()
    score.append(tempo.MetronomeMark(number=120))
    score.append(note.Note('C4', quarterLength=1))
    xml = tmp_path / 'score.mxl'
    score.write('musicxml', fp=xml)
    parsed = read_score_notes(xml, MelodyTokenizer())
    assert [(n.pitch, n.onset, n.duration) for n in parsed] == [(60, 0, 500)]


def ddp_worker(rank, port, settings):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port),
                      WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    from improvnet.model.melody_train import train
    train(settings)


def test_two_rank_cpu_training(tmp_path):
    import torch.multiprocessing as mp
    path = tmp_path / 'data.jsonl'
    make_data(path)
    settings = tune(DATA_FILE=path, MODEL_KWARGS=tiny_kwargs(), NUM_WORKERS=0,
                    BATCH_SIZE=2, ACCUM_STEPS=2, INIT_CHECKPOINT=None,
                    N_STEPS=2, WARMUP_STEPS=0, DECAY_STEPS=1, VAL_EVERY=1,
                    LOG_EVERY=1, CHECKPOINT_EVERY=1, SEQ_LEN=256,
                    SAVE_DIR=str(tmp_path / 'ddp'))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    mp.set_forkserver_preload(['torch', 'improvnet.model.melody_train'])
    context = mp.start_processes(ddp_worker, args=(port, settings), nprocs=2,
                                 join=False, start_method='forkserver')
    try:
        deadline = time.monotonic() + 120
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail('Two-rank training timed out')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
    checkpoint = torch.load(tmp_path / 'ddp/latest_checkpoint.pt', weights_only=False)
    assert checkpoint['step'] == 2 and checkpoint['world_size'] == 2
    assert len(checkpoint['rng_state']['rank_states']) == 2
