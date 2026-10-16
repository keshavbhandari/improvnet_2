"""Musical data invariants and a small CPU train/resume/inference integration."""
from collections import Counter
from dataclasses import astuple, replace
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
from improvnet.model.melody_model import (MelodyRefinementModel, checkpoint_metadata,
                                         initialize_from_ar, validate_checkpoint)
from improvnet.utils.melody import (CORRUPTIONS, MelodyTokenizer, Note, corrupt_melody,
                                   mask_timing, melody_voice, notes_to_tokens, read_midi_notes,
                                   skyline, tokens_to_notes, window, write_midi_notes)
from improvnet.utils.melody_data import MelodyDataset, collate_examples, prepare_example
from improvnet.utils.melody_chunks import (chunk_end, chunk_starts, history_tokens,
                                          shorten_chunk, token_count)

torch.set_num_threads(2)


def tune(**updates):
    values = {k: v for k, v in vars(cfg).items() if k.isupper()}
    if updates.get('SEQ_LEN', cfg.SEQ_LEN) == 256:
        values.update(CHUNK_TOKENS=96, HISTORY_TOKENS=64)
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
                genre='jazz', source='fixture', windows=[0, 1], harmony_windows=[0, 1],
                drums=[astuple(Note('drum', 51, 80, t, 100)) for t in range(0, 6000, 500)])


def long_record(attacks=2400):
    notes = [Note('Acoustic Piano', pitch, 90, i * 120, 600)
             for i in range(attacks) for pitch in (72 + i % 5, 60, 55, 48)]
    return dict(notes=[astuple(n) for n in notes], melody_indices=list(range(0, len(notes), 4)),
                genre='jazz', source='fixture', has_harmony=True)


def response_notes(example):
    tok = MelodyTokenizer()
    return tokens_to_notes([tok.id_to_tok[i] for i in example['target'].tolist() if i != -100], tok)


@pytest.mark.parametrize('task', ('refine', 'rhythm', 'harmonize'))
@pytest.mark.parametrize('strength', (0, .7, 1))
@pytest.mark.parametrize('history_prob', (0, 1))
def test_long_training_chunks_respect_all_budgets(task, strength, history_prob):
    rec, tok = long_record(), MelodyTokenizer()
    # Choose a late window so present histories exercise the full token cap.
    example = prepare_example(rec, random.Random(17), tune(HISTORY_PROB=history_prob, INPUT_MONO_PROB=1),
                              task=task, strength=strength)
    assert 0 < example['source_length'] <= cfg.CHUNK_TOKENS
    assert example['history_length'] <= cfg.HISTORY_TOKENS
    assert bool(example['history_length']) == bool(history_prob)
    assert len(example['input']) + 3 <= cfg.SEQ_LEN  # Full stream, including final E and conditions.
    assert example['end_ms'] > example['start_ms']
    stream = [tok.id_to_tok[i] for i in example['input'].tolist()]
    assert stream.count('<SEP>') == 2  # No right-context slot.
    history = tokens_to_notes(stream[1:1 + example['history_length']], tok)
    if history:
        # Whole clean prefix, rebased at its first retained attack.
        absolute = [Note(*n) for n in rec['notes'] if n[3] < example['start_ms']]
        if task != 'harmonize':
            absolute = [n for n in absolute if n.pitch >= 72]
        assert history[-1].onset < example['start_ms']
        origin = absolute[-1].onset - history[-1].onset
        assert history == [replace(n, onset=n.onset - origin) for n in absolute if n.onset >= origin]
    response = response_notes(example)
    reference = [Note(n[0], n[1], n[2], n[3] - example['start_ms'], n[4])
                 for n in rec['notes'] if example['start_ms'] <= n[3] < example['end_ms']
                 and (task == 'harmonize' or n[1] >= 72)]
    assert response == reference
    if task != 'harmonize':
        assert example['response_length'] < cfg.CHUNK_TOKENS
    if strength == 0 and task != 'harmonize':
        assert example['source_length'] > 450
        if history_prob:
            assert len(example['input']) > 2900


def test_history_probability_and_short_piece_fallback():
    # A small piece fitting one chunk must still learn left-history conditioning.
    lengths = [prepare_example(record(), random.Random(i), task='rhythm', strength=0)['history_length']
               for i in range(100)]
    assert 35 <= sum(bool(n) for n in lengths) <= 65
    assert all(n <= cfg.HISTORY_TOKENS for n in lengths)


def test_chunking_keeps_whole_chords_and_bounded_history():
    tok = MelodyTokenizer()
    notes = [Note(*n) for n in long_record(300)['notes']]
    starts = chunk_starts(notes, 600)
    chunks = []
    for start in starts:
        stop = chunk_end(notes, start, 600)
        assert start % 4 == stop % 4 == 0
        local = [replace(n, onset=n.onset - notes[start].onset) for n in notes[start:stop]]
        assert token_count(notes, start, stop) == len(notes_to_tokens(local, tok)) <= 600
        chunks.extend(notes[start:stop])
    assert chunks == notes
    # A long preceding rest must not consume the history budget as leading T's.
    late = [replace(n, onset=n.onset + 1_000_000) for n in notes]
    history = history_tokens(late, 2048, tok)
    assert len(history) <= 2048 and history[1] != '<T>'
    decoded = tokens_to_notes(history, tok)
    assert len(decoded) % 4 == 0
    assert decoded[0].onset == 0 and decoded[-1].pitch == notes[-1].pitch
    # Never spin forever if the last remaining note/group cannot fit.
    with pytest.raises(ValueError, match='complete onset group'):
        shorten_chunk(notes[:1], 0, 1, .5)
    with pytest.raises(ValueError, match='complete onset group'):
        shorten_chunk(notes[:4], 0, 4, .5)


@pytest.mark.parametrize('task', ('refine', 'rhythm', 'harmonize'))
@pytest.mark.parametrize('history_view', ('match', 'mono', 'poly'))
def test_inference_history_uses_only_generated_output(task, history_view):
    from improvnet.inference.melody_inference import MelodyInference
    inference = MelodyInference.__new__(MelodyInference)
    inference.tokenizer = tok = MelodyTokenizer()
    inference.meta = dict(seq_len=256, chunk_tokens=64, history_tokens=42,
                          harmony_expansion=3., max_corruptions=3)
    melody = [Note('Acoustic Piano', 60 + i % 12, 90, i * 200, 500) for i in range(48)]
    calls = []
    def fake_generate(prompt, genre, strength, instruments, window_ms, *args, **kwargs):
        calls.append(prompt)
        source_start = prompt.index('<SEP>') + 1
        core = tokens_to_notes(prompt[source_start:-2], tok)
        # Distinctive generated velocity/accompaniment distinguishes history from source.
        generated = [replace(n, velocity=60) for n in core]
        if task == 'harmonize':
            generated += [replace(n, pitch=36, velocity=60) for n in core]
        return generated
    inference.generate = fake_generate
    # Keep sources readable while retaining transform's nonzero-strength path.
    from unittest.mock import patch
    with patch('improvnet.inference.melody_inference.corrupt_melody',
               lambda core, *args: notes_to_tokens(core, tok)), patch(
               'improvnet.inference.melody_inference.mask_timing',
               lambda core, *args: notes_to_tokens(core, tok)):
        result, stats = inference.transform(melody, task, 'jazz', .5, random.Random(2),
                                           history_view=history_view)
    assert len(calls) > 2 and len(result) == len(melody) * (2 if task == 'harmonize' else 1)
    assert stats[0]['history_tokens'] == 0 and calls[0][1] == '<SEP>'
    expected_view = 'poly' if task == 'harmonize' else 'mono' if history_view == 'match' else history_view
    assert all(s['history_view'] == expected_view for s in stats)
    for prompt, stat in zip(calls[1:], stats[1:]):
        history = tokens_to_notes(prompt[1:prompt.index('<SEP>')], tok)
        assert history and len(prompt[1:prompt.index('<SEP>')]) <= 42
        assert any(n.velocity == 60 for n in history)
        assert all(n.velocity in ((60, 90) if task == 'harmonize' else (60,)) for n in history)
        previous = [n for n in result if n.onset < stat['start_ms']]
        assert notes_to_tokens(history, tok) == history_tokens(previous, 42, tok)
        if task == 'harmonize':
            assert any(n.pitch == 36 for n in history)  # Even a global mono option keeps harmony voices.
    assert all(s['total_token_budget'] <= 256 for s in stats)
    assert sum(s['input_notes'] for s in stats) == len(melody)


def test_expanding_corruption_shrinks_complete_training_chunk(monkeypatch):
    def grow(core, *args):
        return notes_to_tokens(core + [replace(n, pitch=n.pitch - 12) for n in core])
    monkeypatch.setattr('improvnet.utils.melody_data.corrupt_melody', grow)
    example = prepare_example(long_record(), random.Random(1), task='refine', strength=1)
    assert example['source_length'] <= cfg.CHUNK_TOKENS
    assert 100 < example['response_length'] < cfg.CHUNK_TOKENS * .6
    assert len(example['input']) + 3 <= cfg.SEQ_LEN


@pytest.mark.parametrize('name', ('note_modification', 'fragmentation'))
def test_added_or_removed_source_notes_keep_training_target_complete(name, monkeypatch):
    notes = [Note('Acoustic Piano', 72, 90, i * 1000, 2000) for i in range(1200)]
    rec = dict(notes=[astuple(n) for n in notes], melody_indices=list(range(len(notes))),
               genre='jazz', source='weimar', has_harmony=False)
    def corrupt(core, strength, rng, tokenizer, *args):
        return corrupt_melody(core, strength, rng, tokenizer, names=[name])
    monkeypatch.setattr('improvnet.utils.melody_data.corrupt_melody', corrupt)
    example = prepare_example(rec, random.Random(1), task='refine', strength=1)
    assert len(example['input']) + 3 <= cfg.SEQ_LEN
    assert example['source_length'] <= cfg.CHUNK_TOKENS
    expected = window(notes, example['start_ms'], example['end_ms'])
    assert response_notes(example) == expected
    if name == 'note_modification':
        assert example['source_length'] > example['response_length'] * 1.8
    else:
        assert example['source_length'] == 2 and len(expected) > 100


@pytest.mark.parametrize('name', ('note_modification', 'fragmentation'))
def test_inference_advances_by_clean_input_when_corruption_changes_count(name, monkeypatch):
    from improvnet.inference.melody_inference import MelodyInference
    inference = MelodyInference.__new__(MelodyInference)
    inference.tokenizer = tok = MelodyTokenizer()
    inference.meta = dict(seq_len=8192, chunk_tokens=3000, history_tokens=2048,
                          harmony_expansion=3., max_corruptions=3)
    notes = [Note('Acoustic Piano', 72, 90, i * 1000, 2000) for i in range(1500)]
    def corrupt(core, strength, rng, tokenizer, *args):
        return corrupt_melody(core, strength, rng, tokenizer, names=[name])
    monkeypatch.setattr('improvnet.inference.melody_inference.corrupt_melody', corrupt)
    def generate(prompt, genre, strength, instruments, window_ms, *args, **kwargs):
        assert len(prompt) + kwargs['max_new_tokens'] + 2 <= 8192
        return [Note('Acoustic Piano', 65, 60, window_ms - 1000, 100)]
    inference.generate = generate
    result, stats = inference.transform(notes, 'refine', 'jazz', 1, random.Random(1))
    assert len(result) == len(stats) >= 3
    assert sum(s['input_notes'] for s in stats) == len(notes)
    assert stats[0]['start_ms'] == 0
    for previous, current in zip(stats, stats[1:]):
        assert previous['end_ms'] == current['start_ms']
        assert current['start_ms'] == previous['start_ms'] + previous['input_notes'] * 1000
    assert stats[-1]['start_ms'] + stats[-1]['input_notes'] * 1000 == len(notes) * 1000
    assert all(s['source_tokens'] <= 3000 and s['total_token_budget'] <= 8192 for s in stats)
    if name == 'fragmentation':
        assert all(s['source_tokens'] == 2 for s in stats)


@pytest.mark.parametrize('spacing', (200, 1000))
def test_early_refinement_end_continues_unprocessed_music(monkeypatch, spacing):
    from improvnet.inference.melody_inference import MelodyInference
    inference = MelodyInference.__new__(MelodyInference)
    inference.tokenizer = tok = MelodyTokenizer()
    inference.meta = dict(seq_len=8192, chunk_tokens=3000, history_tokens=2048,
                          harmony_expansion=3., max_corruptions=3)
    melody = [Note('Acoustic Piano', 60 + i % 12, 90, i * spacing, 200) for i in range(20 if spacing == 200 else 100)]
    monkeypatch.setattr('improvnet.inference.melody_inference.corrupt_melody',
                        lambda core, *args: notes_to_tokens(core, tok))
    prompts = []
    def generate(prompt, *args, **kwargs):
        prompts.append(prompt)
        core = tokens_to_notes(prompt[prompt.index('<SEP>') + 1:-2], tok)
        # Reproduce premature <E> after a short prefix of a long input.
        if len(prompts) == 1:
            core = core[:8]
        return [replace(n, velocity=60) for n in core]
    inference.generate = generate
    result, stats = inference.transform(melody, 'refine', 'jazz', .2, random.Random(42))
    assert [n.onset for n in result] == [n.onset for n in melody]
    assert all(n.velocity == 60 for n in result)  # No copying of the skipped classical tail.
    assert len(stats) == 2 and stats[0]['early_end'] and not stats[1]['early_end']
    assert stats[0]['consumed_input_notes'] == 8 and stats[0]['advance_to_ms'] == 8 * spacing
    assert stats[1]['start_ms'] == 8 * spacing and stats[1]['history_tokens'] > 0
    assert sum(s['consumed_input_notes'] for s in stats) == len(melody)
    assert all(s['total_token_budget'] <= 8192 for s in stats)
    history = tokens_to_notes(prompts[1][1:prompts[1].index('<SEP>')], tok)
    assert history == result[:8]


def test_empty_refinement_cannot_silently_discard_input():
    from improvnet.inference.melody_inference import MelodyInference
    inference = MelodyInference.__new__(MelodyInference)
    inference.tokenizer = MelodyTokenizer()
    inference.meta = dict(seq_len=256, chunk_tokens=64, history_tokens=42,
                          harmony_expansion=3., max_corruptions=3)
    inference.generate = lambda *args, **kwargs: []
    with pytest.raises(RuntimeError, match='Empty refinement response'):
        inference.transform(performance(), 'refine', 'jazz', .2, random.Random(42))


def test_fixed_timing_generation_across_multiple_clock_segments():
    from improvnet.inference.melody_inference import MelodyInference
    from improvnet.utils.melody import make_prompt
    inference = MelodyInference.__new__(MelodyInference)
    inference.tokenizer = tok = MelodyTokenizer()
    inference.meta = dict(seq_len=256, genres=list(cfg.GENRES))
    inference.device = torch.device('cpu')
    inference.model = MelodyRefinementModel(**tiny_kwargs()).eval()
    from improvnet.utils.ar_utils import ProcessData
    inference.processor = ProcessData()
    for times in ((0,), (0, 10, 4990, 5000, 14990, 20000)):
        notes = [Note('Acoustic Piano', 60 + i, 90, t, 10) for i, t in enumerate(times)]
        prompt = make_prompt('rhythm', mask_timing(notes, 1, random.Random(1), tok))
        result = inference.generate(prompt, 'jazz', 1, ['Acoustic Piano'], times[-1] + 10,
                                    temperature=0, fixed_notes=notes, monophonic=True,
                                    max_new_tokens=len(notes_to_tokens(notes, tok)) - 1)
        assert [(n.pitch, n.velocity) for n in result] == [(n.pitch, n.velocity) for n in notes]
        assert [n.onset // 5000 for n in result] == [n.onset // 5000 for n in notes]
        assert all(a.onset < b.onset for a, b in zip(result, result[1:]))
        assert result[-1].onset < times[-1] + 10


@pytest.mark.parametrize('prefer_clock', (False, True))
@pytest.mark.parametrize('budget', (5, 6, 9, 11, 13, 14, 64))
def test_note_adding_generation_finishes_inside_token_budget(budget, prefer_clock):
    from improvnet.inference.melody_inference import MelodyInference
    from improvnet.utils.ar_utils import ProcessData
    from improvnet.utils.melody import make_prompt
    inference = MelodyInference.__new__(MelodyInference)
    inference.tokenizer = tok = MelodyTokenizer()
    inference.meta = dict(seq_len=256, genres=list(cfg.GENRES))
    inference.device = torch.device('cpu')
    inference.processor = ProcessData()
    trace = []
    def eager_notes(ids, **kwargs):
        trace.append(ids[0].tolist())
        logits = torch.zeros(1, ids.shape[1], tok.vocab_size)
        logits[..., tok.tok_to_id[('instrument', 'Acoustic Piano')]] = 100
        if prefer_clock:
            logits[..., tok.tok_to_id['<T>']] = 200
        return logits, ()
    inference.model = eager_notes
    # Complete fragmentation: no notes remain in the source, but the response
    # may add notes. The test model always requests more notes/time markers.
    prompt = make_prompt('refine', ['<S>', '<E>'])
    result = inference.generate(prompt, 'jazz', 1, ['Acoustic Piano'], 1_000_000,
                                temperature=0, monophonic=True, max_new_tokens=budget)
    response = ['<S>'] + [tok.id_to_tok[ids[0]] for ids in trace[1:]] + ['<E>']
    assert len(response) - 1 <= budget
    assert len(prompt) + len(response) - 1 + 2 <= 256
    assert tokens_to_notes(response, tok) == result and result
    assert response[-2][0] == 'dur'  # Complete notes, no truncation of attributes.


def test_checkpoint_records_long_chunk_protocol():
    model = MelodyRefinementModel(**tiny_kwargs())
    meta = checkpoint_metadata(model, cfg)
    assert validate_checkpoint({'refinement': meta}) == meta
    assert (meta['version'], meta['chunk_tokens'], meta['history_tokens'], meta['history_prob']) == (4, 512, 2048, .5)
    assert meta['harmony_history_view'] == 'poly'
    assert meta['history_policy'] == 'clean_target_view' and meta['stem_history_view'] == 'full_mix'
    assert 'history_mono_prob' not in meta
    with pytest.raises(ValueError, match='Resume settings differ'):
        validate_checkpoint({'refinement': dict(meta, history_policy='independent')}, expected=meta)
    assert 'context_windows' not in meta and 'window_ms' not in meta
    with pytest.raises(ValueError, match='Old refinement protocol'):
        validate_checkpoint({'refinement': dict(meta, version=1)})


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
    assert tok.vocab_size == base.vocab_size + 5
    assert tok.vocab[base.vocab_size:] == ('<REFINE>', '<RHYTHM>', '<HARMONIZE>', '<DRUMS>', '<STEM>')
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
    x = prepare_example(rec, random.Random(1), tune(HISTORY_PROB=0, INPUT_MONO_PROB=1), task='rhythm', strength=0)
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
                rows.append(dict(offset=handle.tell(), split=split, genre='jazz', source='fixture', has_harmony=True, has_drums=True, instruments=['Acoustic Piano', 'drum'], has_multi_instrument=True, sha256=f'{i:064x}'))
                handle.write((json.dumps(row) + '\n').encode())
    path.with_suffix('.index.json').write_text(json.dumps(dict(
        version=3, tokenizer=MelodyTokenizer().signature(), rows=rows)))


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
        '--rhythm-strength', '1', '--temperature', '0', '--device', 'cpu', '--no-style-conditioning'])
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
