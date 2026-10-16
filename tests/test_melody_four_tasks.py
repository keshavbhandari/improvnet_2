"""Four-task supervision, aggregate controls, polyphonic timing and drum addition."""
from collections import Counter
from dataclasses import astuple, replace
import json
import random
import sys
from types import SimpleNamespace

import pytest
import torch

from improvnet.model import melody_config as cfg
from improvnet.model.melody_model import MelodyRefinementModel, checkpoint_metadata, initialize_from_ar
from improvnet.utils.melody import (Note, MelodyTokenizer, TASKS, make_prompt, mask_timing,
                                   melody_voice, notes_to_tokens, tokens_to_notes, read_midi_notes, write_midi_notes)
from improvnet.utils.melody_data import prepare_example
from improvnet.utils.melody_chunks import history_tokens
from improvnet.utils.melody_style import STYLE_FEATURES, performance_statistics, style_tensors, choose_reference


def settings(**updates):
    return SimpleNamespace(**dict({k: v for k, v in vars(cfg).items() if k.isupper()}, **updates))


def arrangement(count=450):
    notes = [Note('Acoustic Piano', p, 90, i * 200, 600)
             for i in range(count) for p in (76 + i % 5, 60, 48)]
    drums = [Note('drum', 42 if j % 2 else 51, 80, i * 200 + j * 20, 20)
             for i in range(count) for j in range(6)]
    return dict(notes=[astuple(n) for n in notes], melody_indices=list(range(0, len(notes), 3)),
                drums=[astuple(n) for n in drums], has_harmony=True, has_drums=True,
                genre='jazz', source='fixture', path='fixture.mid')


def decoded_response(example):
    tok = MelodyTokenizer()
    return tokens_to_notes([tok.id_to_tok[i] for i in example['target'].tolist() if i != -100], tok)


@pytest.mark.parametrize('task', TASKS)
@pytest.mark.parametrize('mono', (0, 1))
@pytest.mark.parametrize('history_prob', (0, 1))
def test_target_matched_histories_with_complete_bounded_responses(task, mono, history_prob):
    record = arrangement()
    x = prepare_example(record, random.Random(17), settings(
        INPUT_MONO_PROB=mono, HISTORY_PROB=history_prob,
        HARMONY_STYLE_PROB=1, RHYTHM_STYLE_PROB=1), task, .4)
    expected_history_mono = bool(mono) and task != 'harmonize'
    assert x['input_mono'] == bool(mono) and x['history_mono'] == expected_history_mono
    assert 0 < x['source_length'] <= 512 and x['history_length'] <= 2048
    assert len(x['input']) + 3 <= 8192
    if task != 'drums':
        assert x['response_length'] + 1 <= 512
    tok = MelodyTokenizer()
    stream = [tok.id_to_tok[i] for i in x['input'].tolist()]
    history = tokens_to_notes(stream[1:1 + x['history_length']], tok)
    assert bool(history) == bool(history_prob)
    clean = [Note(*n) for n in record['notes']]
    if expected_history_mono:
        clean = melody_voice([clean[i] for i in record['melody_indices']])
    if task == 'drums':
        clean += [Note(*n) for n in record['drums']]
    preceding = [n for n in clean if n.onset < x['start_ms']]
    expected_history = history_tokens(preceding, 2048, tok) if history_prob else []
    assert stream[1:1 + x['history_length']] == expected_history
    if history:
        pitched_history = [n for n in history if n.instrument != 'drum']
        simultaneous = Counter(n.onset for n in pitched_history)
        assert (max(simultaneous.values()) == 1) == expected_history_mono
        assert any(n.instrument == 'drum' for n in history) == (task == 'drums')
    response = decoded_response(x)
    pitched = [n for n in response if n.instrument != 'drum']
    assert any(n.instrument == 'drum' for n in response) == (task == 'drums')
    if task in ('refine', 'rhythm', 'drums'):
        assert (max(Counter(n.onset for n in pitched).values()) == 1) == bool(mono)
    else:
        assert max(Counter(n.onset for n in pitched).values()) == 3
    assert x['style_mask'].sum() > 0 if task in ('rhythm', 'harmonize') else x['style_mask'].sum() == 0


def test_drum_response_can_exceed_512_and_transposition_does_not_change_drum_pitches():
    record = arrangement()
    record['_augment'] = True
    x = prepare_example(record, random.Random(5), settings(INPUT_MONO_PROB=1), 'drums')
    response = decoded_response(x)
    assert x['source_length'] <= 512 < x['response_length'] and len(x['input']) + 3 <= 8192
    assert {n.pitch for n in response if n.instrument == 'drum'} <= {42, 51}


@pytest.mark.parametrize('mono', (False, True))
def test_refinement_history_follows_target_even_when_corruption_changes_polyphony(mono, monkeypatch):
    tok = MelodyTokenizer()
    def corrupt(core, *args):
        changed = core + [replace(n, pitch=n.pitch - 12) for n in core] if mono else melody_voice(core)
        return notes_to_tokens(changed, tok)
    monkeypatch.setattr('improvnet.utils.melody_data.corrupt_melody', corrupt)
    x = prepare_example(arrangement(), random.Random(17), settings(
        INPUT_MONO_PROB=int(mono), HISTORY_PROB=1), 'refine', 1)
    stream = [tok.id_to_tok[i] for i in x['input'].tolist()]
    history = tokens_to_notes(stream[1:1 + x['history_length']], tok)
    source_start = stream.index('<SEP>') + 1
    source = tokens_to_notes(stream[source_start:source_start + x['source_length']], tok)
    assert max(Counter(n.onset for n in source).values()) == (2 if mono else 1)
    assert max(Counter(n.onset for n in history).values()) == (1 if mono else 3)
    assert max(Counter(n.onset for n in decoded_response(x)).values()) == (1 if mono else 3)


def test_dense_drum_supervision_shrinks_input_without_truncating_the_mix():
    record = arrangement(200)
    record['drums'] = [astuple(Note('drum', p, 80, t, 10))
                       for t in range(0, 40000, 10) for p in (36, 42, 51)]
    x = prepare_example(record, random.Random(5), settings(
        INPUT_MONO_PROB=1, HISTORY_PROB=1), 'drums')
    assert x['source_length'] < 512 < x['response_length'] and len(x['input']) + 3 <= 8192
    drums = [n for n in decoded_response(x) if n.instrument == 'drum']
    expected = [Note(n[0], n[1], n[2], n[3] - x['start_ms'], n[4]) for n in record['drums']
                if x['start_ms'] <= n[3] < x['end_ms']]
    assert Counter(drums) == Counter(expected)


@pytest.mark.parametrize('task', ('rhythm', 'harmonize'))
def test_condition_dropout_changes_controls_only(task):
    absent = prepare_example(arrangement(), random.Random(18), settings(
        HARMONY_STYLE_PROB=0, RHYTHM_STYLE_PROB=0), task, .5)
    present = prepare_example(arrangement(), random.Random(18), settings(
        HARMONY_STYLE_PROB=1, RHYTHM_STYLE_PROB=1), task, .5)
    assert torch.equal(absent['input'], present['input'])
    assert torch.equal(absent['target'], present['target'])
    assert absent['style_mask'].sum() == 0 and present['style_mask'].sum() > 0
    allowed = {0, 1} if task == 'harmonize' else set(range(2, 7))
    assert set(present['style_mask'].nonzero().flatten().tolist()) <= allowed


def test_statistics_count_performed_groups_and_held_accompaniment():
    melody = [Note('Acoustic Piano', 80, 90, t, 100) for t in (0, 1000, 2000)]
    accompaniment = [Note('Acoustic Piano', p, 70, t, 1500)
                     for t, pitches in ((0, (48, 52, 55)), (1000, (50, 53, 57))) for p in pitches]
    # Independent value-equal note objects still identify the melody correctly.
    stats = performance_statistics(melody + accompaniment, [replace(n) for n in melody], 3000)
    assert stats['accompaniment_attacks_per_second'] == pytest.approx(2 / 3)
    assert stats['accompaniment_voicing_density'] == 4.5
    assert stats['ioi_mean_ms'] == stats['ioi_q25_ms'] == stats['ioi_q75_ms'] == 1000
    assert stats['ioi_std_ms'] == 0
    assert set(stats) <= set(STYLE_FEATURES)
    carried = [Note('Acoustic Piano', p, 70, -500, 2500) for p in (40, 43, 47)]
    current = performance_statistics(melody[1:] + accompaniment[3:], melody[1:], 2000,
                                     accompaniment_context=carried)
    assert current['accompaniment_voicing_density'] == 6
    assert current['accompaniment_attacks_per_second'] == .5  # Held notes are not new attacks.
    bad = dict(stats, ioi_mean_ms=float('nan'))
    with pytest.raises(ValueError, match='Invalid style feature'):
        style_tensors(bad, 'rhythm')


def tiny_kwargs():
    return dict(vocab_size=MelodyTokenizer().vocab_size, embed_dim=32, n_heads=4,
                n_kv_heads=2, n_layers=1, dropout=0, num_genres=4, num_instruments=41)


def test_style_controls_are_cached_and_hidden_values_cannot_leak():
    torch.manual_seed(12)
    model = MelodyRefinementModel(**tiny_kwargs()).eval()
    model.style_proj[-1].weight.data.normal_(std=.05)
    ids = torch.tensor([[0, 20, 30, 1]])
    values, mask = style_tensors({'ioi_mean_ms': 300, 'ioi_std_ms': 100}, 'rhythm')
    args = dict(strength=torch.tensor([.5]), style_values=values[None], style_mask=mask[None])
    with torch.no_grad():
        whole = model(ids, **args)
        _, cache = model(ids[:, :3], use_cache=True, **args)
        suffix, _ = model(ids[:, 3:], past_key_values=cache, seq_offset=5, use_cache=True)
        assert torch.allclose(whole[:, -1:], suffix, atol=1e-6)
        no_style = model(ids, strength=torch.tensor([.5]))
        hidden = model(ids, strength=torch.tensor([.5]), style_values=values[None] * 9,
                       style_mask=torch.zeros_like(mask[None]))
        assert torch.equal(no_style, hidden) and not torch.allclose(whole, hidden)


def test_old_three_task_weights_can_initialize_four_task_model():
    original = MelodyRefinementModel(**tiny_kwargs())
    old = {k: v.clone() for k, v in original.state_dict().items() if not k.startswith('style_proj.')}
    old['token_emb.weight'] = old['token_emb.weight'][:-1]
    old['lm_head.weight'] = old['lm_head.weight'][:-1]
    model = MelodyRefinementModel(**tiny_kwargs())
    initialize_from_ar(model, {'model_state_dict': old, 'refinement': {
        'version': 2, 'tokenizer': MelodyTokenizer().signature()[:-1]}})
    assert torch.equal(model.token_emb.weight[:-1], old['token_emb.weight'])
    assert model.style_proj[-1].weight.count_nonzero() == 0


def test_old_resume_is_rejected_before_loading_data_or_allocating_model(tmp_path, monkeypatch):
    from improvnet.model import melody_train
    torch.save({'refinement': {'version': 2}}, tmp_path / 'latest_checkpoint.pt')
    def forbidden(*args, **kwargs):
        pytest.fail('An old protocol must fail before loading data/allocating the model')
    monkeypatch.setattr(melody_train, 'build_dataloader', forbidden)
    monkeypatch.setattr(melody_train, 'MelodyRefinementModel', forbidden)
    with pytest.raises(ValueError, match='Old refinement protocol'):
        melody_train.train(settings(SAVE_DIR=tmp_path), n_steps=1)


def test_polyphonic_timing_preserves_chord_attacks():
    from improvnet.inference.melody_inference import MelodyInference
    from improvnet.utils.ar_utils import ProcessData
    inference = MelodyInference.__new__(MelodyInference)
    inference.tokenizer = tok = MelodyTokenizer()
    inference.meta = dict(seq_len=256, genres=cfg.GENRES)
    inference.device = torch.device('cpu')
    inference.model = MelodyRefinementModel(**tiny_kwargs()).eval()
    inference.processor = ProcessData()
    notes = [Note('Acoustic Piano', p, 90, t, 10) for t in (0, 100, 5000) for p in (72, 60, 48)]
    prompt = make_prompt('rhythm', mask_timing(notes, 1, random.Random(1), tok))
    result = inference.generate(prompt, 'jazz', 1, ['Acoustic Piano'], 5010,
                                temperature=0, fixed_notes=notes, monophonic=False,
                                max_new_tokens=len(notes_to_tokens(notes, tok)) - 1)
    assert [(n.pitch, n.velocity) for n in result] == [(n.pitch, n.velocity) for n in notes]
    assert sorted(Counter(n.onset for n in result).values()) == [3, 3, 3]
    assert [n.onset // 5000 for n in result] == [n.onset // 5000 for n in notes]


def test_drum_generation_preserves_every_input_note_and_bounds_complete_mix():
    from improvnet.inference.melody_inference import MelodyInference
    inference = MelodyInference.__new__(MelodyInference)
    inference.tokenizer = MelodyTokenizer()
    inference.meta = dict(seq_len=8192, chunk_tokens=512, history_tokens=2048,
                          harmony_expansion=3., max_corruptions=3)
    anchors = [Note('Acoustic Piano', p, 90, i * 200, 200)
               for i in range(100) for p in (72, 60, 48)]
    def generate(prompt, genre, strength, instruments, window_ms, *args, **kwargs):
        # A bad decoder also invents piano: the drum task must discard it.
        return [Note('Acoustic Piano', 65, 40, 0, 10)] + [
            Note('drum', p, 80, t, 10) for t in range(0, window_ms, 10) for p in (36, 42, 51)]
    inference.generate = generate
    result, stats = inference.transform(anchors, 'drums', 'jazz', 1, random.Random(1), monophonic=False)
    assert [n for n in result if n.instrument != 'drum'] == anchors
    assert any(n.instrument == 'drum' for n in result)
    assert all(s['total_token_budget'] <= 8192 and s['source_tokens'] <= 512 for s in stats)
    assert all(s['response_tokens'] <= s['response_token_budget'] for s in stats)
    assert sum(s['consumed_input_notes'] for s in stats) == len(anchors)


@pytest.mark.parametrize('mono', (False, True))
def test_drum_inference_history_matches_pitched_target_and_retains_generated_drums(mono):
    from improvnet.inference.melody_inference import MelodyInference
    inference = MelodyInference.__new__(MelodyInference)
    inference.tokenizer = tok = MelodyTokenizer()
    inference.meta = dict(seq_len=512, chunk_tokens=64, history_tokens=100,
                          harmony_expansion=3., max_corruptions=3)
    anchors = [Note('Acoustic Piano', p + (i % 5 if mono else 0), 90, i * (20 if mono else 200), 100)
               for i in range(30) for p in ((72,) if mono else (72, 60, 48))]
    prompts = []
    def generate(prompt, *args, **kwargs):
        prompts.append(prompt)
        core = tokens_to_notes(prompt[prompt.index('<SEP>') + 1:-2], tok)
        return core + [Note('drum', 42, 80, t, 20) for t in sorted({n.onset for n in core})]
    inference.generate = generate
    result, stats = inference.transform(anchors, 'drums', 'jazz', 1, random.Random(1), monophonic=mono)
    assert len(stats) > 1 and stats[0]['history_tokens'] == 0
    for prompt, stat in zip(prompts[1:], stats[1:]):
        history = prompt[1:prompt.index('<SEP>')]
        previous = [n for n in result if n.onset < stat['start_ms']]
        assert history == history_tokens(previous, 100, tok)
        decoded = tokens_to_notes(history, tok)
        assert any(n.instrument == 'drum' for n in decoded)
        pitched = [n for n in decoded if n.instrument != 'drum']
        assert max(Counter(n.onset for n in pitched).values()) == (1 if mono else 3)
        assert stat['history_view'] == ('mono' if mono else 'poly')


def test_drum_preparation_and_roundtrip(tmp_path):
    from improvnet.scripts.melody_prepare import prepare
    root = tmp_path / 'raw' / 'doug_mcenzie_jazz'
    root.mkdir(parents=True)
    record = arrangement(8)
    notes = [Note(*n) for n in record['notes'] + record['drums']]
    path = root / 'with_drums.mid'
    write_midi_notes(notes, path)
    restored = read_midi_notes(path, include_drums=True)
    assert Counter((n.instrument, n.pitch, n.velocity, n.onset) for n in restored) == Counter(
        (n.instrument, n.pitch, n.velocity, n.onset) for n in notes)
    assert Counter(n for n in restored if n.instrument == 'drum') == Counter(n for n in notes if n.instrument == 'drum')
    assert all(n.instrument != 'drum' for n in read_midi_notes(path))
    output = tmp_path / 'data.jsonl'
    prepare(root.parent, output)
    row = json.loads(output.read_text())
    index = json.loads(output.with_suffix('.index.json').read_text())
    assert row['drums'] and row['has_drums'] and index['rows'][0]['has_drums']
    assert all(n[0] != 'drum' for n in row['notes'])


def test_references_are_train_only_and_seeded(tmp_path):
    from improvnet.scripts.melody_style_references import build_references
    data = tmp_path / 'data.jsonl'
    rows, record = [], arrangement(10)
    with data.open('wb') as f:
        for i, split in enumerate(('train', 'validation', 'test')):
            rows.append(dict(offset=f.tell(), split=split, source='fixture', genre='jazz', sha256=f'{i:064x}'))
            f.write((json.dumps(record) + '\n').encode())
    data.with_suffix('.index.json').write_text(json.dumps({'rows': rows}))
    path = tmp_path / 'styles.json'
    bank = build_references(data, path)
    assert len(bank['examples']) == 1 and bank['examples'][0]['split'] == 'train'
    assert 'notes' not in bank['examples'][0]
    a, b = (choose_reference(path, 'jazz', random.Random(1)) for _ in range(2))
    assert a == b


def test_reordered_pipeline_with_poly_input_and_midi_reference(tmp_path, monkeypatch):
    from improvnet.inference.melody_inference import main
    config = settings(SEQ_LEN=256, CHUNK_TOKENS=96, HISTORY_TOKENS=64)
    model = MelodyRefinementModel(**tiny_kwargs())
    checkpoint = tmp_path / 'model.pt'
    torch.save(dict(model_state_dict=model.state_dict(), refinement=checkpoint_metadata(model, config), step=10), checkpoint)
    record = arrangement(4)
    source, output = tmp_path / 'input.mid', tmp_path / 'mix.mid'
    write_midi_notes([Note(*n) for n in record['notes']], source)
    monkeypatch.setattr(sys, 'argv', ['melody_inference', '--model', str(checkpoint),
        '--input', str(source), '--output', str(output), '--input-view', 'poly',
        '--pipeline-order', 'harmonize,rhythm,drums', '--style-reference', str(source),
        '--device', 'cpu', '--temperature', '0', '--no-progress'])
    main()
    report = json.loads(output.with_suffix('.json').read_text())
    assert report['task_order'] == ['harmonize', 'rhythm', 'drums']
    assert report['style_reference']['id'] == str(source)
    assert report['input_view'] == 'poly'
    assert all(c['total_token_budget'] <= 256 for c in report['chunks'])
    before_drums = read_midi_notes(output.with_suffix('.rhythm.mid'))
    after_drums = read_midi_notes(output, include_drums=True)
    assert [n for n in after_drums if n.instrument != 'drum'] == before_drums
