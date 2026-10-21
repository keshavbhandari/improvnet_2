"""Current V5 pipeline contracts."""
import random
import pytest
import torch
from copy import copy
from dataclasses import replace
from types import SimpleNamespace
from improvnet.inference.melody_inference import MelodyInference
from improvnet.inference.melody_pipeline import ForwardRequest, forward_batch, run_pipeline
from improvnet.model.melody_model import MelodyRefinementModel
from improvnet.utils.ar_utils import ProcessData
from improvnet.utils.melody import MelodyTokenizer, Note, TASKS, notes_to_tokens, ordered, tokens_to_notes
from improvnet.utils.melody_style import STYLE_FEATURES
from melody_test_helpers import settings


def tiny_model():
    torch.manual_seed(173)
    model = MelodyRefinementModel(embed_dim=32, n_heads=4, n_kv_heads=2, n_layers=2,
                                  num_genres=4, num_instruments=41, dropout=0.).eval()
    # Exercise the conditioning paths instead of their initial zero projections.
    with torch.no_grad():
        model.strength_proj[-1].weight.normal_(0, .03)
        model.style_proj[-1].weight.normal_(0, .03)
    return model


def prefill(owner, length):
    return ForwardRequest(owner, torch.randint(0, 2700, (1, length)), dict(
        genre=torch.tensor([owner % 4]), multi_hot=torch.rand(1, 41),
        strength=torch.tensor([.1 + owner*.2]), style_values=torch.rand(1, len(STYLE_FEATURES)),
        style_mask=torch.randint(0, 2, (1, len(STYLE_FEATURES))).float(), use_cache=True))


@torch.no_grad()
def test_batched_variable_prompts_and_caches_match_independent_forwards():
    model = tiny_model()
    requests = [prefill(i, n) for i, n in enumerate((3, 27, 8, 15))]
    individual = [model(r.ids, **r.kwargs) for r in requests]
    batched = forward_batch(model, requests)
    for step in range(5):
        for (single_logits, single_cache), (batch_logits, batch_cache) in zip(individual, batched):
            torch.testing.assert_close(single_logits[:, -1:], batch_logits[:, -1:], rtol=2e-5, atol=2e-6)
            for (sk, sv), (bk, bv) in zip(single_cache, batch_cache):
                torch.testing.assert_close(sk, bk, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(sv, bv, rtol=2e-5, atol=2e-6)
        # A completed row leaves, and a newly prefilled task can join a cache batch.
        if step == 1:
            individual.pop(1); batched.pop(1)
            new = prefill(1, 6)
            individual.append(model(new.ids, **new.kwargs))
            batched.extend(forward_batch(model, [new]))
        tokens = [torch.randint(0, 2700, (1, 1)) for _ in individual]
        independent_requests = [ForwardRequest(i, token, dict(use_cache=True, past_key_values=cache,
                                                               seq_offset=cache[0][0].size(1)))
                                for i, (token, (_, cache)) in enumerate(zip(tokens, individual))]
        batch_requests = [ForwardRequest(i, token, dict(use_cache=True, past_key_values=cache,
                                                        seq_offset=cache[0][0].size(1)))
                          for i, (token, (_, cache)) in enumerate(zip(tokens, batched))]
        individual = [model(r.ids, **r.kwargs) for r in independent_requests]
        batched = forward_batch(model, batch_requests)


def fixture_infer():
    infer = MelodyInference.__new__(MelodyInference)
    infer.device = torch.device('cpu')
    infer.model = tiny_model()
    infer.tokenizer = MelodyTokenizer()
    infer.processor = ProcessData()
    infer.options = settings()
    infer.meta = dict(version=9, tasks=list(TASKS), seq_len=1024, genres=['jazz'], chunk_tokens=26,
                      history_tokens=64, mix_response_tokens=104, harmonize_expansion=4,
                      skyline_tolerance_ms=50, style_features=list(STYLE_FEATURES))
    return infer


@torch.no_grad()
def fixture_generate(self, prompt, genre, strength, instruments, window_ms, **kwargs):
    """Reuse real batching/chunking, returning predictable complete musical events."""
    ids = torch.tensor([[self.tokenizer.tok_to_id[t] for t in prompt]])
    _, cache = self.model(ids, genre=torch.tensor([0]), multi_hot=torch.zeros(1, 41),
                          strength=torch.tensor([strength]), use_cache=True)
    self.model(ids[:, -1:], past_key_values=cache, seq_offset=len(prompt)+2, use_cache=True)
    first = prompt.index('<SEP>')+1
    end = prompt.index('<SEP>', first)
    source, attribute, note_index = [], 0, 0
    for token in prompt[first:end]:
        if isinstance(token, tuple) and token[0] == 'instrument' or token == '<MASK>' and attribute == 0:
            attribute = 0
        if token == '<MASK>':
            token = (('instrument', 'Acoustic Piano') if attribute == 0 else
                     ('onset', note_index*100) if attribute == 2 else ('dur', 700))
        source.append(token)
        if isinstance(token, tuple):
            attribute = (attribute+1) % 4
            if attribute == 0:
                note_index += 1
    notes = ordered(kwargs.get('fixed_notes') or tokens_to_notes(source, self.tokenizer))
    count = getattr(self, 'fixture_chunks', 0)
    self.fixture_chunks = count+1
    if prompt[0] in ('<HARMONIZE>', '<ARRANGE>'):
        notes = ordered(notes + [replace(n, pitch=n.pitch-12) for n in notes])
    if kwargs.get('stem_add') and count % 2:
        notes = ordered(notes + [Note('drum', 42, 90, 0, 20)])
    self.last_decode_stats = dict(end_reason='fixture')
    return notes


def pipeline_arguments(tmp_path):
    return SimpleNamespace(task='pipeline', genre='jazz', melody_strength=.2, rhythm_strength=0.,
                           target_instrument=None, temperature=0.)


def test_pipeline_overlaps_tasks_and_matches_offline_boundaries_history_and_notes(tmp_path, monkeypatch):
    monkeypatch.setattr(MelodyInference, 'generate', fixture_generate)
    monkeypatch.setattr('improvnet.inference.melody_inference.refinement_source',
                        lambda notes, strength, rng, tok, options, bounded: notes_to_tokens(notes, tok))
    infer = fixture_infer()
    original = [Note('Acoustic Piano', 72+i%5, 90, i*100, 700) for i in range(70)]
    requested = ['Acoustic Piano', 'drum']
    args = pipeline_arguments(tmp_path)
    parallel, logs, tasks, report = run_pipeline(infer, original, list(TASKS), args, requested,
                                                ['Acoustic Piano'], None, None, tmp_path/'parallel', ['drum'])
    current, serial_logs = original, []
    for task in TASKS:
        local = copy(infer)
        current, log = local.transform(current, task, 'jazz', .2 if task=='refine' else 0. if task=='harmonize' else 1.,
            random.Random(42), 0., instruments=['Acoustic Piano'] if task in ('refine', 'harmonize') else requested,
            monophonic=True if task=='harmonize' else None, target_instrument='drum' if task=='stem' else None)
        serial_logs.extend(log)
    assert parallel == current
    assert logs == serial_logs
    assert tasks == list(TASKS)
    assert all(log['history_tokens'] <= 64 and log['total_token_budget'] <= 1024 for log in logs)
    stats = report['parallel_inference']
    assert stats['max_batch_size'] >= 2
    assert stats['mean_decode_batch_size'] > 1
    times = stats['stage_timing']
    assert times[1]['first_forward_seconds'] < times[0]['finished_seconds']
    assert times[2]['first_forward_seconds'] < times[1]['finished_seconds']
    assert any(log['task']=='stem' and log['added_stem_notes']==0 for log in logs)
    assert any(n.instrument=='drum' for n in parallel)
    assert any(n.onset+n.duration > log['end_ms'] for log in logs for n in parallel
               if log['start_ms'] <= n.onset < log['end_ms'])
    assert all((tmp_path/'parallel'/f'{task}.mid').exists() for task in TASKS)


def test_task_failure_wakes_waiters_and_propagates_without_hanging(tmp_path, monkeypatch):
    def fail(self, *args, **kwargs):
        raise ValueError('broken decoder')
    monkeypatch.setattr(MelodyInference, 'generate', fail)
    infer = fixture_infer()
    with pytest.raises(RuntimeError, match='Parallel inference stopped') as error:
        run_pipeline(infer, [Note('Acoustic Piano', 72, 90, i*100, 80) for i in range(20)],
                     list(TASKS), pipeline_arguments(tmp_path), ['Acoustic Piano', 'drum'],
                     ['Acoustic Piano'], None, None, tmp_path/'failure', ['drum'])
    assert 'broken decoder' in str(error.value.__cause__)
    assert not list((tmp_path/'failure').glob('*.mid'))


def test_canonical_v5_cli_batches_three_tasks_and_harmonizes_requested_pitched_mix(tmp_path, monkeypatch):
    import json
    import sys
    from improvnet.inference import melody_inference as module
    from improvnet.utils.melody import write_midi_notes
    infer = fixture_infer()
    infer.meta.update(version=9, tasks=list(TASKS))
    infer.checkpoint_step = 1
    calls = []
    def generate(self, prompt, genre, strength, instruments, *args, **kwargs):
        calls.append((prompt[0], tuple(instruments), kwargs.get('protected_pitches')))
        notes = fixture_generate(self, prompt, genre, strength, instruments, *args, **kwargs)
        if prompt[0] in ('<HARMONIZE>', '<ARRANGE>'):
            notes = [replace(n, instrument=instruments[i % len(instruments)]) for i, n in enumerate(notes)]
        return ordered(notes)
    monkeypatch.setattr(MelodyInference, 'generate', generate)
    monkeypatch.setattr(module, 'MelodyInference', lambda _: infer)
    monkeypatch.setattr(module, 'refinement_source',
                        lambda notes, strength, rng, tok, options, bounded: notes_to_tokens(notes, tok))
    source, output = tmp_path/'source.mid', tmp_path/'clair.mid'
    write_midi_notes([Note('Acoustic Piano', 72+i%5, 90, i*100, 80) for i in range(70)]+[Note('drum',60,80,0,20)], source)
    monkeypatch.setattr(sys, 'argv', ['inference', '--model', 'fixture', '--input', str(source),
        '--output', str(output), '--instruments', 'Acoustic Piano', 'Electric Bass', 'Tenor Sax', 'drum',
        '--rhythm-strength', '0', '--temperature', '0', '--no-rhythm-conditioning'])
    # No custom pipeline_runner: the canonical command now batches by default.
    module.main()
    report = json.loads((tmp_path/'clair'/'report.json').read_text())
    assert report['nonkit_input_drum_notes_filtered']==1
    assert report['task_order'] == list(TASKS) == ['refine', 'harmonize', 'stem']
    assert report['parallel_inference']['max_batch_size'] >= 2
    assert {p.name for p in (tmp_path/'clair').iterdir()} == {
        'melody.mid', 'refine.mid', 'harmonize.mid', 'stem.mid', 'report.json'}
    assert {instruments for task, instruments, _ in calls if task == '<HARMONIZE>'} == {
        ('Acoustic Piano', 'Electric Bass', 'Tenor Sax')}
    assert all(protected is not None for task, _, protected in calls if task == '<ADD_STEM>')
    assert output.exists()


def test_v5_cli_rejects_retired_orchestration_before_reading_music(tmp_path, monkeypatch):
    import sys
    from improvnet.inference import melody_inference as module
    infer = fixture_infer()
    infer.meta.update(version=9, tasks=list(TASKS))
    monkeypatch.setattr(module, 'MelodyInference', lambda _: infer)
    monkeypatch.setattr(sys, 'argv', ['inference', '--model', 'fixture', '--input', 'missing.mid',
        '--output', str(tmp_path/'out.mid'), '--task', 'orchestrate'])
    with pytest.raises(SystemExit):
        module.main()
    assert not (tmp_path/'out.mid').exists()

