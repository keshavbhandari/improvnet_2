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
from melody_test_helpers import settings


def tiny_model():
    torch.manual_seed(173)
    model = MelodyRefinementModel(embed_dim=32, n_heads=4, n_kv_heads=2, n_layers=2,
                                  num_genres=4, num_instruments=41, dropout=0.).eval()
    # Exercise the conditioning paths instead of their initial zero projections.
    with torch.no_grad():
        model.strength_proj[-1].weight.normal_(0, .03)
        model.style_proj[-1].weight.normal_(0, .03)
        model.beat_proj[-1].weight.normal_(0, .03)
    return model


def prefill(owner, length):
    return ForwardRequest(owner, torch.randint(0, 2700, (1, length)), dict(
        genre=torch.tensor([owner % 4]), multi_hot=torch.rand(1, 41),
        strength=torch.tensor([.1 + owner*.2]), style_embedding=torch.rand(1, 256),
        style_mask=torch.randint(0, 2, (1,)).float(), beat_embedding=torch.rand(1,256),beat_mask=torch.randint(0,2,(1,)).float(),use_cache=True))


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
    infer.meta = dict(version=11, tasks=list(TASKS), seq_len=1024, genres=['jazz'], chunk_tokens=26,
                      history_tokens=64, mix_response_tokens=104, harmonize_expansion=4,
                      skyline_tolerance_ms=50, style_dim=256)
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
                                                ['Acoustic Piano'], None, tmp_path/'parallel', ['drum'])
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
                     ['Acoustic Piano'], None, tmp_path/'failure', ['drum'])
    assert 'broken decoder' in str(error.value.__cause__)
    assert not list((tmp_path/'failure').glob('*.mid'))


@pytest.mark.parametrize('input_drum_pitch', (None,60,42))
def test_canonical_v5_cli_batches_three_tasks_and_harmonizes_requested_pitched_mix(tmp_path, monkeypatch, input_drum_pitch):
    import json
    import sys
    from improvnet.inference import melody_inference as module
    from improvnet.utils.melody import write_midi_notes
    infer = fixture_infer()
    infer.meta.update(version=11, tasks=list(TASKS))
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
    original=[Note('Acoustic Piano',72+i%5,90,i*100,80) for i in range(70)]
    if input_drum_pitch is not None:
        original.append(Note('drum',input_drum_pitch,80,0,20))
    write_midi_notes(original,source)
    monkeypatch.setattr(sys, 'argv', ['inference', '--model', 'fixture', '--input', str(source),
        '--output', str(output), '--instruments', 'Acoustic Piano', 'Electric Bass', 'Tenor Sax', 'drum',
        '--rhythm-strength', '0', '--temperature', '0', '--no-rhythm-conditioning'])
    # No custom pipeline_runner: the canonical command now batches by default.
    module.main()
    report = json.loads((tmp_path/'clair'/'report.json').read_text())
    assert report['nonkit_input_drum_notes_filtered']==int(input_drum_pitch==60)
    assert report['task_order'] == list(TASKS) == ['refine', 'harmonize', 'stem']
    assert 'drum' in report['instruments'] and not report['missing_instruments']
    assert report['stem_summary'][0]['generated_notes']>0
    assert report['parallel_inference']['max_batch_size'] >= 2
    assert {p.name for p in (tmp_path/'clair').iterdir()} == {
        'melody.mid', 'refine.mid', 'harmonize.mid', 'stem.mid', 'final.mid', 'report.json'}
    assert {instruments for task, instruments, _ in calls if task == '<HARMONIZE>'} == {
        ('Acoustic Piano', 'Electric Bass', 'Tenor Sax')}
    assert all(protected is not None for task, _, protected in calls if task == '<ADD_STEM>')
    assert (tmp_path/'clair'/'final.mid').is_file() and not output.exists()
    assert report['output']==report['files']['final']==str(tmp_path/'clair'/'final.mid')


def test_v5_cli_rejects_retired_orchestration_before_reading_music(tmp_path, monkeypatch):
    import sys
    from improvnet.inference import melody_inference as module
    infer = fixture_infer()
    infer.meta.update(version=11, tasks=list(TASKS))
    monkeypatch.setattr(module, 'MelodyInference', lambda _: infer)
    monkeypatch.setattr(sys, 'argv', ['inference', '--model', 'fixture', '--input', 'missing.mid',
        '--output', str(tmp_path/'out.mid'), '--task', 'orchestrate'])
    with pytest.raises(SystemExit):
        module.main()
    assert not (tmp_path/'out.mid').exists()



def test_pipeline_uses_solo_reference_for_harmony_and_mixture_for_drums(tmp_path,monkeypatch):
    infer=fixture_infer()
    reference=dict(style_solo_embedding=torch.ones(256),style_mixture_embedding=torch.ones(256)*2,beat_embedding=torch.ones(256)*3)
    seen={task:[] for task in TASKS}
    def capture(self,prompt,*args,**kwargs):
        task={'<REFINE>':'refine','<HARMONIZE>':'harmonize','<ADD_STEM>':'stem'}[prompt[0]]
        seen[task].append((kwargs['style_embedding'],kwargs['beat_embedding']))
        return fixture_generate(self,prompt,*args,**kwargs)
    monkeypatch.setattr(MelodyInference,'generate',capture)
    monkeypatch.setattr('improvnet.inference.melody_inference.refinement_source',
        lambda notes,strength,rng,tok,options,bounded:notes_to_tokens(notes,tok))
    run_pipeline(infer,[Note('Acoustic Piano',72+i%5,90,i*100,80) for i in range(70)],list(TASKS),
        pipeline_arguments(tmp_path),['Acoustic Piano','drum'],['Acoustic Piano'],reference,tmp_path/'shared-reference',['drum'])
    assert all(a is None and b is None for a,b in seen['refine'])
    assert seen['harmonize'] and all(a is reference['style_solo_embedding'] and b is reference['beat_embedding'] for a,b in seen['harmonize'])
    assert seen['stem'] and all(a is reference['style_mixture_embedding'] and b is reference['beat_embedding'] for a,b in seen['stem'])


@pytest.mark.parametrize('regions',('same','different'))
def test_cli_reference_regions_reaches_both_encoders_and_json_report(tmp_path,monkeypatch,regions):
    import json,sys
    from improvnet.inference import melody_inference as module
    from improvnet.utils.melody import write_midi_notes
    infer=fixture_infer();infer.checkpoint_step=1
    calls=[]
    reference=dict(style_solo_embedding=torch.ones(256),style_mixture_embedding=torch.ones(256)*2,beat_embedding=torch.ones(256)*3,
        regions=regions,style_region=dict(start_ms=0,end_ms=1000),beat_region=dict(start_ms=0 if regions=='same' else 2000,end_ms=1000 if regions=='same' else 3000))
    def embed(notes,tok,rng,regions):
        calls.append(regions);return dict(reference)
    infer.encoders=SimpleNamespace(from_notes=embed)
    monkeypatch.setattr(module,'MelodyInference',lambda _:infer)
    monkeypatch.setattr(MelodyInference,'generate',fixture_generate)
    monkeypatch.setattr(module,'refinement_source',lambda notes,strength,rng,tok,options,bounded:notes_to_tokens(notes,tok))
    source=tmp_path/'source.mid';output=tmp_path/'output.mid'
    write_midi_notes([Note('Acoustic Piano',72+i%5,90,i*100,80) for i in range(50)],source)
    monkeypatch.setattr(sys,'argv',['inference','--model','fixture','--input',str(source),'--output',str(output),
        '--style-reference',str(source),'--reference-regions',regions,'--temperature','0'])
    module.main()
    from improvnet.inference.melody_outputs import inference_output_paths
    report=json.loads(inference_output_paths(output)['report'].read_text())
    assert calls==[regions] and report['style_reference']['regions']==regions
    assert all(log['beat_conditioned'] for log in report['chunks'] if log['task'] in ('harmonize','stem'))


@pytest.mark.parametrize('produce_drums', [True, False])
def test_trio_streams_bass_after_drums_and_keeps_separate_outputs(tmp_path, monkeypatch, produce_drums):
    """The fourth worker must receive piano+drums and overlap its upstream worker."""
    def add_stem(self, prompt, genre, strength, instruments, window_ms, **kwargs):
        notes = fixture_generate(self, prompt, genre, strength, instruments, window_ms, **kwargs)
        target = kwargs.get('target_instrument')
        if target:
            notes = list(kwargs['fixed_notes'])
            if target != 'drum' or produce_drums:
                notes.append(Note(target, 36 if target == 'Acoustic Bass' else 42, 80, 0, 100))
        return ordered(notes)
    monkeypatch.setattr(MelodyInference, 'generate', add_stem)
    monkeypatch.setattr('improvnet.inference.melody_inference.refinement_source',
                        lambda notes, strength, rng, tok, options, bounded: notes_to_tokens(notes, tok))
    infer = fixture_infer()
    source = [Note('Acoustic Piano', 72+i%5, 90, i*100, 80) for i in range(70)]
    notes, logs, tasks, extra = run_pipeline(infer, source, ['refine','harmonize','stem','stem'],
        pipeline_arguments(tmp_path), ['Acoustic Piano','Acoustic Bass','drum'], ['Acoustic Piano'],
        None, tmp_path/'trio', ['drum','Acoustic Bass'])
    assert {n.instrument for n in notes} == {'Acoustic Piano','Acoustic Bass'} | ({'drum'} if produce_drums else set())
    assert {p.name for p in (tmp_path/'trio').glob('*.mid')} == {'refine.mid','harmonize.mid','drums.mid','bass.mid'}
    stages = extra['parallel_inference']['stage_timing']
    assert stages[3]['target_instrument'] == 'Acoustic Bass'
    assert stages[3]['first_forward_seconds'] < stages[2]['finished_seconds']
    assert extra['parallel_inference']['max_batch_size'] >= 2
    assert {log['target_instrument'] for log in logs if log['task']=='stem'} == {'drum','Acoustic Bass'}


def test_default_cli_requests_trio_and_reports_each_stem(tmp_path, monkeypatch):
    import json
    import sys
    from improvnet.inference import melody_inference as module
    from improvnet.utils.melody import write_midi_notes
    infer = fixture_infer()
    infer.checkpoint_step = 1
    calls = []
    def generate(self, prompt, genre, strength, instruments, window_ms, **kwargs):
        target = kwargs.get('target_instrument')
        calls.append((prompt[0], list(instruments), target))
        notes = fixture_generate(self, prompt, genre, strength, instruments, window_ms, **kwargs)
        if target:
            notes += [Note(target, 36 if target=='Acoustic Bass' else 42, 90, 0, 50)]
        return ordered(notes)
    monkeypatch.setattr(MelodyInference,'generate',generate)
    monkeypatch.setattr(module,'MelodyInference',lambda _: infer)
    source, output = tmp_path/'input.mid', tmp_path/'trio.mid'
    write_midi_notes([Note('Acoustic Piano',72,80,i*200,150) for i in range(30)],source)
    monkeypatch.setattr(sys,'argv',['inference','--model','fixture','--input',str(source),
        '--output',str(output),'--no-reference-conditioning','--temperature','0'])
    module.main()
    report = json.loads((tmp_path/'trio'/'report.json').read_text())
    assert report['missing_instruments'] == []
    assert {r['instrument'] for r in report['stem_summary']} == {'drum','Acoustic Bass'}
    assert report['files']['drums'].endswith('/drums.mid')
    assert report['files']['bass'].endswith('/bass.mid')
    assert report['output']==report['files']['final']==str(tmp_path/'trio'/'final.mid')
    assert (tmp_path/'trio'/'final.mid').is_file() and not output.exists()
    assert not any('Acoustic Bass' in instruments for task,instruments,_ in calls if task=='<HARMONIZE>')


@pytest.mark.parametrize('entrypoint', ['melody_inference', 'parallel_inference'])
def test_zero_melody_strength_copies_skyline_through_streamed_trio_pipeline(tmp_path, monkeypatch, entrypoint):
    import importlib
    import json
    import sys
    from improvnet.inference import melody_inference as canonical
    from improvnet.utils.melody import write_midi_notes, read_midi_notes, melody_voice, skyline
    infer = fixture_infer()
    infer.checkpoint_step = 1
    tasks_seen = []
    def generate(self, prompt, genre, strength, instruments, window_ms, **kwargs):
        assert prompt[0] != '<REFINE>', 'A zero-strength refinement must never call the model'
        tasks_seen.append(prompt[0])
        notes = fixture_generate(self,prompt,genre,strength,instruments,window_ms,**kwargs)
        target = kwargs.get('target_instrument')
        if target:
            notes = list(kwargs['fixed_notes']) + [Note(target,36 if target=='Acoustic Bass' else 42,80,0,50)]
        return ordered(notes)
    monkeypatch.setattr(MelodyInference,'generate',generate)
    monkeypatch.setattr(canonical,'MelodyInference',lambda _:infer)
    original = [Note('Acoustic Piano',pitch,velocity,100+i*150+delay,duration)
                for i in range(70)
                for pitch,velocity,delay,duration in ((48,80,0,300),(72+i%3,90,30,90))]
    source,output=tmp_path/'rolled.mid',tmp_path/'trio.mid'
    write_midi_notes(original,source,infer.tokenizer)
    expected=melody_voice(skyline(read_midi_notes(source,infer.tokenizer),50)[0])
    monkeypatch.setattr(sys,'argv',['inference','--model','fixture','--input',str(source),
        '--output',str(output),'--task','pipeline','--melody-strength','0','--rhythm-strength','.8',
        '--instruments','Acoustic Piano','drum','Acoustic Bass','--no-reference-conditioning'])
    importlib.import_module('improvnet.inference.'+entrypoint).main()
    folder=tmp_path/'trio'
    assert read_midi_notes(folder/'refine.mid',infer.tokenizer)==expected
    assert (folder/'refine.mid').read_bytes()==(folder/'melody.mid').read_bytes()
    report=json.loads((folder/'report.json').read_text())
    refinement=[log for log in report['chunks'] if log['task']=='refine']
    assert len(refinement)>1  # Exercise chunk boundaries and retained absolute onsets.
    assert sum(log['output_notes'] for log in refinement)==len(expected)
    assert all(log['decoding']['model_bypassed'] for log in refinement)
    assert report['missing_instruments']==[]
    assert '<HARMONIZE>' in tasks_seen and '<ADD_STEM>' in tasks_seen
    assert report['parallel_inference']['stage_order']==['refine','harmonize','drums','bass']
