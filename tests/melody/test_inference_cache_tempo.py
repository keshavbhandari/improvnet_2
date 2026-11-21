"""Inference buffers and optional retiming preserve V14's training contract."""
from dataclasses import replace
import random
import pytest
import torch
from test_pipeline import tiny_model, prefill
from melody_test_helpers import reference_checkpoint
from improvnet.inference.melody_pipeline import ForwardRequest, forward_batch
from improvnet.model.kv_cache import StaticKVCache, reserve_kv_cache
from improvnet.model.encoders.runtime import FrozenConditioners
from improvnet.utils.melody import Note, MelodyTokenizer
from improvnet.utils.melody_reference import retime_reference, reference_tempo


@torch.no_grad()
def test_static_cache_matches_dynamic_logits_and_reuses_storage():
    model = tiny_model()
    request = prefill(0, 19)
    _, dynamic = model(request.ids, **request.kwargs)
    static = reserve_kv_cache(dynamic, 128)
    addresses = [(c.key.data_ptr(), c.value.data_ptr()) for c in static]
    prefix = [c[0].clone() for c in static]
    for step in range(12):
        token = torch.tensor([[100+step]])
        offset = dynamic[0][0].size(1)
        expected, dynamic = model(token, past_key_values=dynamic, use_cache=True, seq_offset=offset)
        actual, static = model(token, past_key_values=static, use_cache=True, seq_offset=offset)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert addresses == [(c.key.data_ptr(), c.value.data_ptr()) for c in static]
        for old, current, golden in zip(prefix, static, dynamic):
            torch.testing.assert_close(current[0][:,:old.size(1)],old,rtol=0,atol=0)
            for a,b in zip(current,golden):torch.testing.assert_close(a,b,rtol=0,atol=0)


@torch.no_grad()
def test_static_requests_survive_unequal_batches_and_return_to_single_decode():
    model = tiny_model()
    golden=[]; actual=[]
    for owner, length in enumerate((4, 17)):
        request=prefill(owner,length)
        logits,cache=model(request.ids,**request.kwargs)
        golden.append((logits,cache))
        actual.append((logits,reserve_kv_cache(cache,160)))
    pointers=[x[1][0].key.data_ptr() for x in actual]
    for step, owners in enumerate(((0,1),(1,),(0,1),(0,))):
        requests=[]
        for owner in owners:
            cache=actual[owner][1];token=torch.tensor([[80+step+owner]])
            expected,dynamic=model(token,past_key_values=golden[owner][1],seq_offset=cache[0].length,use_cache=True)
            golden[owner]=(expected,dynamic)
            requests.append(ForwardRequest(owner,token,dict(past_key_values=cache,seq_offset=cache[0].length,use_cache=True)))
        outputs=forward_batch(model,requests)
        for owner, result in zip(owners,outputs):
            actual[owner]=result
            torch.testing.assert_close(result[0],golden[owner][0],rtol=2e-5,atol=2e-6)
            assert result[1][0].key.data_ptr()==pointers[owner]
            for layer,expected in zip(result[1],golden[owner][1]):
                for a,b in zip(layer,expected):torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)


def test_cache_requires_inference_and_rejects_overflow():
    key=torch.zeros(1,2,2,4)
    with pytest.raises(RuntimeError,match='no_grad'):StaticKVCache(key,key,3)
    with torch.no_grad():
        cache=StaticKVCache(key,key,3)
        before=cache[0].clone()
        with pytest.raises(ValueError,match='capacity'):cache.append(key,key)
        assert cache.length==2
        torch.testing.assert_close(cache[0],before)
        cache.append(key[:,:1],key[:,:1]);assert cache.length==3


def trio(count=200):
    return sorted([Note(inst,pitch,80,i*500,duration) for i in range(count)
                   for inst,pitch,duration in (('Acoustic Piano',60,200),('Acoustic Bass',48,300),('drum',42,20))],key=lambda n:n.onset)


def test_tempo_scaling_is_uniform_nondestructive_and_preserves_sound_palette(monkeypatch):
    notes=trio(10)
    monkeypatch.setattr('improvnet.utils.melody_reference.reference_tempo',lambda _: (120.,1.))
    retimed,meta=retime_reference(notes,60.)
    assert meta['time_scale']==2.
    for old,new in zip(notes,retimed):
        assert new==replace(old,onset=old.onset*2,duration=old.duration*2)
    assert notes==trio(10)
    with pytest.raises(ValueError):retime_reference(notes,float('nan'))


def test_short_sources_cannot_silently_use_fallback_tempo():
    with pytest.raises(ValueError,match='rhythmic evidence'):reference_tempo([Note('Acoustic Piano',60,80,0,100)])


def test_reference_retiming_reaches_both_encoders_and_per_window_drum_profiles(tmp_path):
    runtime=FrozenConditioners(reference_checkpoint(tmp_path),reference_checkpoint(tmp_path,'beat'),'cpu')
    tok=MelodyTokenizer();notes=trio()
    original=[replace(n) for n in notes]
    instruments={'drums':['Acoustic Piano','drum']}
    normal=runtime.from_notes(notes,tok,stage_instruments=instruments,rng=random.Random(42))
    adjusted=runtime.from_notes(notes,tok,stage_instruments=instruments,rng=random.Random(42),tempo_bpm=60.)
    assert normal['tempo_adjustment'] is None
    assert adjusted['tempo_adjustment']['time_scale']>1.5
    a,b=normal['stage_conditions']['drums'],adjusted['stage_conditions']['drums']
    assert not torch.equal(a['reference_features'],b['reference_features'])
    assert b['drum_profile'][:,256+42].lt(a['drum_profile'][:,256+42]).all()
    assert a['drum_palette']==b['drum_palette']==[42]
    assert notes==original


@pytest.mark.parametrize('selection',['catalogue','midi'])
@pytest.mark.parametrize('tempo',['original','source'])
def test_cli_passes_optional_tempo_to_either_reference_selection_path(tmp_path,monkeypatch,selection,tempo):
    import json,sys
    from test_pipeline import fixture_infer,fake_reference
    from improvnet.inference.melody_cli import run_cli
    from improvnet.utils.melody import write_midi_notes
    infer=fixture_infer();infer.checkpoint_step=1
    source=tmp_path/'source.mid';ref=tmp_path/'reference.mid'
    write_midi_notes([n for n in trio(20) if n.instrument=='Acoustic Piano'],source,infer.tokenizer)
    ref_notes=[replace(n,onset=n.onset*2,duration=n.duration*2) for n in trio(20)]
    write_midi_notes(ref_notes,ref,infer.tokenizer)
    calls=[]
    def encode(*args,**kwargs):
        calls.append(kwargs)
        if tempo=='original':assert 'tempo_bpm' not in kwargs
        else:assert 100<kwargs['tempo_bpm']<140
        result=fake_reference(**kwargs)
        result.update(path=str(ref),_reference_notes=ref_notes)
        if tempo=='source':
            adjusted,adjustment=retime_reference(ref_notes,kwargs['tempo_bpm'])
            result.update(_conditioning_notes=adjusted,tempo_adjustment=adjustment)
        return result
    infer.encoders.from_notes=encode
    monkeypatch.setattr('improvnet.inference.melody_cli.choose_reference',encode)
    argv=['infer','--model','fixture','--input',str(source),'--output',str(tmp_path/'out.mid'),
          '--task','refine','--reference-tempo',tempo]
    argv+=['--style-reference',str(ref)] if selection=='midi' else ['--reference-description','slow']
    monkeypatch.setattr(sys,'argv',argv)
    def pipeline(*args):return args[1],[],['refine'],{}
    run_cli(lambda _:infer,pipeline)
    report=json.loads((tmp_path/'out'/'report.json').read_text())
    assert len(calls)==1
    assert report['reference_tempo']==tempo
    assert bool(report['source_tempo'])==(tempo=='source')
    saved=tmp_path/'out'/'reference.mid'
    assert saved.read_bytes()==ref.read_bytes()
    assert report['files']['reference']==str(saved)
    adjusted_path=tmp_path/'out'/'reference_tempo.mid'
    assert adjusted_path.exists()==(tempo=='source')
    if tempo=='source':
        from improvnet.utils.melody import read_midi_notes
        adjusted=read_midi_notes(adjusted_path,infer.tokenizer,include_drums=True)
        assert adjusted and {n.instrument for n in adjusted}=={'Acoustic Piano','Acoustic Bass','drum'}
        assert max(n.onset for n in adjusted)<max(n.onset for n in ref_notes)
        assert report['files']['reference_tempo']==str(adjusted_path)
    assert '_conditioning_notes' not in report['style_reference']


def test_report_records_observed_generation_time_without_changing_decode_statistics(monkeypatch):
    from melody_test_helpers import scripted_inference,attrs,settings
    infer,_=scripted_inference(attrs(60,0,100)+['<E>'])
    infer.options=settings();infer.meta.update(skyline_tolerance_ms=50,history_tokens=0,chunk_tokens=512)
    infer.progress_disabled=True
    _,logs=infer.transform([Note('Acoustic Piano',60,80,0,100)],'refine','jazz',.2,random.Random(42))
    assert logs[0]['generation_seconds']>0 and logs[0]['tokens_per_second']>0
    assert logs[0]['tokens_per_second']*logs[0]['generation_seconds']==pytest.approx(logs[0]['response_tokens'])


def test_cli_rejects_manual_reference_bpm_before_loading_model(monkeypatch):
    import sys
    from improvnet.inference.melody_cli import run_cli
    monkeypatch.setattr(sys,'argv',['infer','--model','unused','--input','unused.mid','--output','out.mid','--reference-tempo','90'])
    def load(_):raise AssertionError('Invalid option must fail before loading')
    with pytest.raises(SystemExit):run_cli(load)


def test_cli_reference_export_cannot_overwrite_the_original_reference(tmp_path,monkeypatch):
    import sys
    from test_pipeline import fixture_infer
    from improvnet.inference.melody_cli import run_cli
    from improvnet.utils.melody import write_midi_notes
    infer=fixture_infer()
    source=tmp_path/'source.mid'
    write_midi_notes([n for n in trio(20) if n.instrument=='Acoustic Piano'],source,infer.tokenizer)
    folder=tmp_path/'out';folder.mkdir()
    reference=folder/'reference.mid'
    write_midi_notes(trio(20),reference,infer.tokenizer)
    original_bytes=reference.read_bytes()
    monkeypatch.setattr(sys,'argv',['infer','--model','fixture','--input',str(source),
        '--output',str(tmp_path/'out.mid'),'--task','refine','--style-reference',str(reference)])
    with pytest.raises(SystemExit):run_cli(lambda _:infer)
    assert reference.read_bytes()==original_bytes
    assert not (folder/'melody.mid').exists()
