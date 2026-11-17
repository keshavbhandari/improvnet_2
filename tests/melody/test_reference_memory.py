"""V12 prefix memory, source-safe controls, frozen weights and cached attention."""
from dataclasses import replace
import random
from types import SimpleNamespace
import pytest
import torch
from melody_test_helpers import settings,mix,reference_checkpoint,response,scripted_inference,attrs
from improvnet.model.melody_model import MelodyRefinementModel,initialize_from_ar
from improvnet.model.ar_model import ARContextModel
from improvnet.model.encoders.runtime import FrozenConditioners
from improvnet.utils.melody import MelodyTokenizer,Note,notes_to_tokens,tokens_to_notes,make_prompt,ordered
from improvnet.utils.melody_data import prepare_example,collate_examples
from improvnet.utils.drum_profile import drum_profile,reference_kit,DRUM_PROFILE_DIM
from improvnet.model.melody_train import loss_for_batch


def tiny(**kwargs):
    return MelodyRefinementModel(embed_dim=32,n_heads=4,n_kv_heads=2,n_layers=2,
        num_genres=4,num_instruments=41,dropout=0.,memory_dim=32,**kwargs).eval()


def condition(batch=1):
    return dict(reference_features=torch.randn(batch,20,256),reference_mask=torch.ones(batch,20,dtype=torch.bool),
        reference_positions=torch.rand(batch,20,3),drum_profile=torch.zeros(batch,DRUM_PROFILE_DIM))


def test_prefix_order_original_decoder_blocks_and_absent_reference_neutrality():
    torch.manual_seed(12)
    ar=ARContextModel(vocab_size=MelodyTokenizer().vocab_size,embed_dim=32,n_heads=4,n_kv_heads=2,n_layers=2,
        num_genres=4,num_instruments=41,dropout=0.).eval()
    model=tiny();initialize_from_ar(model,ar.state_dict())
    assert set(model.layers.state_dict())==set(ar.layers.state_dict())
    ids=torch.tensor([[3,4,5,6]])
    supplied=condition()
    captured=[]
    handle=model.layers[0].register_forward_pre_hook(lambda m,args,kwargs:captured.append(args[0].detach()),with_kwargs=True)
    actual=model(ids,strength=torch.ones(1),**supplied)
    handle.remove()
    memory=model.memory_projection(model.reference_memory(supplied['reference_features'],supplied['reference_mask'],supplied['reference_positions'],supplied['drum_profile']))
    assert captured[0].shape==(1,34+ids.size(1),32)
    torch.testing.assert_close(captured[0][:,1:33],memory)
    torch.testing.assert_close(captured[0][:,34:],model.token_emb(ids))
    torch.testing.assert_close(actual,ar(ids,prefix_memory=memory))
    supplied['reference_mask'].zero_()
    torch.testing.assert_close(model(ids,strength=torch.ones(1),**supplied),model(ids,strength=torch.ones(1)),rtol=0,atol=0)


def test_32_memory_tokens_and_projection_receive_gradients_on_first_update():
    model=tiny();inputs=condition();ids=torch.tensor([[3,4,5]])
    seen=[]
    handle=model.reference_memory.register_forward_hook(lambda m,a,out:seen.append(out.shape))
    model(ids,strength=torch.ones(1),**inputs).square().mean().backward()
    assert seen==[torch.Size((1,32,32))]
    assert model.memory_projection.weight.grad.abs().sum()>0
    assert model.reference_memory.queries.grad.abs().sum()>0
    handle.remove()


@torch.no_grad()
def test_reference_prefix_is_cached_once_and_cached_logits_match_full_decoding():
    model=tiny();inputs=condition();ids=torch.tensor([[3,4,5,6,7]])
    expected=model(ids,strength=torch.ones(1),**inputs)
    calls=[]
    handle=model.reference_memory.register_forward_hook(lambda *a:calls.append(1))
    _,cache=model(ids[:,:2],strength=torch.ones(1),use_cache=True,**inputs)
    assert all(len(layer)==2 and layer[0].size(1)==36 for layer in cache)
    saved=cache[0][0][:,:34].clone()
    decoded=[]
    for i in range(2,5):
        logits,cache=model(ids[:,i:i+1],past_key_values=cache,use_cache=True,seq_offset=i+34)
        decoded.append(logits)
        torch.testing.assert_close(cache[0][0][:,:34],saved)
    torch.testing.assert_close(torch.cat(decoded,1),expected[:,2:],rtol=2e-5,atol=2e-6)
    assert calls==[1];handle.remove()


def test_banks_exclude_target_held_releases_keep_percussion_and_all_streams_unmarked():
    rec=mix(400)
    rec['notes'] += [('Violin',66,80,i*200,100) for i in range(400)]
    rec['drums'] += [('drum',60,80,i*100,20) for i in range(800)]
    tok=MelodyTokenizer()
    for probability,expected in ((1.,{'Acoustic Piano','Acoustic Bass','Violin','drum'}),
                                 (0.,{'Acoustic Piano','Acoustic Bass','drum'})):
        config=settings(REFERENCE_PROB=1.,REFERENCE_EXACT_PROB=probability)
        ex=prepare_example(rec,random.Random(3),config,'stem',target_instrument='drum')
        protected_end=max([ex['end_ms']]+[n[3]+n[4] for n in rec['notes']+rec['drums'] if ex['start_ms']<=n[3]<ex['end_ms']])
        assert len(ex['reference_bounds'])==1
        for a,b in ex['reference_bounds']:assert b<=ex['start_ms']-250 or a>=protected_end+250
        for i,(a,b) in enumerate(ex['reference_bounds']):
            assert all(b<=c or a>=d for c,d in ex['reference_bounds'][i+1:])
        for stream in ex['reference_ids']:
            stream=stream[stream.ne(tok.pad_id)]
            if not len(stream):continue
            notes=tokens_to_notes([tok.id_to_tok[i] for i in stream.tolist()])
            assert {n.instrument for n in notes}==expected
            assert any(n.instrument=='drum' and n.pitch==60 for n in notes)
        assert tok.tok_to_id[tok.blank_tok] not in ex['input']
        assert ex['drum_profile'][60]==1 and any(n.instrument=='drum' and n.pitch==60 for n in response(ex))


def test_frozen_local_features_both_encoders_train_memory_without_encoder_updates(tmp_path):
    runtime=FrozenConditioners(reference_checkpoint(tmp_path),reference_checkpoint(tmp_path,'beat'),'cpu')
    config=settings(CHUNK_TOKENS=32,HISTORY_TOKENS=16,SEQ_LEN=128,REFERENCE_PROB=1.)
    examples=[prepare_example(mix(300),random.Random(i),config,'harmonize') for i in range(2)]
    batch=collate_examples(examples);runtime.condition_batch(batch)
    assert batch['reference_features'].shape==(2,20,256)
    assert batch['reference_mask'].any() and not batch['reference_features'].requires_grad
    model=tiny()
    loss_for_batch(model,batch).backward()
    assert model.reference_memory.queries.grad.abs().sum()>0
    assert all(not p.requires_grad and p.grad is None for enc in (runtime.style,runtime.beat) for p in enc.model.parameters())
    # Refinement supplies neither profiles nor reference windows.
    empty=collate_examples([prepare_example(mix(300),random.Random(1),config,'refine')])
    runtime.condition_batch(empty)
    assert not empty['reference_mask'].any() and not empty['drum_profile'].any()


@pytest.mark.parametrize('aggregation',['window','average'])
def test_shared_reference_window_or_averaging_reuses_stage_views(tmp_path,aggregation):
    runtime=FrozenConditioners(reference_checkpoint(tmp_path),reference_checkpoint(tmp_path,'beat'),'cpu')
    runtime.style.max_tokens=runtime.beat.max_tokens=34
    notes=[Note(inst,pitch,80,i*200,80) for i in range(50) for inst,pitch in [('Acoustic Piano',60),('drum',51),('drum',38),('drum',60)]]
    streams=[]
    for encoder in (runtime.style,runtime.beat):
        original=encoder.feature_windows
        def counted(ids,mask,bins,original=original):
            streams.append(ids.clone())
            return original(ids,mask,bins)
        encoder.feature_windows=counted
    result=runtime.from_notes(notes,MelodyTokenizer(),aggregation=aggregation,rng=random.Random(7),
        stage_instruments={'one':['Acoustic Piano','drum'],'two':['Acoustic Piano','drum']})
    a,b=result['stage_conditions'].values()
    assert len(streams)==2
    torch.testing.assert_close(*streams)
    assert streams[0].shape[0]==(1 if aggregation=='window' else a['reference_region']['candidate_windows'])
    assert a['reference_features'] is b['reference_features']
    assert a['reference_features'].shape==(20,256)
    assert len(a['reference_region']['selected_windows'])==streams[0].shape[0]
    assert a['drum_palette']==[38,51]
    assert a['drum_profile'][38]==a['drum_profile'][51]==1 and a['drum_profile'][60]==0
    with pytest.raises(ValueError,match='Reference lacks'):
        runtime.from_notes(notes,MelodyTokenizer(),stage_instruments={'bass':['Acoustic Piano','drum','Acoustic Bass']})


def test_training_reference_pitch_shifts_preserve_timing_instruments_and_drums():
    config=settings(REFERENCE_PROB=1.)
    ex=prepare_example(dict(mix(400),_augment=True),random.Random(12),config,'stem',target_instrument='drum')
    tok=MelodyTokenizer()
    def decode(bank):return tokens_to_notes([tok.id_to_tok[i] for i in bank[0].tolist() if i!=tok.pad_id])
    clean=decode(ex['reference_ids'])
    for kind,key in enumerate(('reference_style_ids','reference_beat_ids')):
        shifted=decode(ex[key]);shift=ex['reference_pitch_shifts'][0][kind]
        assert 0<abs(shift)<=7
        assert shifted==ordered([replace(n,pitch=n.pitch+shift) if n.instrument!='drum' else n for n in clean])
    assert ex['drum_profile'].equal(drum_profile(clean))


def test_reference_palette_constrains_new_drums_without_inventing_density():
    infer,calls=scripted_inference([]);tok=infer.tokenizer
    anchor=Note('Acoustic Piano',72,90,0,100)
    script=attrs(72,0,100)+attrs(42,0,20,'drum')+['<E>']
    def model(ids,**kwargs):
        index=len(calls);calls.append(ids)
        logits=torch.full((1,ids.size(1),tok.vocab_size),-100.)
        logits[...,tok.tok_to_id[('note',38,90)]]=80.
        logits[...,tok.tok_to_id[script[index]]]=100.
        return logits,()
    infer.model=model
    out=infer.generate(make_prompt('stem',notes_to_tokens([anchor]),target_instrument='drum'),'jazz',1.,
        ['Acoustic Piano','drum'],200,temperature=0.,fixed_notes=[anchor],stem_add=True,target_instrument='drum',
        drum_palette=[38,51],max_new_tokens=40)
    assert {n.pitch for n in out if n.instrument=='drum'}=={38}
    assert anchor in out


def test_profile_is_coarse_and_standard_palette_is_derived_without_transposing_drums():
    notes=[Note('drum',p,80,i*250,20) for i,p in enumerate((22,38,51,60))]
    p=drum_profile(notes,duration_ms=1000)
    assert p.shape==(400,) and torch.isfinite(p).all()
    assert p[:128].sum()==4 and p[60]==1
    assert reference_kit(notes)==[38,42,51]
    assert not drum_profile([]).any()
