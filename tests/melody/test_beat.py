"""Aligned track groups, hard temporal negatives and both frozen conditions."""
from dataclasses import replace
import math
import random
from types import SimpleNamespace
import pytest
import torch
from improvnet.model.encoders.beat.data import partitions,aligned_window,encode_views,BeatPairDataset,BeatCollator
from improvnet.model.encoders.objective import beat_loss
from improvnet.model.encoders.runtime import FrozenConditioners
from improvnet.model.encoders.beat import config as beat_config
from improvnet.model.encoders.style import config as style_config
from improvnet.model.encoders.train import lr_factor,settings_dict
from improvnet.model.encoders.beat.train import train
from improvnet.utils.melody import Note,MelodyTokenizer,notes_to_tokens,tokens_to_notes
from improvnet.utils.melody_data import prepare_example,collate_examples
from melody_test_helpers import reference_checkpoint,corpus,mix,settings


def tracks(count=200):
    return [Note(inst,pitch,90,i*200+offset,130) for i in range(count)
            for inst,pitch,offset in [('Acoustic Piano',72,0),('drum',42,70),('Violin',65,30),('Tenor Sax',58,100)]]


def test_bipartitions_include_single_to_mix_and_mix_to_mix():
    names={'Acoustic Piano','drum','Violin','Tenor Sax','Acoustic Guitar'}
    seen=set()
    for seed in range(300):
        a,b=partitions(names,random.Random(seed))
        assert a and b and a.isdisjoint(b) and a|b==names
        seen.add((frozenset(a),frozenset(b)))
    assert any(len(a)==1 and len(b)==4 for a,b in seen)
    assert any(len(a)==2 and len(b)==3 for a,b in seen)
    assert any({'Acoustic Piano','drum'}<=a and 'Violin' in b for a,b in seen)
    with pytest.raises(ValueError,match='multiple'):partitions(['Acoustic Piano'],random.Random(1))


def test_aligned_views_preserve_shared_origin_offsets_and_complete_budgets():
    notes=tracks()
    for seed in range(10):
        views,(start,end)=aligned_window(notes,({'Acoustic Piano','Violin'},{'drum','Tenor Sax'}),random.Random(seed),128)
        for view in views:
            assert len(notes_to_tokens(view))<=128
            assert len({n.onset for n in view})>=2
            for n in view:
                assert any(o.instrument==n.instrument and o.pitch==n.pitch and o.onset==n.onset+start and n.duration<=o.duration for o in notes)
                assert n.onset+n.duration<=end-start
        # A per-view rebase would make both first notes zero and destroy this.
        assert views[0][0].onset!=views[1][0].onset
        negative,nt=aligned_window(notes,({'Acoustic Piano','Violin'},{'drum','Tenor Sax'}),random.Random(seed+50),128,excluded=(start,end))
        assert nt[1]+1000<=start or nt[0]>=end+1000


def test_large_absolute_timestamps_do_not_create_unbounded_clock_tokens():
    notes=[replace(n,onset=n.onset+10**13) for n in tracks(50)]
    views,bounds=aligned_window(notes,({'Acoustic Piano','Violin'},{'drum','Tenor Sax'}),random.Random(1),64)
    assert bounds[0]>=10**13
    assert all(len(notes_to_tokens(v))<=64 for v in views)


def test_augmentation_changes_view_keys_but_not_clocks_or_drum_numbers():
    tok=MelodyTokenizer()
    views,_=aligned_window(tracks(),({'Acoustic Piano','Violin'},{'drum','Tenor Sax'}),random.Random(4))
    for seed in range(25):
        ids,shifts=encode_views(views,random.Random(seed),tok,augment=True,max_semitones=7,mask_prob=0,mask_rate_range=(.15,.45))
        assert shifts[0]!=shifts[1] and all(-7<=s<=7 for s in shifts)
        for original,encoded in zip(views,ids):
            recovered=tokens_to_notes([tok.id_to_tok[int(i)] for i in encoded],tok)
            assert [(n.onset,n.duration,n.instrument) for n in recovered]==[(n.onset,tok._quantize_dur(n.duration),n.instrument) for n in original]
            assert [n.pitch for n in recovered if n.instrument=='drum']==[n.pitch for n in original if n.instrument=='drum']
    masked,_=encode_views(views,random.Random(2),tok,augment=True,max_semitones=7,mask_prob=1,mask_rate_range=(.3,.3))
    assert all(int((v==tok.tok_to_id['<MASK>']).sum())==round((len(v)-2)*.3) for v in masked)


def test_same_piece_separate_window_is_a_real_negative_and_overlap_is_not():
    a=b=torch.tensor([[1.,0.]],requires_grad=True)
    na=nb=torch.tensor([[0.,1.]],requires_grad=True)
    ids=torch.tensor([1]);time=torch.tensor([[0,100]])
    loss,stats=beat_loss(a,b,na,nb,ids,time,ids,torch.tensor([[1200,1300]]),temperature=.5)
    assert float(loss.detach())==pytest.approx(math.log1p(math.exp(-2)))
    assert stats[4]==1 and stats[6]==1
    loss.backward();assert na.grad is not None and a.grad is not None
    with pytest.raises(ValueError,match='No separate'):
        beat_loss(a,b,na,nb,ids,time,ids,time)
    # Different-piece negatives remain valid regardless of timestamp coincidence.
    assert torch.isfinite(beat_loss(a,b,na,nb,ids,time,torch.tensor([2]),time)[0])


def test_beat_backbone_masking_optimizer_and_full_curve_match_style():
    same=('EMBED_DIM','N_HEADS','N_KV_HEADS','N_LAYERS','FFN_MULT','DROPOUT','PROJECTION_DIM','MAX_LENGTH','MIN_LENGTH','EVAL_LENGTH',
          'BATCH_SIZE','ACCUM_STEPS','LR','MIN_LR','WARMUP_STEPS','BETAS','WEIGHT_DECAY','GRAD_CLIP','TEMPERATURE','MASK_PROB','MASK_RATE_RANGE','N_STEPS')
    assert all(getattr(beat_config,k)==getattr(style_config,k) for k in same)
    for step in (0,300,500,10000,19999,20000,25000,30000):
        assert lr_factor(step,beat_config)==pytest.approx(lr_factor(step,style_config))
    assert not hasattr(beat_config,'CONTINUATION_START_STEP')


def test_multitrack_dataset_fold_and_cpu_training_roundtrip(tmp_path,monkeypatch):
    config,_=corpus(tmp_path,ratio=1,raw_count=0)
    import improvnet.model.encoders.data as style_module
    monkeypatch.setattr(style_module,'melody_config',config)
    data=BeatPairDataset(config.DATA_FILE,'train',piano_drum_draw_prob=0)
    val=BeatPairDataset(config.DATA_FILE,'validation')
    assert all(len(set(r['instruments']))>=2 for r in data.rows+val.rows)
    assert {r['group'] for r in data.rows}.isdisjoint({r['group'] for r in val.rows})
    row=data[0]
    assert set(row['a_instruments']).isdisjoint(row['b_instruments'])
    if row['piece_id']==row['negative_piece_id']:
        assert row['bounds'][1]+1000<=row['negative_bounds'][0] or row['negative_bounds'][1]+1000<=row['bounds'][0]
    batch=BeatCollator(MelodyTokenizer().pad_id)([data[0],data[1]])
    assert batch['negative_a'].shape==batch['negative_a_mask'].shape
    # Exact current trainer/checkpoint format, using local synthetic music only.
    tiny=SimpleNamespace(**settings_dict(beat_config))
    tiny.MANIFEST=str(config.DATA_FILE);tiny.SAVE_DIR=str(tmp_path/'beat-run')
    tiny.EMBED_DIM=16;tiny.N_HEADS=2;tiny.N_KV_HEADS=1;tiny.N_LAYERS=1
    tiny.BATCH_SIZE=2;tiny.VALIDATION_BATCH_SIZE=2;tiny.NUM_WORKERS=0
    tiny.N_STEPS=2;tiny.DECAY_STEPS=1;tiny.WARMUP_STEPS=0;tiny.SAMPLES_PER_EPOCH=4
    tiny.TENSORBOARD=False;tiny.DEVICE='cpu';tiny.VALIDATE_EVERY=1;tiny.SAVE_EVERY=1;tiny.LOG_EVERY=1
    tiny.DATASET_OPTIONS=dict(tiny.DATASET_OPTIONS,piano_drum_draw_prob=0)
    train(tiny);train(tiny)
    state=torch.load(tmp_path/'beat-run'/'latest_checkpoint.pt',weights_only=True)
    assert state['trained_steps']==2 and state['format']=='beat_v1'
    assert all(s['step']==2 for s in state['optimizer_state_dict']['state'].values())


def test_dual_references_both_exclude_target_and_held_releases():
    rec=mix(300)
    # Keep a late release in the ownership window: both references must avoid it.
    rec['notes'].append(('Acoustic Piano',100,90,10000,15000))
    for task in ('harmonize','stem'):
        for seed in range(12):
            ex=prepare_example(rec,random.Random(seed),settings(REFERENCE_PROB=1.),task=task)
            end=max([ex['end_ms']]+[n[3]+n[4] for n in rec['notes']+rec['drums'] if ex['start_ms']<=n[3]<ex['end_ms']])
            for kind in ('style','beat'):
                if ex[kind+'_present']:
                    a,b=ex[kind+'_bounds']
                    assert b<=ex['start_ms']-250 or a>=end+250
                    assert len(ex[kind+'_ids'])<=512
    batch=collate_examples([prepare_example(rec,random.Random(1),task='refine')])
    assert not batch['style_present'].any() and not batch['beat_present'].any()


def test_frozen_dual_runtime_same_different_regions_and_gradients(tmp_path):
    style=reference_checkpoint(tmp_path)
    beat=reference_checkpoint(tmp_path,kind='beat')
    enc=FrozenConditioners(style,beat,'cpu')
    notes=tracks(300)
    a=enc.from_notes(notes,MelodyTokenizer(),random.Random(5),'same')
    b=enc.from_notes(notes,MelodyTokenizer(),random.Random(5),'different')
    assert a['style_region']['start_ms']==a['beat_region']['start_ms']
    assert a['style_region']['end_ms']==a['beat_region']['end_ms']
    sr,br=b['style_region'],b['beat_region']
    assert sr['end_ms']+250<=br['start_ms'] or br['end_ms']+250<=sr['start_ms']
    assert all(v.shape==(256,) for k,v in a.items() if k.endswith('_embedding') and v is not None)
    assert all(not p.requires_grad for model in (enc.style.model,enc.beat.model) for p in model.parameters())
    with pytest.raises(ValueError,match='too short'):
        enc.from_notes(tracks(2),MelodyTokenizer(),random.Random(1),'different')


def test_other_performance_of_same_piece_is_not_a_temporal_negative():
    features=torch.eye(2)
    ids=torch.tensor([1,1]);times=torch.tensor([[0,100],[3000,3100]])
    # Two recordings with separate clocks: only their own distant-window negatives count.
    loss,stats=beat_loss(features,features,features.flip(0),features.flip(0),ids,times,ids,times+6000,
        record_ids=torch.tensor([10,20]),negative_record_ids=torch.tensor([10,20]))
    assert torch.isfinite(loss) and stats[4]==2


def test_both_ar_projections_learn_while_both_encoders_stay_fixed(tmp_path):
    from improvnet.model.melody_model import MelodyRefinementModel
    from improvnet.model.melody_train import loss_for_batch
    enc=FrozenConditioners(reference_checkpoint(tmp_path),reference_checkpoint(tmp_path,kind='beat'),'cpu')
    config=settings(CHUNK_TOKENS=32,MIX_RESPONSE_TOKENS=64,HISTORY_TOKENS=16,SEQ_LEN=128,REFERENCE_PROB=1.)
    batch=collate_examples([prepare_example(mix(200),random.Random(i),config,task='harmonize') for i in range(3)])
    enc.condition_batch(batch)
    assert batch['style_mask'].sum()>0 and batch['beat_mask'].sum()>0
    model=MelodyRefinementModel(embed_dim=32,n_heads=4,n_kv_heads=2,n_layers=1,num_genres=4,num_instruments=41,dropout=0.)
    loss_for_batch(model,batch).backward()
    assert model.style_proj[-1].weight.grad.abs().sum()>0
    assert model.beat_proj[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None for e in (enc.style,enc.beat) for p in e.model.parameters())


def test_same_record_with_different_metadata_alias_is_not_a_false_negative():
    features=torch.eye(2)
    ids=torch.tensor([1,2]);times=torch.tensor([[0,100],[0,100]])
    _,stats=beat_loss(features,features,features.flip(0),features.flip(0),ids,times,ids,times+6000,
        record_ids=torch.tensor([10,10]),negative_record_ids=torch.tensor([10,10]))
    # Two far-window negatives/anchor; the overlapping aliased positive is blocked.
    assert stats[4]==4


def test_budget_counts_clock_tokens_after_ten_ms_rounding():
    class OrderedStarts(random.Random):
        def shuffle(self,items):pass
    notes=[Note(inst,pitch,90,round(i*4999/126),20) for i in range(127)
           for inst,pitch in [('Acoustic Piano',60),('drum',42)]]
    views,_=aligned_window(notes,({'Acoustic Piano'},{'drum'}),OrderedStarts(),510)
    assert all(len(notes_to_tokens(view))<=510 for view in views)
    assert all(view[-1].onset<4999 for view in views)
