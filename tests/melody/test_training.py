"""Current V5 training contracts."""
import hashlib
import json
import random
import sqlite3
import pytest
import torch
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from improvnet.model import melody_config as cfg
from improvnet.model.melody_model import MelodyRefinementModel, checkpoint_metadata, initialize_from_ar, validate_checkpoint
from improvnet.scripts.melody_manifest import build_manifest
from improvnet.utils.melody import Note, MelodyTokenizer, notes_to_tokens, ordered, tokens_to_notes, write_midi_notes, TASKS
from improvnet.utils.melody_corruptions import corrupt_history
from improvnet.utils.melody_style import STYLE_FEATURES as RHYTHM_FEATURES, style_tensors as rhythm_tensors
from improvnet.utils.melody_data import MelodyDataset, prepare_example, NoStemWindow, stem_source
from melody_test_helpers import settings, mix, response, corpus


@pytest.mark.parametrize('history', (0., 1.))
@pytest.mark.parametrize('corrupt', (0., 1.))
@pytest.mark.parametrize('strength', (0., .5, 1.))
def test_harmonization_is_mono_to_clean_pitched_mix_with_poly_history(history, corrupt, strength):
    rec = dict(mix(), _augment=True)
    # Disable transposition so exact ground-truth comparison is straightforward.
    class NoShift(random.Random):
        def randint(self, a, b):
            return 0 if (a, b) == (-5, 5) else super().randint(a, b)
    ex = prepare_example(rec, NoShift(17), settings(POLY_HISTORY_PROB=history,
                         HISTORY_CORRUPTION_PROB=corrupt), task='harmonize', strength=strength)
    tok = MelodyTokenizer()
    stream = [tok.id_to_tok[i] for i in ex['input'].tolist()]
    assert stream[0] == '<HARMONIZE>' and stream.count('<SEP>') == 2
    assert ex['input_mono'] and not ex['history_mono']
    assert bool(ex['history_length']) == bool(history)
    assert ex['history_corrupted'] == bool(history and corrupt)
    hist = tokens_to_notes(stream[1:1 + ex['history_length']], tok)
    if hist:
        assert not any(n.instrument == 'drum' for n in hist)
        assert len({n.instrument for n in hist}) == 2
    expected = ordered([Note(n[0], n[1], n[2], n[3] - ex['start_ms'], n[4])
                        for n in rec['notes'] if ex['start_ms'] <= n[3] < ex['end_ms']])
    assert response(ex) == expected  # Corrupted history never changes labels.
    assert ex['multi_hot'].sum() == 2
    assert ex['style_values'].shape == ex['style_mask'].shape == (len(RHYTHM_FEATURES),)
    assert ex['source_length'] <= 512 and ex['response_length'] < cfg.MIX_RESPONSE_TOKENS
    assert ex['history_length'] <= 2048 and len(ex['input']) + 3 <= 8192


def test_harmonization_history_probability_and_validation_are_stable():
    rec = dict(mix(50), _augment=True)
    examples = [prepare_example(rec, random.Random(i), task='harmonize') for i in range(200)]
    present = [x for x in examples if x['history_length']]
    assert 135 <= len(present) <= 165
    assert 25 <= sum(x['history_corrupted'] for x in present) <= 65
    clean = dict(rec, _augment=False)
    ex = prepare_example(clean, random.Random(8), settings(HISTORY_CORRUPTION_PROB=1., POLY_HISTORY_PROB=1.), 'harmonize')
    assert not ex['history_corrupted']


def test_history_addition_bias_and_drum_pitch_safety():
    notes = [Note('Acoustic Piano', 60, 90, i * 100, 80) for i in range(100)]
    differences = [len(corrupt_history(notes, random.Random(i))) - len(notes) for i in range(100)]
    assert sum(differences) > 100
    drums = [replace(n, instrument='drum', pitch=42) for n in notes]
    edited = corrupt_history(drums, random.Random(8), .5, end_ms=10000)
    assert all(n.pitch == 42 and n.instrument == 'drum' for n in edited)
    assert all(0 <= n.onset < 10000 and n.duration > 0 for n in edited)
    assert corrupt_history(notes, random.Random(1), 0) == notes


def test_no_harmony_features_or_parameters_enter_new_model():
    values, mask = rhythm_tensors({'accompaniment_attacks_per_second': 1000.,
                                  'accompaniment_voicing_density': 1000., 'ioi_mean_ms': 500.})
    assert len(RHYTHM_FEATURES)==20 and mask.sum()==1 and mask[0]==1
    model = MelodyRefinementModel(embed_dim=32, n_heads=4, n_kv_heads=2, n_layers=1,
                                   num_genres=4, num_instruments=41, dropout=0.)
    assert model.style_proj[0].in_features == 2*len(RHYTHM_FEATURES)


def test_original_ar_initialization_retains_backbone_and_initializes_new_conditions():
    from improvnet.model.ar_model import ARContextModel
    kwargs=dict(embed_dim=32,n_heads=4,n_kv_heads=2,n_layers=1,num_genres=4,num_instruments=41,dropout=0.)
    ar=ARContextModel(vocab_size=MelodyTokenizer().base_vocab_size,**kwargs)
    model=MelodyRefinementModel(**kwargs)
    initial_task=model.token_emb.weight[-1].clone()
    initialize_from_ar(model,dict(model_state_dict=ar.state_dict()))
    for key,value in ar.state_dict().items():
        actual=model.state_dict()[key]
        if key in ('token_emb.weight','lm_head.weight'):
            actual=actual[:value.shape[0]]
        assert torch.equal(value,actual),key
    assert torch.equal(initial_task,model.token_emb.weight[-1])
    assert model.strength_proj[-1].weight.count_nonzero()==0
    assert model.style_proj[-1].weight.count_nonzero()==0


def test_validation_separates_missing_stem_from_copied_notes():
    from improvnet.model.melody_train import evaluate_validation
    from improvnet.utils.ar_utils import ProcessData
    class FixedModel(torch.nn.Module):
        def forward(self,target,**kwargs):
            return torch.tensor([[[4.,0.,0.],[0.,4.,0.],[4.,0.,0.],[4.,0.,0.]]]).expand(2,-1,-1)
    batch = dict(input=torch.zeros(2,4,dtype=torch.long),target=torch.tensor([[0,1,2,-100]]*2),
        task=torch.tensor([2,2]),genre=torch.tensor([1,1]),multi_hot=torch.zeros(2,41),strength=torch.ones(2),
        style_values=torch.zeros(2,5),style_mask=torch.zeros(2,5),
        stem_mask=torch.tensor([[False,False,True,False],[False]*4]),
        stem_instrument=torch.tensor([ProcessData().INSTRUMENT_CLASSES.index('drum'),-1]))
    metrics = evaluate_validation(FixedModel(),[batch],torch.device('cpu'))
    assert metrics['stem_target/drum'] == metrics['stem_target']
    assert metrics['stem_target'] > metrics['stem']+2


@pytest.mark.parametrize('task', ('refine','harmonize','stem'))
def test_repeated_sampling_respects_all_token_caps(task):
    config = settings(CHUNK_TOKENS=64,MIX_RESPONSE_TOKENS=128,HISTORY_TOKENS=64,SEQ_LEN=256)
    for seed in range(30):
        example = prepare_example(dict(mix(200),_augment=True),random.Random(seed),config,task=task,strength=1.)
        assert example['source_length'] <= 64
        assert example['response_length']+1 <= (128 if task in ('harmonize','stem') else 64)
        assert example['history_length'] <= 64
        assert len(example['input'])+3 <= 256


@pytest.mark.parametrize('expand',(False,True))
def test_changed_source_note_count_preserves_complete_clean_target(expand,monkeypatch):
    def corrupt(core,*args):
        return notes_to_tokens(core+[replace(n,pitch=n.pitch-12) for n in core] if expand else [])
    monkeypatch.setattr('improvnet.utils.melody_data.refinement_source',corrupt)
    rec = mix(300)
    example = prepare_example(rec,random.Random(17),settings(CHUNK_TOKENS=64),task='refine',strength=1.)
    from improvnet.utils.melody import window,skyline,melody_voice
    expected = window(melody_voice(skyline([Note(*n) for n in rec['notes']],50)[0]),
                      example['start_ms'],example['end_ms'])
    assert response(example)==expected
    assert example['source_length']<=64
    assert example['source_length']>example['response_length']*1.8 if expand else example['source_length']==2


def ddp_worker(rank,port,config):
    import os
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),WORLD_SIZE='2',RANK=str(rank),LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    from improvnet.model.melody_train import train
    train(config)


def test_two_rank_training_with_three_task_dataset(tmp_path):
    import socket
    import time
    import torch.multiprocessing as mp
    config,_ = corpus(tmp_path,ratio=1,raw_count=0)
    updates=dict(NUM_WORKERS=0,BATCH_SIZE=1,ACCUM_STEPS=2,INIT_CHECKPOINT=None,N_STEPS=2,
        WARMUP_STEPS=0,DECAY_STEPS=1,VAL_EVERY=1,CHECKPOINT_EVERY=1,
        CHUNK_TOKENS=48,MIX_RESPONSE_TOKENS=96,HISTORY_TOKENS=64,SEQ_LEN=256,
        SAVE_DIR=str(tmp_path/'ddp'),
        MODEL_KWARGS=dict(embed_dim=32,n_heads=4,n_kv_heads=2,n_layers=1,num_genres=4,num_instruments=41,dropout=0.))
    for key,value in updates.items():setattr(config,key,value)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port = sock.getsockname()[1]
    context=mp.start_processes(ddp_worker,args=(port,config),nprocs=2,join=False,start_method='forkserver')
    try:
        deadline=time.monotonic()+90
        while not context.join(timeout=1):
            if time.monotonic()>deadline:
                pytest.fail('Two-rank training timed out')
    finally:
        for process in context.processes:
            if process.is_alive():process.terminate()
            process.join(timeout=10)
    checkpoint=torch.load(Path(config.SAVE_DIR)/'latest_checkpoint.pt',weights_only=False)
    assert checkpoint['step']==2 and checkpoint['world_size']==2
    assert checkpoint['refinement']['tasks']==['refine','harmonize','stem']
    assert len(checkpoint['rng_state']['rank_states'])==2


def test_manifest_does_not_parse_new_midi_and_preserves_held_out_groups(tmp_path):
    config, manifest = corpus(tmp_path)
    assert len(manifest['raw_train']) == 12
    assert not any(Path(x['path']).exists() for x in manifest['raw_train'])
    assert all(x['group'].startswith('raw:') for x in manifest['raw_train'])
    assert len(manifest['protected_melody_fingerprints']) > 0
    assert build_manifest(config) == manifest


def test_lazy_raw_record_cache_and_protected_copy_rejection(tmp_path, monkeypatch):
    config, manifest = corpus(tmp_path)
    path = Path(manifest['raw_train'][0]['path'])
    write_midi_notes([Note('Acoustic Piano',60+i,90,i*330,200) for i in range(10)],path)
    dataset = MelodyDataset(config.DATA_FILE,'train',config)
    row = next(r for r in dataset.other if r.get('path')==str(path))
    record = dataset.load_record(row)
    assert record['genre']=='pop'
    dataset.cache.clear()
    monkeypatch.setattr('improvnet.utils.melody_data.read_job',lambda *a: pytest.fail('Must reuse lazy parsed cache'))
    assert dataset.load_record(row)==record
    dataset.cache.clear()
    cached = next(Path(config.PARSED_CACHE).rglob('*.json'))
    digest=json.loads(cached.read_text())['sha256']
    dataset.protected_hashes.add(digest)
    with pytest.raises(ValueError,match='protected'):
        dataset.load_record(row)


def test_tiny_training_resume_and_checkpoint_protocol(tmp_path):
    from improvnet.model.melody_train import train
    config, _ = corpus(tmp_path,ratio=1,raw_count=0)
    config.NUM_WORKERS=0
    config.BATCH_SIZE=2
    config.ACCUM_STEPS=1
    config.CHUNK_TOKENS=48
    config.HISTORY_TOKENS=64
    config.MIX_RESPONSE_TOKENS=96
    config.SEQ_LEN=256
    config.N_STEPS=2
    config.WARMUP_STEPS=0
    config.DECAY_STEPS=1
    config.VAL_EVERY=2
    config.CHECKPOINT_EVERY=1
    config.INIT_CHECKPOINT=None
    config.SAVE_DIR=str(tmp_path/'run')
    config.MODEL_KWARGS=dict(embed_dim=32,n_heads=4,n_kv_heads=2,n_layers=1,num_genres=4,num_instruments=41,dropout=0.)
    train(config,n_steps=1)
    checkpoint=torch.load(Path(config.SAVE_DIR)/'latest_checkpoint.pt',weights_only=False)
    assert checkpoint['step']==1 and validate_checkpoint(checkpoint)['harmony_conditioning'] is False
    train(config,n_steps=2)
    checkpoint=torch.load(Path(config.SAVE_DIR)/'latest_checkpoint.pt',weights_only=False)
    assert checkpoint['step']==2 and checkpoint['refinement']['tasks']==['refine','harmonize','stem']
    assert checkpoint['refinement']['data_policy_revision']==10
    assert checkpoint['scheduler_state_dict']['last_epoch']==2
    assert all(state['step']==2 for state in checkpoint['optimizer_state_dict']['state'].values())
    progress=(Path(config.SAVE_DIR)/'training.log').read_text()
    assert 'Step      1 | CE' in progress and 'Step      2 | CE' in progress
    assert 'Resumed checkpoint at step 1; optimizer restored=True; scheduler restored=True' in progress
    assert 'Validation starting at step 2' in progress and 'Checkpoint saved at step 2.' in progress
    assert 'Data (train): retries=' in progress and '% of attempts' in progress
    from improvnet.inference.melody_inference import MelodyInference
    loaded=MelodyInference(Path(config.SAVE_DIR)/'latest_checkpoint.pt')
    assert loaded.model.style_proj[0].in_features==2*len(RHYTHM_FEATURES) and loaded.checkpoint_step==2


def test_joint_inference_uses_generated_full_mix_history_and_generates_new_timing():
    from improvnet.inference.melody_inference import MelodyInference
    infer=MelodyInference.__new__(MelodyInference)
    infer.tokenizer=tok=MelodyTokenizer()
    infer.meta=dict(seq_len=512,chunk_tokens=64,history_tokens=100,mix_response_tokens=128,
                    harmonize_expansion=4.,skyline_tolerance_ms=50)
    infer.options=cfg
    melody=[Note('Acoustic Piano',72+i%4,90,i*200,100) for i in range(30)]
    prompts=[]
    def generate(prompt,genre,strength,instruments,window_ms,*a,**kw):
        prompts.append(prompt)
        assert kw['stop_at_window_end'] and not kw['monophonic']
        assert instruments==['Acoustic Bass','Acoustic Piano']
        assert kw['style_values'].shape==(len(RHYTHM_FEATURES),)
        source=tokens_to_notes(prompt[prompt.index('<SEP>')+1:-2],tok)
        infer.last_decode_stats=dict(duplicate_attacks_removed=0,end_reason='window_boundary')
        return ([replace(n,onset=n.onset+10) for n in source]
                +[Note('Acoustic Bass',48,70,0,1000)])
    infer.generate=generate
    result,logs=infer.transform(melody,'harmonize','jazz',0.,random.Random(1),
                               instruments=['Acoustic Piano','Acoustic Bass','drum'])
    assert len(logs)>1 and all(x['history_view']=='poly' for x in logs)
    assert [n.onset for n in result if n.instrument=='Acoustic Piano']==[n.onset+10 for n in melody]
    assert all(x['total_token_budget']<=512 and x['response_tokens']<=x['response_token_budget'] for x in logs)
    for prompt in prompts[1:]:
        history=tokens_to_notes(prompt[1:prompt.index('<SEP>')],tok)
        assert not any(n.instrument=='drum' for n in history)
        assert any(n.instrument=='Acoustic Bass' for n in history)


def test_joint_inference_early_end_does_not_discard_unprocessed_melody():
    from improvnet.inference.melody_inference import MelodyInference
    infer=MelodyInference.__new__(MelodyInference)
    infer.tokenizer=tok=MelodyTokenizer()
    infer.meta=dict(seq_len=512,chunk_tokens=96,history_tokens=100,mix_response_tokens=256,
                    harmonize_expansion=3.,skyline_tolerance_ms=50)
    infer.options=cfg
    calls=[]
    def generate(prompt,*a,**kw):
        source=tokens_to_notes(prompt[prompt.index('<SEP>')+1:-2],tok)
        calls.append(source)
        infer.last_decode_stats={}
        return source[:2] if len(calls)==1 else source
    infer.generate=generate
    melody=[Note('Acoustic Piano',72+i%4,90,i*200,100) for i in range(30)]
    result,logs=infer.transform(melody,'harmonize','jazz',0.,random.Random(1),instruments=['Acoustic Piano'])
    assert result==melody and logs[0]['early_end']
    assert sum(x['consumed_input_notes'] for x in logs)==len(melody)


def test_ar_tokens_preserve_clocks_and_require_raw_midi_for_legacy_drum_fields():
    from improvnet.utils.melody_data import ar_record
    def event(inst,pitch,velocity,onset,duration):
        return [['instrument',inst],['pitch',pitch],['velocity',velocity],['onset',onset],['dur',duration]]
    entry=dict(midi_filepath='ar.mid',genre=None,tokens=[['<S>']*5,
        event('Acoustic Piano',72,90,0,500),['<T>']*5,
        event('Electric Bass',40,80,100,500),event('drum',42,80,100,20),['<E>']*5])
    record,_=ar_record(entry)
    assert record['genre']=='unknown'
    assert [Note(*n) for n in record['notes']]==[Note('Acoustic Piano',72,90,0,500),Note('Electric Bass',40,80,5100,500)]
    assert [Note(*n) for n in record['drums']]==[Note('drum',42,80,5100,20)]
    entry['tokens'].insert(2,event('drum',42,0,0,0))
    with pytest.raises(ValueError,match='require source MIDI'):
        ar_record(entry)


def test_ar_mixture_index_contains_only_original_training_offsets_and_unique_pieces(tmp_path):
    import pickle
    from improvnet.model import ar_config
    from improvnet.utils.ar_utils import load_split_byte_index
    from improvnet.scripts.melody_manifest import build_ar_mixtures
    config,_=corpus(tmp_path)
    ar_path=tmp_path/'ar.jsonl'
    offsets=[]
    with ar_path.open('wb') as handle:
        for i in range(100):
            offsets.append(handle.tell())
            parts=([('drum',42)] if i%3==0 else [('Acoustic Piano',72),('drum',42)] if i%3==1 else
                   [('Acoustic Piano',72),('Electric Bass',40)])
            notes=[[["instrument",inst],["pitch",p],["velocity",90],["onset",0],["dur",500]]
                   for inst,p in parts]
            handle.write((json.dumps(dict(midi_filepath=f'ar{i:03}.mid',genre=None,tokens=notes))+'\n').encode())
    with ar_path.with_name('ar_index.pkl').open('wb') as handle:pickle.dump(offsets,handle)
    config.AR_JSONL_FILES=[ar_path]
    config.AR_MIXTURE_INDEX=tmp_path/'ar_mixtures.json'
    config.AR_MIXTURE_LIMIT=100
    config.AR_MIXTURE_SCAN_PER_FILE=100
    rows=json.loads(config.PREPARED_DATA_FILE.with_suffix('.index.json').read_text())['rows']
    digest=hashlib.sha256(config.PREPARED_DATA_FILE.with_suffix('.index.json').read_bytes()).hexdigest()
    candidates,checksum=build_ar_mixtures(config,rows,digest)
    train=set(load_split_byte_index(str(ar_path),'train',ar_config))
    wanted={offset for i,offset in enumerate(offsets) if i%3!=0 and offset in train}
    assert {r['offset'] for r in candidates}==wanted
    assert all(r['ar_train'] and len(r['instruments'])==2 for r in candidates)
    assert len({r['path'] for r in candidates})==len(candidates)
    assert build_ar_mixtures(config,rows,digest)==(candidates,checksum)
    # Reproduce the old cache's neighbour contamination, then verify an in-place
    # cache upgrade checks the original record instead of trusting those labels.
    cached=json.loads(config.AR_MIXTURE_INDEX.read_text())
    cached['signature']['version']=1
    cached['rows']=[dict(kind='ar',jsonl=str(ar_path),offset=offset,path=f'ar{i:03}.mid',
                         source='ar_misc',genre='unknown',instruments=['Acoustic Piano','Electric Bass'],
                         has_multi_instrument=True,ar_train=True)
                    for i,offset in enumerate(offsets) if offset in train]
    config.AR_MIXTURE_INDEX.write_text(json.dumps(cached))
    repaired,_=build_ar_mixtures(config,rows,digest)
    assert {r['offset'] for r in repaired}==wanted
    assert all(len(r['instruments'])==2 for r in repaired)


def test_ar_mixture_uses_original_midi_for_real_drums_and_current_genre(tmp_path):
    config,_=corpus(tmp_path)
    dataset=MelodyDataset(config.DATA_FILE,'train',config)
    midi=tmp_path/'ar_source.mid'
    write_midi_notes([Note('Acoustic Piano',72,90,0,300),Note('Electric Bass',40,80,100,200),
                      Note('drum',42,110,100,20)],midi)
    row=dict(kind='ar',jsonl='unreadable_legacy_tokens.jsonl',offset=123,path='old/source.mid',
             raw_path=str(midi),genre='unknown',source='ar_gigamidi')
    record=dataset.load_record(row)
    assert record['genre']=='unknown' and record['source']=='ar_gigamidi'
    assert [Note(*n) for n in record['drums']]==[Note('drum',42,110,100,20)]
    dataset.cache.clear()
    # Reuse parsed notes while retaining this row's genre, even if a different
    # task previously parsed the same source under another genre label.
    row['genre']='classical'
    assert dataset.load_record(row)['genre']=='classical'


@pytest.mark.parametrize('failure',('malformed','no_pitched'))
def test_permanent_parse_failures_are_shared_and_invalidated_after_file_repair(tmp_path,monkeypatch,failure):
    import improvnet.utils.melody_data as module
    config,_=corpus(tmp_path)
    source=tmp_path/'broken.mid'
    if failure=='malformed':
        source.write_bytes(b'not a MIDI file')
    else:
        write_midi_notes([Note('drum',42,80,0,20)],source)
    row=dict(kind='raw',path=str(source),genre='classical',source='fixture')
    calls=[]
    read_job=module.read_job
    def spy(*args):
        calls.append(args)
        return read_job(*args)
    monkeypatch.setattr(module,'read_job',spy)
    with pytest.raises(ValueError):
        MelodyDataset(config.DATA_FILE,'train',config).load_record(row)
    assert len(calls)==1
    with pytest.raises(ValueError,match='Cached unusable source'):
        MelodyDataset(config.DATA_FILE,'train',config).load_record(row)
    assert len(calls)==1
    write_midi_notes([Note('Acoustic Piano',72,90,0,300),Note('Acoustic Piano',67,90,500,200)],source)
    assert MelodyDataset(config.DATA_FILE,'train',config).load_record(row)['notes']
    assert len(calls)==2


def test_source_resolution_maps_symphonynet_by_full_relative_path(tmp_path):
    from improvnet.scripts.melody_manifest import resolve_ar_midi_paths
    config,_=corpus(tmp_path)
    correct=tmp_path/'SymphonyNet_Dataset'/'SymphonyNet_Dataset'/'contemporary'/'123.mid'
    other=tmp_path/'SymphonyNet_Dataset'/'SymphonyNet_Dataset'/'classical'/'123.mid'
    with sqlite3.connect(config.CLASSICAL_INDEX) as db:
        for path in (correct,other):
            db.execute('INSERT INTO files VALUES(?,?,?,?,?,?)',(str(path),'symphonynet','unknown','[]',str(path),'train'))
    rows=[dict(path='/old/SymphonyNet_Dataset/contemporary/123.mid',source='ar_misc')]
    assert resolve_ar_midi_paths(rows,config)[0]['raw_path']==str(correct)


def test_transient_read_errors_are_not_persisted_as_bad_sources(tmp_path,monkeypatch):
    config,_=corpus(tmp_path)
    source=tmp_path/'temporarily_unavailable.mid'
    source.write_bytes(b'existing source')
    row=dict(kind='raw',path=str(source),genre='classical',source='fixture')
    calls=[]
    def transient(job):
        calls.append(job)
        return job,None,None,'OSError: [Errno 5] Input/output error'
    monkeypatch.setattr('improvnet.utils.melody_data.read_job',transient)
    for _ in range(2):
        with pytest.raises(ValueError,match='Input/output error'):
            MelodyDataset(config.DATA_FILE,'train',config).load_record(row)
    assert len(calls)==2


def test_progress_log_is_flushed_even_if_stdout_drops_messages(tmp_path,monkeypatch):
    import io
    from improvnet.model.melody_train import progress_logger
    class DiscardOutput(io.StringIO):
        def write(self,text):
            return len(text)
    monkeypatch.setattr('sys.stdout',DiscardOutput())
    log=progress_logger(tmp_path)
    try:
        for step in (1299,1300,1301):
            log.info('Step %s | CE 0.4',step)
        text=(tmp_path/'training.log').read_text()
        assert all(f'Step {step} | CE' in text for step in (1299,1300,1301))
    finally:
        for handler in log.handlers:
            handler.close()


def test_stem_drums_are_prioritized_and_solo_rejected():
 rec=mix(30);notes=ordered([Note(*n) for n in rec['notes']+rec['drums']])
 counts=Counter(stem_source(notes,random.Random(i),cfg)[0] for i in range(200))
 assert 150<=counts['drum']<=190
 solo=dict(rec,notes=[['Acoustic Piano',72,90,i*200,100] for i in range(30)])
 assert prepare_example(solo,random.Random(1),task='stem',target_instrument='drum')['stem_name']=='drum'
 with pytest.raises(ValueError,match='Unknown v5 task'):prepare_example(solo,random.Random(1),task='orchestrate')
 with pytest.raises(NoStemWindow):prepare_example(dict(solo,drums=[]),random.Random(1),task='stem')


def test_full_jazz_file_passes_and_auxiliary_mixture_draws_are_both_preserved(tmp_path,monkeypatch):
 config,_=corpus(tmp_path,raw_count=12)
 D=MelodyDataset(config.DATA_FILE,'train',config);loaded=[]
 monkeypatch.setattr(D,'load_record',lambda row:loaded.append(row) or dict(mix(20,row['genre'])))
 main=Counter();tasks=Counter();genre=Counter()
 fraction=sum(config.TASK_WEIGHTS[:2])/sum(config.TASK_WEIGHTS)
 for idx in range(len(D)):
  ex=D[idx];tasks[TASKS[ex['task']]]+=1;genre[ex['genre'].item()]+=1
  if math.floor((idx+1)*fraction+1e-9)>math.floor(idx*fraction+1e-9):
   if loaded[-1] in D.jazz_blues:main[loaded[-1]['offset']]+=1
 assert list(main.values())==[4]*len(D.jazz_blues)
 assert sum(tasks.values())==len(D) and tasks['stem']>0 and tasks['orchestrate']==0
 assert D.main_slots()==len(D.jazz_blues)*12
 # Evaluation is still unique, deterministic and unexpanded.
 V=MelodyDataset(config.DATA_FILE,'validation',config)
 assert len(V)==2 and torch.equal(V[0]['input'],V[0]['input'])


def test_new_protocol_rejects_old_resume_and_changed_training_controls(tmp_path):
 config,_=corpus(tmp_path)
 model=MelodyRefinementModel(embed_dim=32,n_heads=4,n_kv_heads=2,n_layers=1,num_genres=4,num_instruments=41)
 meta=checkpoint_metadata(model,config)
 validate_checkpoint(dict(refinement=meta),meta)
 for key,value in [('jazz_blues_repeats',3),('task_weights',[.3,.4,.3]),('data_policy_revision',4)]:
  with pytest.raises(ValueError,match='Resume settings differ'):
   validate_checkpoint(dict(refinement=dict(meta,**{key:value})),meta,allow_data_policy_update=True)
 with pytest.raises(ValueError,match='older branch'):
  validate_checkpoint(dict(refinement=dict(meta,version=7)))


def test_stem_eligibility_does_not_rescan_the_full_mix_per_chunk(monkeypatch):
 import improvnet.utils.melody_data as module
 original=module.window;scans=[]
 def counted(notes,*args):
  if len(notes)>=5000:scans.append(len(notes))
  return original(notes,*args)
 monkeypatch.setattr(module,'window',counted)
 example=prepare_example(mix(1000),random.Random(17),task='stem')
 assert example['source_length']<=512 and len(scans)<=2

