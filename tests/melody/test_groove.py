"""Current V5 groove contracts."""
import json
import numpy as np
import pytest
import torch
import random
from collections import OrderedDict
from types import SimpleNamespace
from torch.utils.data import DataLoader, TensorDataset
from improvnet.utils.melody import Note
from improvnet.utils.melody_groove import infer_clock, reference_windows, MAX_CLOCK_ANALYSIS_MS, BeatClock, GrooveGuide, groove_statistics
from improvnet.utils.melody_data import MelodyDataset
from improvnet.utils.ar_utils import resume_epoch_loader
from improvnet.utils.melody_style import style_tensors


def phrase(origin=0,count=40):
    return [Note('Acoustic Piano',60+i%7,90,origin+i*500,200) for i in range(count)]


@pytest.mark.parametrize('origin',(0,10**13))
def test_extreme_timestamps_never_allocate_a_grid_over_the_full_span(monkeypatch,origin):
    normal=phrase(origin)
    stray=Note('Acoustic Piano',72,90,10**13 if not origin else 0,100)
    expected=infer_clock(normal)
    original=np.fft.rfft;shapes=[]
    def bounded_fft(grid,n):
        assert len(grid)<=MAX_CLOCK_ANALYSIS_MS//20+2
        assert n<=131072
        shapes.append((len(grid),n))
        return original(grid,n)
    monkeypatch.setattr(np.fft,'rfft',bounded_fft)
    actual=infer_clock(normal+[stray])
    assert actual==expected and shapes


def test_sparse_attacks_years_apart_have_low_confidence_without_any_fft(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('Sparse distant attacks do not establish a pulse')
    monkeypatch.setattr(np.fft,'rfft',forbidden)
    notes=[Note('Acoustic Piano',60,90,i*10**13,100) for i in range(4)]
    assert infer_clock(notes).confidence==0


def test_reference_windows_skip_huge_empty_spans(monkeypatch):
    original=np.arange
    def bounded_range(start,stop=None,*args,**kwargs):
        assert (stop if stop is not None else start)<=100000
        return original(start,stop,*args,**kwargs) if stop is not None else original(start,*args,**kwargs)
    monkeypatch.setattr(np,'arange',bounded_range)
    result=reference_windows(phrase(0,24)+phrase(10**13,24))
    assert len(result['windows'])==2
    assert result['windows'][-1]['start_ms']>10**12


def fake_dataset(tmp_path,budget=5,limit=4):
    result=MelodyDataset.__new__(MelodyDataset)
    result.config=SimpleNamespace(WORKER_CACHE_NOTES=budget,WORKER_RECORD_CACHE=limit)
    result.cache=OrderedDict();result.rejected=set();result.prepared=tmp_path/'records.jsonl'
    return result


def test_cache_counts_drums_and_evicts_by_total_notes(tmp_path):
    dataset=fake_dataset(tmp_path)
    a=dict(notes=[0]*2,drums=[0]);b=dict(notes=[0]*3,drums=[0]);c=dict(notes=[0])
    dataset._cache_record('a',a);dataset._cache_record('b',b)
    assert list(dataset.cache)==['b']
    dataset._cache_record('c',c)
    assert list(dataset.cache)==['b','c']
    dataset.cache.move_to_end('b')
    dataset._cache_record('d',dict(notes=[0]*2))
    assert list(dataset.cache)==['d']


def test_large_records_remain_usable_without_entering_the_worker_cache(tmp_path):
    dataset=fake_dataset(tmp_path)
    record=dict(notes=[['Acoustic Piano',60,90,i*500,200] for i in range(20)],drums=[])
    dataset.prepared.write_text(json.dumps(record)+'\n')
    assert dataset._load_record(dict(kind='prepared',offset=0))==record
    assert not dataset.cache


@pytest.mark.parametrize('workers',(0,1))
def test_resume_preserves_prefetch_and_exact_remaining_sample_order(workers):
    options={'prefetch_factor':1} if workers else {}
    loader=DataLoader(TensorDataset(torch.arange(20)),batch_size=3,num_workers=workers,**options)
    resumed=resume_epoch_loader(loader,epoch=0,start_global_sample_idx=5)
    assert resumed.prefetch_factor==loader.prefetch_factor
    assert torch.cat([batch[0] for batch in resumed]).tolist()==list(range(5,20))


def test_shared_clock_moves_the_phase_with_chunks_and_keeps_expressive_offsets():
 notes=[Note('Acoustic Piano',60,90,int(i*500+(333 if off else 0)),100) for i in range(64) for off in (False,True)]
 clock=infer_clock(notes)
 assert 450<=clock.beat_ms<=550 and clock.confidence>0
 stats=groove_statistics(notes,BeatClock(500,0,1),0)
 assert .6<stats['swing_fraction']<.72
 assert sum(stats[f'beat_phase_{i}'] for i in range(8))==pytest.approx(1.)
 ref=dict(groove=reference_windows(notes),statistics={})
 guide=GrooveGuide(notes,ref,random.Random(42))
 a,b=guide.statistics(0,2000),guide.statistics(500,2500)
 assert a['beat_ms']==b['beat_ms'] and (a['bar_phase_sin'],a['bar_phase_cos'])!=(b['bar_phase_sin'],b['bar_phase_cos'])
 values,mask=style_tensors(a,'stem');assert len(values)==20 and mask.sum()==20
 # The same global tick has the same soft bias, irrespective of chunk origin.
 assert guide.onset_bias([1000],0,a)[0]==pytest.approx(guide.onset_bias([500],500,b)[0])
 assert guide.onset_bias([333],0,a)[0]!=guide.onset_bias([250],0,a)[0]

