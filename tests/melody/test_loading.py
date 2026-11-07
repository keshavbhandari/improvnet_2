"""Bounded worker caches and exact DataLoader resume."""
import json
import pytest
import torch
from types import SimpleNamespace
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset
from improvnet.utils.melody_data import MelodyDataset
from improvnet.utils.ar_utils import resume_epoch_loader

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


