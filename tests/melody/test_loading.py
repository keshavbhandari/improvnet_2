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


@pytest.mark.parametrize('workers',[0,1])
def test_curriculum_schedule_and_prefetched_resume_indices(tmp_path,workers):
    from types import SimpleNamespace
    from torch.utils.data import Dataset,DistributedSampler
    from improvnet.utils.melody_data import jazz_weight,training_epoch_loader
    config=SimpleNamespace(JAZZ_RAMP_START_STEP=10000,JAZZ_RAMP_END_STEP=15000,JAZZ_BLUES_REPEATS=4,ACCUM_STEPS=2)
    assert [jazz_weight(step,config) for step in (0,9999,10000,12500,15000,20000)]==[1,1,1,2.5,4,4]
    class Indexed(Dataset):
        def __init__(self):self.config=config
        def __len__(self):return 24
        def __getitem__(self,index):return torch.tensor(index)
    dataset=Indexed()
    sampler=DistributedSampler(dataset,num_replicas=1,rank=0,seed=42)
    loader=DataLoader(dataset,batch_size=3,sampler=sampler,num_workers=workers,**({'prefetch_factor':1} if workers else {}))
    complete=list(training_epoch_loader(loader,2,0,9998))
    resumed=list(training_epoch_loader(loader,2,12,10000))
    torch.testing.assert_close(torch.cat(complete[4:]),torch.cat(resumed))
    assert [int(batch[0,1]) for batch in complete]==[9998,9998,9999,9999,10000,10000,10001,10001]
