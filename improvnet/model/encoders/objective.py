"""Symmetric global InfoNCE with time-aware same-piece negative eligibility."""
import math
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from torch.nn import functional as F


def gather_metadata(value,distributed):
    if not distributed or not dist.is_initialized():return value
    output=[torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(output,value)
    return torch.cat(output)


def beat_loss(a,b,na,nb,piece_ids,bounds,negative_piece_ids,negative_bounds,*,temperature=.1,distributed=False,gap_ms=1000,record_ids=None,negative_record_ids=None):
    if temperature<=0 or not math.isfinite(temperature):raise ValueError('Invalid temperature')
    if a.ndim!=2 or any(x.shape!=a.shape for x in (b,na,nb)) or bounds.shape!=(len(a),2) or negative_bounds.shape!=bounds.shape:
        raise ValueError('Invalid beat features or interval metadata')
    if not ((bounds[:,1]>bounds[:,0]).all() and (negative_bounds[:,1]>negative_bounds[:,0]).all()):
        raise ValueError('Nonempty reference intervals required')
    features=[F.normalize(x.float(),dim=-1) for x in (a,b,na,nb)]
    use_ddp=distributed and dist.is_initialized()
    world=dist.get_world_size() if use_ddp else 1
    rank=dist.get_rank() if use_ddp else 0
    global_features=[torch.cat(all_gather(x)) if use_ddp else x for x in features]
    ids=gather_metadata(piece_ids,use_ddp);nids=gather_metadata(negative_piece_ids,use_ddp)
    times=gather_metadata(bounds,use_ddp);ntimes=gather_metadata(negative_bounds,use_ddp)
    candidates_ids=torch.cat((ids,nids));candidates_times=torch.cat((times,ntimes))
    other=piece_ids[:,None]!=candidates_ids[None,:]
    separate=(bounds[:,1,None]+gap_ms<=candidates_times[None,:,0]) | (candidates_times[None,:,1]+gap_ms<=bounds[:,0,None])
    record_ids=piece_ids if record_ids is None else record_ids
    negative_record_ids=negative_piece_ids if negative_record_ids is None else negative_record_ids
    records=torch.cat((gather_metadata(record_ids,use_ddp),gather_metadata(negative_record_ids,use_ddp)))
    same_record=record_ids[:,None]==records[None,:]
    negatives=(other & ~same_record)|(same_record & separate)
    targets=torch.arange(len(a),device=a.device)+rank*len(a)
    valid=negatives.clone();valid[torch.arange(len(a),device=a.device),targets]=True
    usable=negatives.any(1)
    count=usable.sum();total=count.clone()
    if use_ddp:dist.all_reduce(total)
    if not total.item():raise ValueError('No separate-region or different-piece beat negatives')
    aa,bb,naa,nbb=global_features
    ab=features[0]@torch.cat((bb,nbb)).T
    ba=features[1]@torch.cat((aa,naa)).T
    logits=[(x/temperature).masked_fill(~valid,-torch.inf) for x in (ab,ba)]
    rows=(F.cross_entropy(logits[0],targets,reduction='none')+F.cross_entropy(logits[1],targets,reduction='none'))/2
    loss=rows[usable].sum()*world/total
    with torch.no_grad():
        i=torch.arange(len(a),device=a.device)
        positive=(ab[i,targets]+ba[i,targets])/2
        correct=((logits[0].argmax(1)==targets).float()+(logits[1].argmax(1)==targets).float())/2
        hardest=(ab.masked_fill(~negatives,-torch.inf).max(1).values+ba.masked_fill(~negatives,-torch.inf).max(1).values)/2
        stats=torch.stack((rows[usable].sum(),correct[usable].sum(),positive[usable].sum(),
            (ab[negatives].sum()+ba[negatives].sum())/2,negatives.sum().float(),(positive[usable]-hardest[usable]).sum(),count.float()))
    return loss,stats
