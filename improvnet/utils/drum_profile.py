"""Coarse percussion controls from the conditioning reference during training and inference."""
import math
from statistics import mean, pstdev, median
import torch
from improvnet.utils.melody_stems import DRUM_KIT_PITCHES, DRUM_KIT_ALIASES

DRUM_PROFILE_DIM = 400  # 128 presence + 128 usage + 128 rates + 8 summaries + 8 IOI bins.
DRUM_PROFILE_SCHEMA = 'presence128_usage128_log_rate128_summary8_ioi8_v1'
IOI_EDGES = (0., .05, .1, .2, .3, .5, .75, 1., float('inf'))


def drum_profile(notes, duration_ms=None):
    """Keep all sound IDs. Quantize continuous controls to avoid encoding note sequences."""
    drums=[n for n in notes if n.instrument=='drum']
    result=torch.zeros(DRUM_PROFILE_DIM)
    if not drums:return result
    duration=max(10,duration_ms if duration_ms is not None else
                 max(n.onset+n.duration for n in notes)-min(n.onset for n in notes))/1000
    counts=torch.bincount(torch.tensor([n.pitch for n in drums]),minlength=128).float()
    result[:128]=counts.gt(0)
    result[128:256]=(counts/counts.sum()*32).round()/32
    result[256:384]=torch.log1p((counts/duration*4).round()/4)
    groups=[]
    for n in sorted(drums,key=lambda n:n.onset):
        if not groups or n.onset-groups[-1][0]>50:groups.append([n.onset])
        else:groups[-1].append(n.onset)
    onsets=[g[0]/1000 for g in groups]
    intervals=[b-a for a,b in zip(onsets,onsets[1:]) if b>a]
    bins=max(1,math.ceil(duration))
    origin=min(n.onset for n in notes)/1000
    coverage=len({min(bins-1,max(0,int(t-origin))) for t in onsets})/bins
    ratios=[max(a,b)/min(a,b) for a,b in zip(intervals[::2],intervals[1::2]) if min(a,b)>.03 and max(a,b)/min(a,b)<=3]
    summaries=[math.log1p(len(drums)/duration),mean(map(len,groups)),
               mean(intervals) if intervals else 0.,pstdev(intervals) if intervals else 0.,
               coverage,median(ratios) if len(ratios)>=3 else 0.,
               float(len(ratios)>=3),len(groups)/duration]
    result[384:392]=torch.tensor(summaries).mul(16).round()/16
    if intervals:
        hist=torch.tensor([sum(lo<=t<hi for t in intervals) for lo,hi in zip(IOI_EDGES,IOI_EDGES[1:])]).float()
        result[392:]=(hist/hist.sum()*32).round()/32
    return result


def reference_kit(notes):
    """Explicit inference palette: standard-kit sounds present in the reference."""
    return sorted({DRUM_KIT_ALIASES.get(n.pitch,n.pitch) for n in notes if n.instrument=='drum'} & DRUM_KIT_PITCHES)


def profile_report(profile):
    values=profile.detach().cpu()
    return dict(sounds=values[:128].nonzero().flatten().tolist(),
                attacks_per_second=round(math.expm1(float(values[384])),3),
                simultaneous_density=float(values[385]),coverage=float(values[388]))
