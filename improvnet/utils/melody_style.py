"""Pitch-free local rhythm controls shared by harmonization and stem generation."""
import json
import math
from pathlib import Path
import torch
from improvnet.utils.melody import attack_groups
from improvnet.utils.melody_groove import GROOVE_FEATURES

IOI_FEATURES=('ioi_mean_ms','ioi_std_ms','ioi_q25_ms','ioi_q75_ms','ioi_lag1_correlation')
STYLE_FEATURES=IOI_FEATURES+GROOVE_FEATURES
STYLE_DEFINITIONS={name:name.replace('_',' ') for name in STYLE_FEATURES}


def _quantile(values,q):
    pos=(len(values)-1)*q;lo,hi=math.floor(pos),math.ceil(pos)
    return values[lo]*(hi-pos)+values[hi]*(pos-lo) if lo!=hi else values[lo]


def performance_statistics(notes,melody=None,duration_ms=None,tolerance_ms=50):
    attacks=[g[0].onset for g in attack_groups([n for n in notes if n.instrument!='drum'],tolerance_ms)]
    intervals=[b-a for a,b in zip(attacks,attacks[1:]) if b>a]
    if not intervals:
        return {}
    mean=sum(intervals)/len(intervals);values=sorted(intervals)
    stats=dict(zip(IOI_FEATURES[:4],(mean,math.sqrt(sum((x-mean)**2 for x in intervals)/len(intervals)),_quantile(values,.25),_quantile(values,.75))))
    if len(intervals)>=3:
        a,b=intervals[:-1],intervals[1:];ma,mb=sum(a)/len(a),sum(b)/len(b)
        denom=math.sqrt(sum((x-ma)**2 for x in a)*sum((x-mb)**2 for x in b))
        stats[IOI_FEATURES[4]]=max(-1.,min(1.,sum((x-ma)*(y-mb) for x,y in zip(a,b))/denom)) if denom else 0.
    return stats


def style_tensors(statistics=None,task='harmonize',visible=True,feature_names=STYLE_FEATURES):
    values,mask=torch.zeros(len(feature_names)),torch.zeros(len(feature_names))
    if not visible or not statistics or task not in ('harmonize','rhythm','stem'):
        return values,mask
    for i,name in enumerate(feature_names):
        if name not in statistics:
            continue
        value=float(statistics[name])
        if not math.isfinite(value):
            raise ValueError(f'Invalid rhythm feature {name}: {value}')
        if name.endswith('_ms'):
            if value<0 and name!='timing_offset_ms':
                raise ValueError(f'Negative rhythm feature {name}')
            value=math.copysign(math.log1p(abs(value)/1000),value)
        elif name in ('ioi_lag1_correlation','bar_phase_sin','bar_phase_cos'):
            if not -1<=value<=1:
                raise ValueError(f'Invalid bounded rhythm feature {name}')
        elif not 0<=value<=1:
            raise ValueError(f'Invalid rhythm fraction {name}')
        values[i]=value;mask[i]=1
    return values,mask


def choose_reference(path,genre,rng,required_tasks=('harmonize',),source_notes=None):
    from improvnet.utils.melody_groove import infer_clock
    payload=json.loads(Path(path).read_text())
    if payload.get('version') not in (1,2):
        raise ValueError('Reference rhythm feature schema differs')
    examples=[x for x in payload['examples'] if x['genre']==genre and x['split']=='train'
              and all(style_tensors(x.get('melody_statistics',x['statistics']),task)[1].any() for task in required_tasks)]
    if not examples:
        raise ValueError(f'No suitable {genre} training reference; provide --style-reference')
    if source_notes and any(x.get('groove',{}).get('windows') for x in examples):
        clock=infer_clock(source_notes)
        # A single coherent performance, weighted by pulse compatibility. Equal
        # piece weighting avoids favouring very long recordings with more windows.
        examples=[x for x in examples if x.get('groove',{}).get('windows')]
        weights=[.05+math.exp(-abs(math.log(x['groove']['beat_ms']/clock.beat_ms))*3)
                 *max(.1,x['groove'].get('confidence',0.)) for x in examples]
        return rng.choices(examples,weights)[0]
    return rng.choice(examples)
