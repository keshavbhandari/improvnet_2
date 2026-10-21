"""A common inferred pulse for pitched notes and drums, with expressive offsets.

The pulse is estimated from attacks when beat annotations are unavailable.
Half/double-tempo and downbeat ambiguity remain; four beats per bar is an
explicit working assumption, not an inferred time signature.
"""
from collections import defaultdict
from dataclasses import dataclass
import math
import numpy as np
from improvnet.utils.melody import attack_groups

# Bound temporary FFT storage independently of full MIDI duration. A corrupt
# distant timestamp must not turn a short excerpt into a multi-terabyte grid.
MAX_CLOCK_ANALYSIS_MS=20*60*1000

GROOVE_FEATURES=('beat_ms','bar_phase_sin','bar_phase_cos','swing_fraction',
                 'timing_offset_ms','timing_spread_ms','pulse_confidence',
                 *(f'beat_phase_{i}' for i in range(8)))


@dataclass(frozen=True)
class BeatClock:
    beat_ms: float
    origin_ms: float
    confidence: float

    def phase(self,time_ms):
        return (time_ms-self.origin_ms)/self.beat_ms


def infer_clock(notes):
    attacks=[g[0].onset for g in attack_groups(notes,50)]
    if attacks and attacks[-1]-attacks[0]>MAX_CLOCK_ANALYSIS_MS:
        # Select the densest bounded interval, retaining usable music even if
        # the file contains a distant stray attack or a very long silent gap.
        left=0;best_start=0;best_stop=0
        for right,onset in enumerate(attacks):
            while onset-attacks[left]>MAX_CLOCK_ANALYSIS_MS:
                left+=1
            if right-left+1>best_stop-best_start:
                best_start,best_stop=left,right+1
        attacks=attacks[best_start:best_stop]
    if len(attacks)<4 or attacks[-1]-attacks[0]<500:
        return BeatClock(500.,float(attacks[0] if attacks else 0),0.)
    # Autocorrelation avoids O(N²) comparisons on long performances. A 20-ms
    # analysis grid never changes the original 10-ms MIDI event positions.
    step=20.; origin=attacks[0]
    grid=np.zeros(int((attacks[-1]-origin)/step)+2)
    for onset in attacks:
        grid[round((onset-origin)/step)]+=1
    n=1<<(2*len(grid)-1).bit_length()
    fft=np.fft.rfft(grid,n);corr=np.fft.irfft(fft*fft.conj(),n)[:len(grid)]
    candidates=np.arange(300.,1001.,20.)
    scores=[]
    for period in candidates:
        raw=0.
        for multiple,weight in ((1,1.),(2,.55),(3,.25),(4,.15)):
            lag=round(period*multiple/step)
            if lag<len(corr):
                raw+=weight*max(corr[max(1,lag-1):min(len(corr),lag+2)],default=0.)
        raw/=max(1.,corr[0])
        scores.append(raw*(.85+.15*math.exp(-abs(math.log(period/500.)))))
    idx=int(np.argmax(scores));period=float(candidates[idx]);score=scores[idx]
    # Circular mean of attack phases estimates pulse origin without snapping
    # every event; a weak mean leaves the first attack as a fallback anchor.
    angles=np.asarray(attacks)*2*np.pi/period;z=np.exp(1j*angles).mean()
    residue=(np.angle(z)*period/(2*np.pi))%period
    clock_origin=origin-((origin-residue+period/2)%period-period/2)
    confidence=min(1.,max(0.,score/1.5))*min(1.,len(attacks)/16)
    return BeatClock(period,float(clock_origin),float(confidence))


def groove_statistics(notes,clock,start_ms=0):
    attacks=[g[0].onset for g in attack_groups(notes,50)]
    if not attacks:
        return {}
    phases=np.asarray([clock.phase(t)%1 for t in attacks])
    off=phases[(phases>=.4)&(phases<=.8)]
    swing=float(np.clip(np.median(off),.5,.75)) if len(off)>=3 else .5
    centers=np.asarray((0.,.125,.25,.375,swing,.625,.75,.875))
    distances=(phases[:,None]-centers[None,:]+.5)%1-.5
    bins=np.abs(distances).argmin(axis=1)
    offsets=distances[np.arange(len(bins)),bins]*clock.beat_ms
    hist=np.bincount(bins,minlength=8).astype(float)+.25
    hist/=hist.sum()
    phase=2*np.pi*((clock.phase(start_ms)/4)%1)
    return dict(beat_ms=clock.beat_ms,bar_phase_sin=math.sin(phase),bar_phase_cos=math.cos(phase),
                swing_fraction=swing,timing_offset_ms=float(np.median(offsets)),
                timing_spread_ms=float(np.clip(np.std(offsets),10,100)),pulse_confidence=clock.confidence,
                **dict(zip(GROOVE_FEATURES[-8:],hist.tolist())))


def reference_windows(notes,melody=None):
    """Coherent eight-bar windows with sufficient attacks, never a global blend."""
    from improvnet.utils.melody_style import performance_statistics
    clock=infer_clock(notes);melody=melody or [n for n in notes if n.instrument!='drum']
    duration=max((n.onset+n.duration for n in notes),default=0)
    width=clock.beat_ms*32; result=[]
    # Visit only occupied windows; allocating a dense list over a stray
    # year-long gap would defeat the bounded pulse analysis above.
    occupied=defaultdict(list)
    limit=max(1,duration-width/2)
    for note in melody:
        block=int(note.onset//width)
        if block*width<limit:
            occupied[block].append(note)
    for block,absolute in sorted(occupied.items()):
        start=block*width
        if len(absolute)<12:
            continue
        stats=performance_statistics(absolute)
        stats.update(groove_statistics(absolute,clock,float(start)))
        result.append(dict(start_ms=int(start),end_ms=int(start+width),statistics=stats))
    return dict(beat_ms=clock.beat_ms,origin_ms=clock.origin_ms,confidence=clock.confidence,
                meter_assumption=4,windows=result)


class GrooveGuide:
    """One reference and one source clock shared by every generation stage."""
    def __init__(self,source_notes,reference,rng):
        self.clock=infer_clock(source_notes)
        self.reference=reference or {}
        windows=self.reference.get('groove',{}).get('windows',[])
        self.windows=windows
        self.offset=rng.randrange(len(windows)) if windows else 0

    def statistics(self,start_ms,end_ms):
        if not self.reference:
            return {}
        if self.windows:
            bar=int(max(0,self.clock.phase(start_ms))//32)
            item=self.windows[(self.offset+bar)%len(self.windows)]
            stats=dict(item['statistics']) if isinstance(item,dict) else dict(zip(self.reference['groove']['feature_names'],item[2:]))
        else:
            stats=dict(self.reference.get('melody_statistics',self.reference.get('statistics',{})))
        ref_period=stats.get('beat_ms',self.reference.get('groove',{}).get('beat_ms',self.clock.beat_ms))
        ratio=self.clock.beat_ms/max(1.,ref_period)
        for key in ('ioi_mean_ms','ioi_std_ms','ioi_q25_ms','ioi_q75_ms','timing_offset_ms','timing_spread_ms'):
            if key in stats:
                stats[key]*=ratio
        phase=2*math.pi*((self.clock.phase(start_ms)/4)%1)
        stats.update(beat_ms=self.clock.beat_ms,bar_phase_sin=math.sin(phase),bar_phase_cos=math.cos(phase),
                     pulse_confidence=min(self.clock.confidence,stats.get('pulse_confidence',0.)))
        return stats

    def onset_bias(self,onsets,start_ms,statistics):
        """A soft, shared beat-phase prior; never quantize expressive offsets."""
        confidence=statistics.get('pulse_confidence',0.)
        if confidence<.15 or not all(k in statistics for k in GROOVE_FEATURES[-8:]):
            return np.zeros(len(onsets))
        phase=(np.asarray(onsets)+start_ms-self.clock.origin_ms)/self.clock.beat_ms%1
        centers=np.asarray((0.,.125,.25,.375,statistics.get('swing_fraction',.5),.625,.75,.875))
        centers=(centers+statistics.get('timing_offset_ms',0.)/self.clock.beat_ms)%1
        width=max(.035,min(.15,statistics.get('timing_spread_ms',30.)/self.clock.beat_ms))
        dist=(phase[:,None]-centers[None,:]+.5)%1-.5
        weights=np.asarray([statistics[k] for k in GROOVE_FEATURES[-8:]])
        density=(np.exp(-.5*(dist/width)**2)*weights).sum(axis=1)
        # Uniform floor prevents hard exclusion of syncopations or pickups.
        return .5*confidence*np.log(.25+density/max(.01,float(weights.mean())))
