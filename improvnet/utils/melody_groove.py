"""A common inferred pulse for pitched notes and drums, with expressive offsets.

This is catalogue analysis only; no pulse tokens or inference timing biases are emitted.
Half/double-tempo and downbeat ambiguity remain; four beats per bar is an
explicit working assumption, not an inferred time signature.
"""
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
