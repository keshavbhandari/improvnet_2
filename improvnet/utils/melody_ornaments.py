"""Conservative, score-free melodic simplification and short contour inversions.

These are candidate motifs, not musicological labels: a MIDI performance alone
cannot disambiguate an appoggiatura from an ordinary accented passing note.
"""
from dataclasses import replace
from statistics import median
from improvnet.utils.melody import ordered


def ornament_spans(notes):
    """Nonoverlapping [start, stop) runs with fast, nearby/returning pitches.

    Grace-like pickups, alternating neighbours, turns and short chromatic runs
    are merged only within one uninterrupted voice. Long rests, instrument
    changes, leaps and slow scalar phrases are protected.
    """
    if len(notes) < 2:
        return []
    gaps = [b.onset-a.onset for a,b in zip(notes,notes[1:]) if b.onset>a.onset]
    typical = median(gaps) if gaps else 250
    fast = min(220, max(90, typical * 1.25))
    spans, i = [], 0
    while i < len(notes)-1:
        j = i+1
        while j < min(len(notes), i+12):
            a,b=notes[j-1],notes[j]
            if (a.instrument!=b.instrument or not 0<b.onset-a.onset<=fast
                    or b.onset>a.onset+a.duration+60 or abs(b.pitch-a.pitch)>2):
                break
            j+=1
        group=notes[i:j]
        steps=[b.pitch-a.pitch for a,b in zip(group,group[1:])]
        pitches=[n.pitch for n in group]
        returning=len(group)>=3 and max(pitches)-min(pitches)<=4 and any(a*b<0 for a,b in zip(steps,steps[1:]))
        chromatic=len(group)>=3 and all(abs(d)==1 for d in steps) and abs(sum(steps))<=7
        # A short lead-in to a longer note can include a two-note grace figure.
        grace=len(group)>=2 and group[0].duration<=min(120,typical*.5) and group[-1].duration>=2*group[0].duration
        if returning or chromatic or grace:
            spans.append((i,j));i=j
        else:
            i+=1
    return spans


def merge_ornaments(notes, strength, rng):
    notes=ordered(notes); result=[]; cursor=0
    for start,stop in ornament_spans(notes):
        result.extend(notes[cursor:start])
        group=notes[start:stop]
        if rng.random()<strength:
            result.append(replace(group[0],duration=max(n.onset+n.duration for n in group)-group[0].onset))
        else:
            result.extend(group)
        cursor=stop
    return ordered(result+notes[cursor:])


def merge_nonornaments(notes, strength, rng):
    """Occasionally omit one ordinary 2–4 note run, protecting motif candidates.

    This deliberately covers plain connected melody as well as the faster
    ornament heuristic. It never bridges a long rest, crosses an instrument
    change or removes any note already identified as an ornament.
    """
    notes = ordered(notes)
    if len(notes) < 2 or strength <= 0 or rng.random() >= strength:
        return notes
    protected = {i for start, stop in ornament_spans(notes) for i in range(start, stop)}
    candidates = []
    for start in range(len(notes) - 1):
        if start in protected:
            continue
        stop = start + 1
        while stop < min(len(notes), start + 4) and stop not in protected:
            a, b = notes[stop - 1], notes[stop]
            if (a.instrument != b.instrument or not 0 < b.onset - a.onset <= 800
                    or b.onset > a.onset + a.duration + 150 or abs(b.pitch - a.pitch) > 12):
                break
            stop += 1
        if stop - start >= 2:
            candidates.append((start, stop))
    if not candidates:
        return notes
    start, stop = rng.choice(candidates)
    stop = rng.randint(start + 2, stop)
    group = notes[start:stop]
    held = replace(group[0], duration=max(n.onset + n.duration for n in group) - group[0].onset)
    return ordered(notes[:start] + [held] + notes[stop:])


def invert_contour(notes, strength, rng):
    """Reflect 3–10 consecutive notes around their first pitch, without clipping."""
    notes=ordered(notes)
    if len(notes)<3 or strength<=0:
        return notes
    count=max(1,round(strength*len(notes)/8)); used=set()
    for _ in range(count):
        start=rng.randrange(len(notes)-2);stop=min(len(notes),start+rng.randint(3,10))
        group=notes[start:stop]
        if any(i in used for i in range(start,stop)) or any(
                a.instrument!=b.instrument or not 0<b.onset-a.onset<=1000 or b.onset>a.onset+a.duration+150
                for a,b in zip(group,group[1:])):
            continue
        pitches=[2*group[0].pitch-n.pitch for n in group]
        if not all(0<=p<=127 for p in pitches):
            continue
        notes[start:stop]=[replace(n,pitch=p) for n,p in zip(group,pitches)]
        used.update(range(start,stop))
    return notes
