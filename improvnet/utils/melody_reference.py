"""Clean reference crops with exact note boundaries and target exclusion."""
from dataclasses import replace
from improvnet.utils.melody import ordered, notes_to_tokens
from improvnet.utils.melody_chunks import chunk_end

PIANOS = frozenset(('Acoustic Piano', 'Electric Piano', 'Bright Piano', 'Honky-tonk Piano'))
TRIO_INSTRUMENTS = PIANOS | {'drum', 'Acoustic Bass'}


def reference_instruments(target_instruments, mode):
    """Exact output instrumentation or its piano/drums/double-bass subset."""
    instruments = set(target_instruments)
    if mode == 'exact':
        return instruments
    if mode == 'trio':
        return instruments & TRIO_INSTRUMENTS
    raise ValueError('Reference mode must be exact or trio')


def crop_notes(notes, rng, max_tokens=512, *, tail=False):
    notes = ordered(notes)
    if not notes:
        raise ValueError('Empty reference view')
    starts = [i for i, n in enumerate(notes) if i == 0 or n.onset != notes[i-1].onset]
    # Try complete onset groups, including dense passages; never split chords.
    if tail:
        # At most max_tokens/4 notes can fit; prefer the closest preceding crop.
        starts = [i for i in starts if len(notes)-i <= (max_tokens-2)//4] or starts[-1:]
    else:
        rng.shuffle(starts)
    for begin in starts:
        try:
            stop = chunk_end(notes, begin, max_tokens)
        except ValueError:
            continue
        selected = notes[begin:stop]
        start = selected[0].onset
        end = max(n.onset+n.duration for n in selected)
        return [replace(n, onset=n.onset-start) for n in selected], start, end
    raise ValueError('No complete onset group fits reference budget')


def neighboring_reference(notes, target_start, target_end, rng, *, max_tokens=512,
                           guard_ms=250, max_distance_ms=30000, solo=None, instruments=None):
    """Return an independent preceding/following view, never any target sound.

    target_end must include target releases. Previous crossing notes are excluded
    rather than clipped, so no portion of a target-overlapping note can enter.
    """
    required = set(instruments or ())
    if solo is None:
        solo = len(required) == 1 and 'drum' not in required
    notes = ordered([n for n in notes if not required or n.instrument in required])
    if solo and notes:
        pitched = sorted({n.instrument for n in notes if n.instrument != 'drum'})
        preferred = [i for i in pitched if i in PIANOS]
        instruments = preferred or pitched
        if not instruments:
            return None
        instrument = rng.choice(instruments)
        notes = [n for n in notes if n.instrument == instrument]
    left = [n for n in notes if target_start-max_distance_ms <= n.onset
            and n.onset+n.duration <= target_start-guard_ms]
    right = [n for n in notes if target_end+guard_ms <= n.onset < target_end+max_distance_ms]
    pools = [(left, True), (right, False)]
    if rng.random() < .5:
        pools.reverse()
    for pool, tail in pools:
        if not pool:
            continue
        # Keep the next view local instead of choosing anywhere in a long tail.
        if not tail:
            pool = pool[:max_tokens]
        for attempt in range(16):
            try:
                crop, start, end = crop_notes(pool, rng, max_tokens, tail=tail and attempt==0)
            except ValueError:
                continue
            if required and {n.instrument for n in crop} != required:
                continue
            return dict(notes=crop, start_ms=start, end_ms=end, view='solo' if solo else 'mixture')
    return None


def reference_ids(reference, tokenizer):
    if reference is None:
        return []
    return [tokenizer.tok_to_id[t] for t in notes_to_tokens(reference['notes'], tokenizer)]


def neighboring_references(notes,target_start,target_end,rng,*,count=1,**kwargs):
    """Distinct nearby windows, excluding target attacks and held releases."""
    remaining=list(notes)
    result=[]
    for _ in range(count):
        ref=neighboring_reference(remaining,target_start,target_end,rng,**kwargs)
        if ref is None:break
        result.append(ref)
        left,right=ref['start_ms'],ref['end_ms']
        remaining=[n for n in remaining if n.onset+n.duration<=left or n.onset>=right]
    return sorted(result,key=lambda r:r['start_ms'])


def transpose_reference(notes,rng,semitones=7):
    """Change reference key only; preserve drums, instruments, timing and notes."""
    pitches=[n.pitch for n in notes if n.instrument!='drum']
    if not pitches:return list(notes),0
    low,high=max(-semitones,-min(pitches)),min(semitones,127-max(pitches))
    choices=[shift for shift in range(low,high+1) if shift]
    shift=rng.choice(choices) if choices else 0
    return [replace(n,pitch=n.pitch+shift) if n.instrument!='drum' else n for n in notes],shift
