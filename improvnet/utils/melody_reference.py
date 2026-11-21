"""Clean reference crops with exact note boundaries and target exclusion."""
from dataclasses import replace
from improvnet.utils.melody import ordered, notes_to_tokens
from improvnet.utils.melody_chunks import chunk_end

PIANOS = frozenset(('Acoustic Piano', 'Electric Piano', 'Bright Piano', 'Honky-tonk Piano'))
TRIO_INSTRUMENTS = PIANOS | {'drum', 'Acoustic Bass'}


def trio_instruments(instruments):
    """Accept a whole piece only when its sole instruments are one piano, drums and double bass."""
    available=set(instruments)
    if len(available)!=3 or not {'drum','Acoustic Bass'}<=available or len(available & PIANOS)!=1:
        return set()
    return available


def reference_instruments(target_instruments, mode, available=None):
    """Exact output instrumentation or its piano/drums/double-bass subset."""
    instruments = set(target_instruments)
    if mode == 'exact':
        return instruments
    if mode == 'trio':
        wanted = instruments & TRIO_INSTRUMENTS
        if available is not None and wanted & PIANOS and not wanted & set(available) & PIANOS:
            alternatives = sorted(set(available) & PIANOS)
            if alternatives:
                wanted = (wanted - PIANOS) | {alternatives[0]}
        return wanted
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


def neighboring_references(notes,target_start,target_end,rng,*,count=2,max_tokens=512,
                           guard_ms=250,max_distance_ms=30000,instruments=None):
    """Two consecutive attack windows, both safely before or both after the target.

    Held releases can cross the boundary between the two references, but never
    the protected target. No onset group is skipped between the selected windows.
    """
    if count!=2:raise ValueError('Reference conditioning requires two consecutive windows')
    required=set(instruments or ())
    notes=ordered([n for n in notes if not required or n.instrument in required])
    pools=[[n for n in notes if target_start-max_distance_ms<=n.onset
            and n.onset+n.duration<=target_start-guard_ms],
           [n for n in notes if target_end+guard_ms<=n.onset<target_end+max_distance_ms]]
    rng.shuffle(pools)
    for pool in pools:
        starts=[i for i,n in enumerate(pool) if i==0 or n.onset!=pool[i-1].onset]
        rng.shuffle(starts)
        for begin in starts:
            try:
                middle=chunk_end(pool,begin,max_tokens)
                if middle==len(pool):continue
                stop=chunk_end(pool,middle,max_tokens)
            except ValueError:
                continue
            crops=(pool[begin:middle],pool[middle:stop])
            if required and any({n.instrument for n in crop}!=required for crop in crops):continue
            return [dict(notes=[replace(n,onset=n.onset-crop[0].onset) for n in crop],
                         start_ms=crop[0].onset,end_ms=max(n.onset+n.duration for n in crop),
                         view='solo' if len(required)==1 and 'drum' not in required else 'mixture')
                    for crop in crops]
    return []


def transpose_reference(notes,rng,semitones=7):
    """Change reference key only; preserve drums, instruments, timing and notes."""
    pitches=[n.pitch for n in notes if n.instrument!='drum']
    if not pitches:return list(notes),0
    low,high=max(-semitones,-min(pitches)),min(semitones,127-max(pitches))
    choices=[shift for shift in range(low,high+1) if shift]
    shift=rng.choice(choices) if choices else 0
    return [replace(n,pitch=n.pitch+shift) if n.instrument!='drum' else n for n in notes],shift


def reference_tempo(notes):
    """Estimate performed pulse; refuse the analysis fallback for empty/short music."""
    from improvnet.utils.melody_groove import infer_clock
    clock = infer_clock(ordered(notes))
    if clock.confidence <= 0:
        raise ValueError('Not enough rhythmic evidence to estimate tempo; keep reference tempo original')
    return 60000. / clock.beat_ms, clock.confidence


def retime_reference(notes, target_bpm):
    """Scale all reference stems together in memory on the tokenizer's 10-ms grid."""
    import math
    if not math.isfinite(target_bpm) or target_bpm <= 0:
        raise ValueError('Reference target BPM must be finite and positive')
    bpm, confidence = reference_tempo(notes)
    scale = bpm / target_bpm
    origin = min(n.onset for n in notes)
    retimed = [replace(n, onset=origin + round((n.onset-origin)*scale/10)*10,
                       duration=max(10,round(n.duration*scale/10)*10)) for n in notes]
    return ordered(retimed), dict(original_bpm=bpm, target_bpm=target_bpm,
                                  time_scale=scale, pulse_confidence=confidence)
