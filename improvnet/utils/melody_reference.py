"""Clean reference crops with exact note boundaries and target exclusion."""
from dataclasses import replace
from improvnet.utils.melody import ordered, notes_to_tokens
from improvnet.utils.melody_chunks import chunk_end

PIANOS = frozenset(('Acoustic Piano', 'Electric Piano', 'Bright Piano', 'Honky-tonk Piano'))


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
        solo = len(required) == 1
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
