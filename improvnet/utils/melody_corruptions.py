"""Musical refinement and generated-history corruptions."""
from dataclasses import replace
from improvnet.utils.melody import notes_to_tokens, ordered, skyline







def _pitch_error(note, rng, semitones=7):
    """Choose a valid nonzero pitch error without clipping at MIDI boundaries."""
    shifts = [shift for shift in range(-semitones, semitones + 1)
              if shift and 0 <= note.pitch + shift <= 127]
    return note.pitch + rng.choice(shifts) if shifts else note.pitch


def _timing_errors(notes, probability, rng, step=10, jitter=50):
    """Bound onset/duration errors; simultaneous attacks share their onset move."""
    times = sorted({n.onset for n in notes})
    count = min(len(times), max(1, round(probability * len(times)))) if probability else 0
    chosen = set(rng.sample(times, count))
    shifts = [value for value in range(-jitter, jitter + step, step) if value] or [0]
    offsets, selected, previous = {}, set(), -step
    for i, onset in enumerate(times):
        moved = onset
        if onset in chosen:
            selected.add(onset)
            low = max(0, onset - jitter, previous + step)
            high = min(onset + jitter, times[i + 1] - step if i + 1 < len(times) else onset + jitter)
            moved = max(low, min(high, onset + rng.choice(shifts)))
        offsets[onset] = moved
        previous = moved
    result = []
    for n in notes:
        if n.onset not in selected:
            result.append(n)
            continue
        duration = max(step, n.duration + rng.choice(shifts))
        if jitter and offsets[n.onset] == n.onset and duration == n.duration:
            duration += step
        result.append(replace(n, onset=offsets[n.onset], duration=duration))
    return ordered(result)


def refinement_strength(notes, requested, config):
    """Signal mandatory ornament simplification as a floor on V5 strength."""
    if not notes:
        return requested
    from improvnet.utils.melody_ornaments import ornament_spans
    removed = sum(stop - start - 1 for start, stop in ornament_spans(ordered(notes)))
    return max(requested, removed / len(notes))


def refinement_source(notes, strength, rng, tokenizer, config, bounded=False):
    """Merge ornaments first, then apply an independent strength-scaled edit.

    Strength here is the requested *additional* edit fraction; the model's
    conditioning separately accounts for compulsory ornament simplification.
    """
    from improvnet.utils.melody_ornaments import merge_ornaments, merge_nonornaments, invert_contour
    changed = merge_ornaments(ordered(notes), 1., rng)
    if strength <= 0:
        return notes_to_tokens(changed, tokenizer)
    generic = config.GENERIC_MERGE_PROB
    choices = ['contour','generic_merge','pitch','timing']
    weights = [.50-generic,generic,.25,.25]
    count = min(config.MAX_CORRUPTIONS, sum(w > 0 for w in weights))
    for _ in range(rng.randint(1, count)):
        name = rng.choices(choices, weights)[0]
        selected = choices.index(name)
        choices.pop(selected); weights.pop(selected)
        if name == 'contour':
            changed = invert_contour(changed, strength, rng)
        elif name == 'generic_merge':
            changed = merge_nonornaments(changed, strength, rng)
        elif name == 'pitch':
            eligible = [i for i,n in enumerate(changed) if n.instrument != 'drum']
            amount = min(len(eligible), max(1, round(strength * len(eligible)))) if eligible else 0
            chosen = set(rng.sample(eligible, amount))
            changed = [replace(n,pitch=_pitch_error(n,rng,config.PITCH_CORRUPTION_SEMITONES))
                       if i in chosen else n for i,n in enumerate(changed)]
        else:
            changed = _timing_errors(changed, strength, rng, tokenizer.time_step_ms,
                                     config.TIMING_CORRUPTION_MS)
    return notes_to_tokens(changed, tokenizer)


def sample_stem_pitch_probability(rng, config):
    minimum = getattr(config, 'STEM_PITCH_SHIFT_MIN_PROB', 0.)
    maximum = getattr(config, 'STEM_PITCH_SHIFT_PROB', .05)
    clean = getattr(config, 'STEM_PITCH_CLEAN_PROB', .20)
    if not 0 <= minimum <= maximum <= 1 or not 0 <= clean <= 1:
        raise ValueError('Invalid dynamic stem pitch corruption rate')
    return 0. if rng.random() < clean else rng.uniform(minimum, maximum)


def corrupt_stem_source(notes, rng, tokenizer, config, window_ms, *, pitch_probability=None):
    """Mild retained-mix denoising, without changing source note identities.

    Exact attack groups move together by at most 50 ms and retain their order.
    Durations move independently by at most 50 ms. Rare pitch errors of up to
    ±7 semitones affect only accompaniment selected before jitter with 50 ms skyline
    grouping. Pitch shifts cannot reorder a chord's retained note anchors.
    Melody pitches, drum IDs, instruments and velocities are intact.
    """
    notes = ordered(notes)
    if not notes:
        return []
    step = tokenizer.time_step_ms
    jitter = getattr(config, 'STEM_TIMING_JITTER_MS', 50)
    probability = sample_stem_pitch_probability(rng, config) if pitch_probability is None else pitch_probability
    pitch_range = getattr(config, 'STEM_PITCH_SHIFT_SEMITONES', 7)
    if jitter < 0 or jitter > 50 or jitter % step or not 0 <= probability <= 1 or not 0 <= pitch_range <= 7:
        raise ValueError('Invalid stem denoising bounds')
    melody = {id(n) for n in skyline([n for n in notes if n.instrument != 'drum'],
                                     config.SKYLINE_TOLERANCE_MS)[0]}
    times = sorted({n.onset for n in notes})
    offsets, previous = {}, -step
    last_tick = max(0, ((window_ms - 1) // step) * step)
    for i, onset in enumerate(times):
        low = max(0, onset - jitter, previous + step)
        high = min(last_tick, onset + jitter,
                   times[i + 1] - step if i + 1 < len(times) else last_tick)
        # Source MIDI is quantized; protecting attack order retains a valid
        # inverse mapping to every clean group, including dense repeated notes.
        moved = max(low, min(high, onset + rng.randint(-jitter // step, jitter // step) * step))
        offsets[onset] = moved
        previous = moved
    occupied = {(n.instrument, n.pitch, n.onset) for n in notes}
    result = []
    for index, n in enumerate(notes):
        pitch = n.pitch
        if n.instrument != 'drum' and id(n) not in melody and pitch_range and rng.random() < probability:
            above = result[-1] if index and notes[index-1].onset == n.onset else None
            below = notes[index+1] if index+1 < len(notes) and notes[index+1].onset == n.onset else None
            shifts = [s for s in range(-pitch_range, pitch_range + 1) if s and 0 <= pitch + s <= 127
                      and (n.instrument, pitch + s, n.onset) not in occupied
                      and (above is None or (-above.pitch, above.instrument) <= (-pitch-s, n.instrument))
                      and (below is None or (-pitch-s, n.instrument) <= (-below.pitch, below.instrument))]
            if shifts:
                occupied.discard((n.instrument, pitch, n.onset))
                pitch += rng.choice(shifts)
                occupied.add((n.instrument, pitch, n.onset))
        duration = max(step, n.duration + rng.randint(-jitter // step, jitter // step) * step)
        result.append(replace(n, pitch=pitch, onset=offsets[n.onset], duration=duration))
    return ordered(result)




def corrupt_history(notes, rng, fraction=.10, addition_weight=2., deletion_weight=1., end_ms=None,
                    *, pitch_semitones=7, timing_ms=50, time_step_ms=10):
    """Mild note edits to a preceding full mix, never to the clean target.

    Added notes are twice as likely as deletions. Onsets stay strictly before
    the target's first onset; drum IDs are not transposed as pitched notes.
    """
    notes = ordered(notes)
    if not notes:
        return []
    if not 0 <= fraction <= 1 or min(addition_weight, deletion_weight) <= 0:
        raise ValueError('Invalid history corruption settings')
    if fraction == 0:
        return notes
    end_ms = end_ms if end_ms is not None else max(n.onset for n in notes) + 10
    if (not 0 <= pitch_semitones <= 7 or int(pitch_semitones) != pitch_semitones
            or not 0 <= timing_ms <= 50 or timing_ms % time_step_ms):
        raise ValueError('Invalid history pitch or timing error bounds')
    result, origins = list(notes), list(notes)
    def moved_onset(note):
        offset = rng.randint(-timing_ms // time_step_ms, timing_ms // time_step_ms) * time_step_ms
        return max(0, min(end_ms - time_step_ms, note.onset + offset))
    for _ in range(max(1, round(len(notes) * fraction))):
        action = rng.choices(('pitch', 'add', 'delete', 'timing'),
                             (1., addition_weight, deletion_weight, 1.))[0]
        i = rng.randrange(len(result))
        n, origin = result[i], origins[i]
        if action == 'add':
            pitch = n.pitch if n.instrument == 'drum' else _pitch_error(origin, rng, pitch_semitones)
            result.append(replace(n, pitch=pitch, onset=moved_onset(origin)))
            origins.append(origin)
        elif action == 'delete' and len(result) > 1:
            result.pop(i)
            origins.pop(i)
        elif action == 'pitch' and n.instrument != 'drum':
            result[i] = replace(n, pitch=_pitch_error(origin, rng, pitch_semitones))
        elif action == 'timing':
            onset = moved_onset(origin)
            duration = max(time_step_ms, origin.duration + rng.randint(-timing_ms // time_step_ms, timing_ms // time_step_ms) * time_step_ms)
            result[i] = replace(n, onset=onset, duration=duration)
    # Do not introduce repeated copies of an identical attack into the history.
    unique = {}
    for n in result:
        unique.setdefault((n.instrument, n.pitch, n.onset), n)
    return ordered(unique.values())

