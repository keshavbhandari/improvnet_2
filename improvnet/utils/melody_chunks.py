"""Note-aware token budgets shared by training and left-to-right inference."""
from dataclasses import replace
import math

from improvnet.utils.melody import notes_to_tokens, ordered, TASKS, tokens_to_notes, PREFIX_TOKENS


def token_count(notes, start=0, stop=None):
    """Exact serialized length after rebasing the first attack to zero."""
    stop = len(notes) if stop is None else stop
    if stop <= start:
        return 2  # <S>, <E>
    return 4 * (stop - start) + (notes[stop - 1].onset - notes[start].onset) // 5000 + 2


def group_end(notes, index):
    end = index + 1
    while end < len(notes) and notes[end].onset == notes[index].onset:
        end += 1
    return end


def chunk_end(notes, start, max_tokens):
    """Largest prefix of whole onset groups that fits; never split a chord."""
    end = start
    while end < len(notes):
        candidate = group_end(notes, end)
        if token_count(notes, start, candidate) > max_tokens:
            break
        end = candidate
    if end == start:
        raise ValueError('A complete onset group does not fit the chunk token budget')
    return end


def chunk_starts(notes, max_tokens):
    starts, start = [], 0
    while start < len(notes):
        starts.append(start)
        start = chunk_end(notes, start, max_tokens)
    return starts


def shorten_chunk(notes, start, stop, fraction):
    """Shrink an oversized example by whole onset groups."""
    end = max(start + 1, start + int((stop - start) * min(.95, fraction)))
    while end > start and end < len(notes) and notes[end].onset == notes[end - 1].onset:
        end -= 1
    if end <= start or end >= stop:
        raise ValueError('A complete onset group cannot fit the combined context budget')
    return end


def chunk_bounds(notes, start, stop):
    begin = notes[start].onset
    # Preserve long gaps between chunks rather than inviting invented attacks in
    # an arbitrarily long silence. A held note keeps its full duration.
    end = max(n.onset + n.duration for n in notes[start:stop])
    if stop < len(notes):
        end = min(end, notes[stop].onset)
    return begin, max(notes[stop - 1].onset + 10, end)


def stem_bounds(notes,start,stop):
    """A missing stem can play during a rest before the next source chunk."""
    begin,end=chunk_bounds(notes,start,stop)
    if stop<len(notes):
        end=notes[stop].onset
    return begin,end


def history_tokens(notes, max_tokens, tokenizer):
    """Fit the most recent complete onset groups."""
    if not notes or max_tokens < 6:
        return []
    notes = ordered(notes)
    start = len(notes)
    while start > 0:
        previous = start - 1
        while previous > 0 and notes[previous-1].onset == notes[start-1].onset:
            previous -= 1
        if token_count(notes, previous) > max_tokens:
            break
        start = previous
    if start == len(notes):
        return []
    origin = notes[start].onset
    return notes_to_tokens([replace(n,onset=n.onset-origin) for n in notes[start:]], tokenizer)


def trim_history(tokens, max_tokens, tokenizer):
    """Drop oldest complete groups to prioritize an intact source and response."""
    if len(tokens) <= max_tokens:
        return tokens
    return history_tokens(tokens_to_notes(tokens, tokenizer),
                          max(0,max_tokens),tokenizer)


def validate_lengths(config):
    if config.REFERENCE_WINDOWS!=1:raise ValueError('V12 training uses exactly one reference window')
    if not 0<=config.REFERENCE_PITCH_SEMITONES<=7 or int(config.REFERENCE_PITCH_SEMITONES)!=config.REFERENCE_PITCH_SEMITONES:
        raise ValueError('Reference pitch augmentation must use integer semitones up to seven')
    if not 0<=config.JAZZ_RAMP_START_STEP<config.JAZZ_RAMP_END_STEP or config.JAZZ_BLUES_REPEATS<1:
        raise ValueError('Invalid jazz curriculum')
    for name in ('REFERENCE_WINDOWS','REFERENCE_LOCAL_BINS','REFERENCE_ENCODER_BATCH_SIZE'):
        if getattr(config,name)<1 or int(getattr(config,name))!=getattr(config,name):
            raise ValueError(f'{name} must be a positive integer')
    if not PREFIX_TOKENS+16 <= config.SEQ_LEN <= 8192:
        raise ValueError('SEQ_LEN must fit the 34-position prefix and music within 8192 positions')
    if not 6 <= config.CHUNK_TOKENS < config.SEQ_LEN - 6:
        raise ValueError('CHUNK_TOKENS must leave room for response and separators')
    if not 0 <= config.HISTORY_TOKENS < config.SEQ_LEN or not 0 <= config.HISTORY_PROB <= 1:
        raise ValueError('Invalid history token cap or probability')
    for name in ('POLY_HISTORY_PROB', 'HISTORY_CORRUPTION_PROB', 'REFERENCE_PROB',
                 'STEM_DRUM_PROB', 'STEM_SOURCE_CORRUPTION_PROB', 'AR_MIXTURE_PROB', 'STEM_PITCH_SHIFT_PROB', 'GENERIC_MERGE_PROB',
                 'STEM_PITCH_SHIFT_MIN_PROB', 'STEM_PITCH_CLEAN_PROB', 'HARMONIZE_FULL_MASK_PROB',
                 'KIT_DRUM_EXAMPLE_PROB','REFERENCE_EXACT_PROB'):
        if not 0 <= getattr(config, name) <= 1:
            raise ValueError(f'{name} must be in [0,1]')
    if len(config.TASK_WEIGHTS) != len(TASKS) or any(not math.isfinite(x) or x < 0 for x in config.TASK_WEIGHTS) or sum(config.TASK_WEIGHTS) <= 0:
        raise ValueError('Provide three nonnegative task weights with positive total')
    if (not 0 <= config.PITCH_CORRUPTION_SEMITONES <= 7
            or int(config.PITCH_CORRUPTION_SEMITONES) != config.PITCH_CORRUPTION_SEMITONES):
        raise ValueError('Pitch corruption must be between 0 and 7 semitones')
    if not 0 <= config.TIMING_CORRUPTION_MS <= 50 or config.TIMING_CORRUPTION_MS % 10:
        raise ValueError('Timing corruption must be a multiple of 10 ms between 0 and 50 ms')
    if not 0 <= config.STEM_TIMING_JITTER_MS <= 50 or config.STEM_TIMING_JITTER_MS % 10:
        raise ValueError('Stem timing jitter must be a multiple of 10 ms between 0 and 50 ms')
    if not 0 <= config.STEM_PITCH_SHIFT_SEMITONES <= 7 or int(config.STEM_PITCH_SHIFT_SEMITONES) != config.STEM_PITCH_SHIFT_SEMITONES:
        raise ValueError('Stem pitch shifts must be between 0 and 7 semitones')
    if not 0 <= config.STEM_PITCH_SHIFT_MIN_PROB <= config.STEM_PITCH_SHIFT_PROB <= .05:
        raise ValueError('Stem pitch corruption rates must lie between 0% and 5%')
    if not .9 <= config.HARMONIZE_MIN_MASK_RATIO <= 1:
        raise ValueError('Harmonization timing masking must be between 90% and 100%')
    if config.GENERIC_MERGE_PROB > .5:
        raise ValueError('Generic merge probability cannot exceed 50% of additional edits')
    if not 1 <= config.MAX_CORRUPTIONS <= 4 or int(config.MAX_CORRUPTIONS) != config.MAX_CORRUPTIONS:
        raise ValueError('Provide between one and four distinct additional refinement edits')
    if config.JAZZ_BLUES_REPEATS < 1 or int(config.JAZZ_BLUES_REPEATS) != config.JAZZ_BLUES_REPEATS:
        raise ValueError('JAZZ_BLUES_REPEATS must be a positive integer')
