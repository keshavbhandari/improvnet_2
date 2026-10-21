"""Instrument preferences, kit-only drum normalization and stem selection."""
from dataclasses import replace

# A drum-set subset of the General MIDI percussion map. Auxiliary percussion
# is deliberately excluded; these are sound IDs, not transposable pitches.
DRUM_KIT_SOUNDS = {
    35: 'Acoustic Bass Drum', 36: 'Bass Drum 1',
    37: 'Side Stick', 38: 'Acoustic Snare', 40: 'Electric Snare',
    41: 'Low Floor Tom', 43: 'High Floor Tom', 45: 'Low Tom',
    47: 'Low-Mid Tom', 48: 'Hi-Mid Tom', 50: 'High Tom',
    42: 'Closed Hi-Hat', 44: 'Pedal Hi-Hat', 46: 'Open Hi-Hat',
    49: 'Crash Cymbal 1', 51: 'Ride Cymbal 1', 52: 'Chinese Cymbal',
    53: 'Ride Bell', 55: 'Splash Cymbal', 57: 'Crash Cymbal 2', 59: 'Ride Cymbal 2',
}
DRUM_KIT_PITCHES = frozenset(DRUM_KIT_SOUNDS)
DRUM_POLICY = 'gm_drum_set_only_v1'
# Extended electronic hi-hat edge articulations, documented by Groove MIDI.
# https://magenta.withgoogle.com/datasets/groove#drum-mapping
DRUM_KIT_ALIASES = {22: 42, 26: 46}


def kit_only_notes(notes):
    """Normalize hi-hat edge hits; retain only kit drums and pitched instruments.

    Coincident aliases/bow hits become one attack with the stronger velocity.
    Pitched instruments are never filtered, remapped or deduplicated here.
    """
    result, attacks = [], {}
    for n in notes:
        if n.instrument != 'drum':
            result.append(n)
            continue
        pitch = DRUM_KIT_ALIASES.get(n.pitch, n.pitch)
        if pitch not in DRUM_KIT_PITCHES:
            continue
        if pitch != n.pitch:
            n = replace(n, pitch=pitch)
        key = (pitch, n.onset)
        if key not in attacks:
            attacks[key] = len(result)
            result.append(n)
        elif n.velocity > result[attacks[key]].velocity:
            result[attacks[key]] = n
    return result


JAZZ_INSTRUMENTS = frozenset(('Tenor Sax','Soprano/Alto Sax','Baritone Sax',
    'Electric Bass','Acoustic Bass','Trombone','Trumpet','Clarinet',
    'Chromatic Percussion','Clean Electric Guitar','Electric Piano'))


def instrument_weight(instrument, jazz_weight=3.):
    return jazz_weight if instrument in JAZZ_INSTRUMENTS else 1.


def choose_missing_stem(instruments,rng,drum_probability=.85,jazz_weight=3.):
    instruments=sorted(instruments)
    if 'drum' in instruments and rng.random()<drum_probability:
        return 'drum'
    pitched=[i for i in instruments if i!='drum']
    candidates=pitched if len(pitched)>1 else instruments
    return rng.choices(candidates,[instrument_weight(i,jazz_weight) for i in candidates])[0]
