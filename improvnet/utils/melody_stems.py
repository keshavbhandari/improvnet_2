"""Percussion palette labels, inference kit constraints and stem selection."""

# The inference kit palette, not a training-data filter. Full percussion
# targets retain every sound ID; these IDs are never transposed as pitches.
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
DRUM_POLICY = 'intact_targets_conditioned_palette_v1'
PERCUSSION_MODES = {'none': 0, 'kit': 1, 'full': 2}
# Extended electronic hi-hat edge articulations, documented by Groove MIDI.
# https://magenta.withgoogle.com/datasets/groove#drum-mapping
# Recognized as kit for labelling; original IDs are preserved in training.
DRUM_KIT_ALIASES = {22: 42, 26: 46}


def percussion_mode(notes):
    """Label authentic target content without removing or replacing any attacks."""
    pitches = [n.pitch for n in notes if n.instrument == 'drum']
    if not pitches:
        return 'none'
    return 'kit' if all(p in DRUM_KIT_PITCHES or p in DRUM_KIT_ALIASES for p in pitches) else 'full'


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
