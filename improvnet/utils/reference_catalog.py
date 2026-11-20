"""Small, reproducible training-reference catalogues and descriptive rhythm summaries."""
from pathlib import Path
import json
import math
import numpy as np
from improvnet.utils.melody import Note, attack_groups, ordered
from improvnet.utils.melody_groove import infer_clock, groove_statistics
from improvnet.utils.melody_reference import trio_instruments

DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / 'inference' / 'reference_catalog.json'
CATALOG_FORMAT = 'melody_reference_catalog_v3'
TEMPO_DESCRIPTORS = ('slow','moderate','fast')
DENSITY_DESCRIPTORS = ('sparse','balanced','dense')
DESCRIPTOR_POLICY = dict(
    tempo_bpm=dict(slow='below 90',moderate='90 to below 140',fast='140 or above'),
    piano_attacks_per_second=dict(sparse='below 2',balanced='2 to below 4',dense='4 or above'),
    scope='Whole selected piano/drums/double-bass view; activity counts onset groups, not chord voices.',
    caveat='Estimated pulse can have half/double-tempo ambiguity. Descriptors select the reference; they do not enforce output tempo.')


def parse_reference_description(text):
    """A small explicit descriptor vocabulary, with optional tempo+density pairs."""
    words=str(text).lower().replace('-', ' ').replace(',', ' ').split()
    aliases={'medium':'moderate','busy':'dense'}
    words=[aliases.get(word,word) for word in words]
    allowed={'any',*TEMPO_DESCRIPTORS,*DENSITY_DESCRIPTORS}
    if not words or any(word not in allowed for word in words):
        raise ValueError('Use any, slow, moderate, fast, sparse, balanced or dense; combine tempo and activity, e.g. "slow sparse"')
    unique=set(words)
    if ('any' in unique and len(unique)>1 or len(unique & set(TEMPO_DESCRIPTORS))>1
            or len(unique & set(DENSITY_DESCRIPTORS))>1):
        raise ValueError('Choose at most one tempo and one activity descriptor; use any alone')
    return ' '.join(word for word in (*TEMPO_DESCRIPTORS,*DENSITY_DESCRIPTORS,'any') if word in unique)


def rhythm_descriptors(rhythm):
    bpm=float(rhythm['estimated_bpm'])
    activity=float(rhythm['piano_attacks_per_second'])
    if not math.isfinite(bpm) or bpm<=0 or not math.isfinite(activity) or activity<0:
        raise ValueError('Reference tempo and activity must be finite and valid')
    return ['slow' if bpm<90 else 'moderate' if bpm<140 else 'fast',
            'sparse' if activity<2 else 'balanced' if activity<4 else 'dense']


def record_notes(record):
    notes = [Note('Acoustic Piano' if n[0] == 'piano' else n[0], *n[1:]) for n in record['notes']]
    return ordered(notes + [Note(*n) for n in record.get('drums', [])])


def load_catalog(path=DEFAULT_CATALOG):
    data = json.loads(Path(path).read_text())
    if data.get('format') != CATALOG_FORMAT:
        raise ValueError('Unsupported reference catalogue format')
    entries = data['examples']
    if len({e['id'] for e in entries}) != len(entries):
        raise ValueError('Duplicate reference catalogue IDs')
    if any(e['split'] != 'train' or e['genre'] not in ('jazz', 'blues') for e in entries):
        raise ValueError('Reference catalogue must contain training-fold jazz/blues only')
    for entry in entries:
        validation=entry.get('trio_validation',{})
        pairs=validation.get('stage_candidate_pairs',{})
        if (not trio_instruments(entry['instruments']) or validation.get('windows')!=2
                or not all(isinstance(pairs.get(stage),int) and pairs[stage]>0 for stage in ('harmonize','drums','bass'))):
            raise ValueError('Reference catalogue must contain validated exact piano/drums/double-bass trios with no extra instruments')
        if entry.get('rhythm_scope')!='whole_trio_view' or entry.get('descriptors')!=rhythm_descriptors(entry['rhythm']):
            raise ValueError('Reference descriptors must describe the whole trio view')
    return data


def load_entry_notes(entry):
    """Read the exact prepared record used when computing the catalogue summary."""
    locator = entry['record']
    with Path(locator['jsonl']).open('rb') as stream:
        stream.seek(locator['offset'])
        record = json.loads(stream.readline())
    if record['path'] != entry['path']:
        raise ValueError('Reference record has moved; rebuild the catalogue')
    return record_notes(record)


def rhythm_summary(notes):
    """Estimates for browsing only; never fed into the trained model as controls."""
    notes = ordered(notes)
    pitched = [n for n in notes if n.instrument != 'drum']
    piano = [n for n in pitched if 'Piano' in n.instrument] or pitched
    clock = infer_clock(notes)
    groove = groove_statistics(piano, clock)
    attacks = np.asarray([g[0].onset for g in attack_groups(piano, 50)], dtype=float)
    intervals = np.diff(attacks)
    duration = max((n.onset + n.duration for n in notes), default=0) - min((n.onset for n in notes), default=0)
    seconds = max(.01, duration / 1000)
    phases = np.asarray([clock.phase(t) % 1 for t in attacks])
    offbeat_count = int(((phases >= .4) & (phases <= .8)).sum())
    fraction = groove.get('swing_fraction', .5)
    feel = ('uncertain' if clock.confidence < .25 or offbeat_count < 3 else
            'unequal_eighth_candidate' if fraction >= .60 else 'near_even_eighth_candidate')
    bpm = 60000 / clock.beat_ms
    tempo = 'slow' if bpm < 90 else 'moderate' if bpm < 140 else 'fast'
    return dict(estimated_bpm=round(bpm, 2), pulse_confidence=round(clock.confidence, 4),
        estimated_offbeat_fraction=round(fraction, 4), offbeat_attack_count=offbeat_count,
        timing_spread_ms=round(groove.get('timing_spread_ms', 0), 2),
        piano_attacks_per_second=round(len(attacks) / seconds, 3),
        piano_ioi_median_ms=round(float(np.median(intervals)), 2) if len(intervals) else None,
        piano_ioi_variation=round(float(intervals.std() / intervals.mean()), 4) if len(intervals) and intervals.mean() else None,
        drum_hits_per_second=round(sum(n.instrument == 'drum' for n in notes) / seconds, 3),
        feel_estimate=feel, description=f'{tempo} estimated pulse; {feel.replace("_", " ")}',
        caveat='Heuristic estimates: half/double tempo, meter, swing and expressive timing can be ambiguous.')
