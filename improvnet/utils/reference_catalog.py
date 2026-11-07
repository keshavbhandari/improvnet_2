"""Small, reproducible training-reference catalogues and descriptive rhythm summaries."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
from improvnet.utils.melody import Note, attack_groups, ordered
from improvnet.utils.melody_groove import infer_clock, groove_statistics
from improvnet.utils.melody_stems import kit_only_notes

DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / 'inference' / 'reference_catalog.json'
CATALOG_FORMAT = 'melody_reference_catalog_v1'


def record_notes(record):
    notes = [Note('Acoustic Piano' if n[0] == 'piano' else n[0], *n[1:]) for n in record['notes']]
    return ordered(kit_only_notes(notes + [Note(*n) for n in record.get('drums', [])]))


def load_catalog(path=DEFAULT_CATALOG):
    data = json.loads(Path(path).read_text())
    if data.get('format') != CATALOG_FORMAT:
        raise ValueError('Unsupported reference catalogue format')
    entries = data['examples']
    if len({e['id'] for e in entries}) != len(entries):
        raise ValueError('Duplicate reference catalogue IDs')
    if any(e['split'] != 'train' or e['genre'] not in ('jazz', 'blues') for e in entries):
        raise ValueError('Reference catalogue must contain training-fold jazz/blues only')
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


def cosine_clusters(vectors, count=4, seed=42):
    """Deterministic spherical k-means; cluster numbers have no musical semantics."""
    if count < 1:
        raise ValueError('Cluster count must be positive')
    x = np.asarray(vectors, dtype=float)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all() or np.any(np.linalg.norm(x, axis=1) == 0):
        raise ValueError('Expected finite, nonzero embedding vectors')
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    indices = [int(np.random.default_rng(seed).integers(len(x)))]
    for _ in range(1, min(count, len(x))):
        distance = 1 - (x @ x[indices].T).max(axis=1)
        if distance.max() < 1e-10:
            break
        indices.append(int(distance.argmax()))
    centers = x[indices].copy()
    for _ in range(50):
        labels = (x @ centers.T).argmax(axis=1)
        updated = centers.copy()
        for i in range(len(centers)):
            members = x[labels == i]
            if len(members):
                mean = members.mean(axis=0)
                if np.linalg.norm(mean) > 1e-12:
                    updated[i] = mean / np.linalg.norm(mean)
        if np.allclose(updated, centers, atol=1e-7):
            break
        centers = updated
    labels = (x @ centers.T).argmax(axis=1)
    return labels.tolist()


def bucket_summaries(entries, key):
    summaries = []
    for bucket in sorted({e[key] for e in entries}):
        rows = [e for e in entries if e[key] == bucket]
        summaries.append(dict(id=bucket, members=[e['id'] for e in rows],
            genres=dict(Counter(e['genre'] for e in rows)), sources=dict(Counter(e['source'] for e in rows)),
            estimated_bpm_range=[min(e['rhythm']['estimated_bpm'] for e in rows), max(e['rhythm']['estimated_bpm'] for e in rows)],
            note='Embedding similarity group, not an annotated groove/genre class.'))
    return summaries
