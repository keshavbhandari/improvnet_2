"""Pitch-free performance summaries and probabilistically visible style controls."""
import json
import math
from collections import Counter
from pathlib import Path

import torch

from improvnet.utils.melody import attack_groups, melody_voice, skyline


STYLE_FEATURES = (
    'accompaniment_attacks_per_second', 'accompaniment_voicing_density',
    'ioi_mean_ms', 'ioi_std_ms', 'ioi_q25_ms', 'ioi_q75_ms', 'ioi_lag1_correlation',
)
STYLE_DEFINITIONS = {
    STYLE_FEATURES[0]: 'Accompaniment attack groups per second; 50 ms anchored grouping.',
    STYLE_FEATURES[1]: 'Mean distinct sounding accompaniment pitch/instrument pairs at group attacks, including held notes.',
    STYLE_FEATURES[2]: 'Mean interval in milliseconds between pitched attack groups.',
    STYLE_FEATURES[3]: 'Population standard deviation of pitched attack intervals in milliseconds.',
    STYLE_FEATURES[4]: '25th percentile of pitched attack intervals in milliseconds.',
    STYLE_FEATURES[5]: '75th percentile of pitched attack intervals in milliseconds.',
    STYLE_FEATURES[6]: 'Lag-one correlation of consecutive attack intervals; negative values describe alternation, not a guaranteed swing ratio.',
}


def _quantile(values, q):
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] * (hi - pos) + values[hi] * (pos - lo) if lo != hi else values[lo]


def performance_statistics(notes, melody=None, duration_ms=None, tolerance_ms=50, accompaniment_context=()):
    """Summarize a clean target window/reference without storing its pitches."""
    pitched = [n for n in notes if n.instrument != 'drum']
    if not pitched:
        return {}
    if melody is None:
        melody = melody_voice(skyline(pitched, tolerance_ms)[0])
    selected = Counter(melody)
    accompaniment = []
    for n in pitched:
        if selected[n]:
            selected[n] -= 1
        else:
            accompaniment.append(n)
    stats = {}
    if accompaniment:
        groups = attack_groups(accompaniment, tolerance_ms)
        duration = duration_ms if duration_ms is not None else max(n.onset + n.duration for n in pitched) - min(n.onset for n in pitched)
        stats[STYLE_FEATURES[0]] = len(groups) * 1000 / max(10, duration)
        # Sweep actual sounding notes at each group's final attack. A held
        # accompaniment note contributes even if it was struck in an earlier group.
        ordered_notes = sorted(accompaniment + [n for n in accompaniment_context if n.instrument != 'drum'],
                               key=lambda n: n.onset)
        active, cursor, densities = {}, 0, []
        for group in groups:
            t = max(n.onset for n in group)
            active = {key: end for key, end in active.items() if end > t}
            while cursor < len(ordered_notes) and ordered_notes[cursor].onset <= t:
                n = ordered_notes[cursor]
                if n.onset + n.duration > t:
                    active[(n.instrument, n.pitch)] = n.onset + n.duration
                cursor += 1
            densities.append(len(active))
        stats[STYLE_FEATURES[1]] = sum(densities) / len(densities)
    attacks = [g[0].onset for g in attack_groups(pitched, tolerance_ms)]
    intervals = [b - a for a, b in zip(attacks, attacks[1:]) if b > a]
    if intervals:
        mean = sum(intervals) / len(intervals)
        variance = sum((x - mean) ** 2 for x in intervals) / len(intervals)
        ordered_intervals = sorted(intervals)
        stats.update(dict(zip(STYLE_FEATURES[2:6], (mean, math.sqrt(variance),
                          _quantile(ordered_intervals, .25), _quantile(ordered_intervals, .75)))))
        if len(intervals) >= 3:
            a, b = intervals[:-1], intervals[1:]
            ma, mb = sum(a) / len(a), sum(b) / len(b)
            denom = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((x - mb) ** 2 for x in b))
            stats[STYLE_FEATURES[6]] = max(-1., min(1., sum((x - ma) * (y - mb) for x, y in zip(a, b)) / denom)) if denom > 0 else 0.
    return stats


def style_tensors(statistics=None, task=None, visible=True):
    """Log-scaled controls plus explicit per-feature presence bits."""
    values, mask = torch.zeros(len(STYLE_FEATURES)), torch.zeros(len(STYLE_FEATURES))
    if not visible or not statistics:
        return values, mask
    indices = range(2) if task == 'harmonize' else range(2, 7) if task == 'rhythm' else ()
    for i in indices:
        name = STYLE_FEATURES[i]
        if name not in statistics:
            continue
        value = float(statistics[name])
        if not math.isfinite(value) or (i != 6 and value < 0) or (i == 6 and not -1 <= value <= 1):
            raise ValueError(f'Invalid style feature {name}: {value}')
        values[i] = value if i == 6 else math.log1p(value / (1000 if i >= 2 else 1))
        mask[i] = 1
    return values, mask


def choose_reference(path, genre, rng, required_tasks=('rhythm', 'harmonize')):
    payload = json.loads(Path(path).read_text())
    if payload.get('version') != 1 or payload.get('feature_names') != list(STYLE_FEATURES):
        raise ValueError('Style reference schema differs from the model')
    examples = [x for x in payload['examples'] if x['genre'] == genre and x['split'] == 'train'
                and all(any(style_tensors(x['statistics'], task)[1]) for task in required_tasks)]
    if not examples:
        raise ValueError(f'No suitable {genre} training reference in {path}; provide --style-reference')
    return rng.choice(examples)
