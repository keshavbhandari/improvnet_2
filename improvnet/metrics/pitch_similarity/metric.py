"""Inspectable pitch identity, ordered motifs and aligned SSM structure.

No learned parameters, timing comparisons, model imports or MIDI repair. The
caller supplies the corresponding reference CANVAS, never the clean prompt.
"""

from dataclasses import dataclass, asdict
import math
from numbers import Integral

import numpy as np

from . import config as defaults


@dataclass(frozen=True)
class PitchSimilarityConfig:
    exact_pitch_weight: float = defaults.EXACT_PITCH_WEIGHT
    pitch_class_weight: float = defaults.PITCH_CLASS_WEIGHT
    smooth_chroma_weight: float = defaults.SMOOTH_CHROMA_WEIGHT
    smooth_chroma_sigma: float = defaults.SMOOTH_CHROMA_SIGMA
    match_baseline: float = defaults.MATCH_BASELINE
    insert_open: float = defaults.INSERT_OPEN
    insert_extend: float = defaults.INSERT_EXTEND
    delete_open: float = defaults.DELETE_OPEN
    delete_extend: float = defaults.DELETE_EXTEND
    min_match_similarity: float = defaults.MIN_MATCH_SIMILARITY
    retention_beta: float = defaults.RETENTION_BETA
    pattern_lengths: tuple[int, ...] = defaults.PATTERN_LENGTHS
    max_pattern_insertions: int = defaults.MAX_PATTERN_INSERTIONS
    identity_weight: float = defaults.IDENTITY_WEIGHT
    pattern_weight: float = defaults.PATTERN_WEIGHT
    structure_weight: float = defaults.STRUCTURE_WEIGHT
    allow_global_transposition: bool = defaults.ALLOW_GLOBAL_TRANSPOSITION
    max_transpose: int = defaults.MAX_TRANSPOSE
    max_notes: int = defaults.MAX_NOTES

    def __post_init__(self):
        for key in ("exact_pitch_weight", "pitch_class_weight", "smooth_chroma_weight", "match_baseline",
                    "insert_open", "insert_extend", "delete_open", "delete_extend", "identity_weight", "pattern_weight", "structure_weight"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        feature_weight = self.exact_pitch_weight + self.pitch_class_weight + self.smooth_chroma_weight
        total_weight = self.identity_weight + self.pattern_weight + self.structure_weight
        if not math.isfinite(feature_weight) or not math.isfinite(total_weight) or feature_weight <= 0 or self.identity_weight <= 0:
            raise ValueError("Need pitch features and a positive identity weight")
        if isinstance(self.min_match_similarity, bool) or not math.isfinite(self.min_match_similarity) or not 0 < self.min_match_similarity <= 1 or not 0 <= self.match_baseline < 1:
            raise ValueError("Invalid alignment similarity thresholds")
        for key in ("smooth_chroma_sigma", "retention_beta"):
            if isinstance(getattr(self, key), bool) or not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.retention_beta > 1000:
            raise ValueError("retention_beta must be <=1000 to avoid unstable weighting")
        for key in ("max_pattern_insertions", "max_transpose", "max_notes"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if self.max_notes < 1 or self.max_transpose > 127 or not isinstance(self.allow_global_transposition, bool):
            raise ValueError("Invalid note limit/transposition settings")
        if not self.pattern_lengths or any(not isinstance(k, int) or isinstance(k, bool) or k < 2 for k in self.pattern_lengths) or len(set(self.pattern_lengths)) != len(self.pattern_lengths):
            raise ValueError("Need distinct pattern lengths >=2")


def _sequence(values, name):
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    if hasattr(values, "ndim") and values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence, not text")
    try:
        return list(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence") from error


def _pitches(values, name):
    values = _sequence(values, name)
    if any(not isinstance(p, Integral) or isinstance(p, bool) or not 0 <= p <= 127 for p in values):
        raise ValueError(f"{name} needs integer MIDI pitches in [0,127]")
    return np.asarray(values, dtype=np.int64)


@dataclass(frozen=True)
class Alignment:
    objective: float
    pairs: tuple[tuple[int, int], ...]


def align_cross_similarity(cross, config):
    """Global affine-gap alignment with deterministic traceback; no note reuse.

    States: M=paired notes, D=deleted reference, I=inserted generated. A gap
    of length L costs open+(L-1)*extend. Every input position is consumed.
    """
    cross = np.asarray(cross, dtype=np.float64)
    if cross.ndim != 2 or not np.isfinite(cross).all() or np.any(cross < 0) or np.any(cross > 1):
        raise ValueError("Need a finite [reference,generated] similarity matrix in [0,1]")
    n, m = cross.shape
    if max(n, m) > config.max_notes:
        raise ValueError("Alignment exceeds max_notes; split into explicit windows")
    costs = np.full((3, n + 1, m + 1), -np.inf)
    previous = np.zeros((3, n + 1, m + 1), dtype=np.uint8)
    costs[0, 0, 0] = 0
    for i in range(1, n + 1):
        costs[1, i, 0] = -config.delete_open - (i - 1) * config.delete_extend
        previous[1, i, 0] = 0 if i == 1 else 1
    for j in range(1, m + 1):
        costs[2, 0, j] = -config.insert_open - (j - 1) * config.insert_extend
        previous[2, 0, j] = 0 if j == 1 else 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            values = costs[:, i - 1, j - 1]
            state = int(np.argmax(values))
            costs[0, i, j] = values[state] + cross[i - 1, j - 1] - config.match_baseline
            previous[0, i, j] = state
            values = (costs[0, i - 1, j] - config.delete_open,
                      costs[1, i - 1, j] - config.delete_extend,
                      costs[2, i - 1, j] - config.delete_open)
            state = int(np.argmax(values))
            costs[1, i, j], previous[1, i, j] = values[state], state
            values = (costs[0, i, j - 1] - config.insert_open,
                      costs[1, i, j - 1] - config.insert_open,
                      costs[2, i, j - 1] - config.insert_extend)
            state = int(np.argmax(values))
            costs[2, i, j], previous[2, i, j] = values[state], state
    i, j, state = n, m, int(np.argmax(costs[:, n, m]))
    objective, pairs = float(costs[state, n, m]), []
    while i or j:
        prev = int(previous[state, i, j])
        if state == 0:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif state == 1:
            i -= 1
        else:
            j -= 1
        state = prev
    return Alignment(objective, tuple(reversed(pairs)))


@dataclass(frozen=True)
class PitchSimilarityResult:
    score: float
    pitch_identity: float
    pattern_preservation: float | None
    structural_preservation: float | None
    reference_coverage: float
    generated_precision: float
    ssm_cosine: float | None
    structural_pair_coverage: float
    matched_notes: int
    reference_notes: int
    generated_notes: int
    generated_to_reference_transpose: int
    alignment: tuple[tuple[int, int], ...]
    pattern_by_length: dict[int, float]

    def to_dict(self):
        return asdict(self)


class PitchSimilarityMetric:
    def __init__(self, config=None, *, tokenizer=None):
        self.config = config or PitchSimilarityConfig()
        self.tokenizer = tokenizer  # Lazy: pure-pitch use does not load the tokenizer.
        distance = np.abs(np.arange(12)[:, None] - np.arange(12)[None, :])
        distance = np.minimum(distance, 12 - distance)
        smooth = np.exp(-.5 * (distance / self.config.smooth_chroma_sigma)**2)
        smooth /= np.linalg.norm(smooth, axis=1, keepdims=True)
        self.chroma_cosine = np.clip(smooth @ smooth.T, 0, 1)

    def cross_similarity(self, reference, generated, *, transpose=0):
        """Cosine of weighted exact-pitch, chroma and smooth-chroma features.

        Feature blocks use square-root weights; both norms are the same. A
        global transpose shifts generated pitches virtually (no edge clipping).
        """
        r, g = _pitches(reference, "reference"), _pitches(generated, "generated")
        if not isinstance(transpose, Integral) or isinstance(transpose, bool) or abs(transpose) > 127:
            raise ValueError("Invalid global transposition")
        g = g + transpose
        c = self.config
        norm = c.exact_pitch_weight + c.pitch_class_weight + c.smooth_chroma_weight
        cross = (c.exact_pitch_weight * (r[:, None] == g[None, :])
                 + c.pitch_class_weight * (r[:, None] % 12 == g[None, :] % 12)
                 + c.smooth_chroma_weight * self.chroma_cosine[r[:, None] % 12, g[None, :] % 12]) / norm
        return np.clip(cross, 0, 1)

    def _patterns(self, r, g, pairs, cross, transpose):
        mapping = dict(pairs)
        by_length = {}
        for length in self.config.pattern_lengths:
            if len(r) < length:
                continue
            scores = []
            for start in range(len(r) - length + 1):
                indices = list(range(start, start + length))
                if any(i not in mapping for i in indices):
                    scores.append(0.)
                    continue
                slots = [mapping[i] for i in indices]
                if slots[-1] - slots[0] + 1 - length > self.config.max_pattern_insertions:
                    scores.append(0.)
                    continue
                pitch_score = float(np.mean(cross[indices, slots]))
                # Compare intervals along the aligned backbone, not along added
                # passing notes. Modulo 12 permits octave redistributions.
                a = np.diff(r[indices]) % 12
                b = np.diff(g[slots] + transpose) % 12
                interval_score = float(np.mean(self.chroma_cosine[a, b]))
                scores.append(pitch_score * interval_score)
            by_length[length] = float(np.mean(scores))
        return (float(np.mean(list(by_length.values()))) if by_length else None), by_length

    def _structure(self, r, g, pairs, cross):
        n, k = len(r), len(pairs)
        if k < 2:
            return None, None, 0.
        ri, gi = np.asarray(pairs).T
        a, b = self.cross_similarity(r[ri], r[ri]), self.cross_similarity(g[gi], g[gi])
        upper = np.triu_indices(k, 1)  # Never include the trivially matching diagonal.
        confidence = cross[ri, gi]
        pair_weights = (confidence[:, None] * confidence[None, :])[upper]
        coverage = float(pair_weights.sum() / (n * (n - 1) / 2))
        a, b = a[upper] * np.sqrt(pair_weights), b[upper] * np.sqrt(pair_weights)
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-12 and nb < 1e-12:
            # No off-diagonal evidence: do not award perfect structural credit
            # to two identity matrices from unrelated all-distinct melodies.
            return None, None, coverage
        cosine = 0. if min(na, nb) < 1e-12 else float(np.clip(np.dot(a, b) / (na * nb), 0, 1))
        return cosine * coverage, cosine, coverage

    def _evaluate(self, r, g, transpose):
        cross = self.cross_similarity(r, g, transpose=transpose)
        alignment = align_cross_similarity(cross, self.config)
        pairs = tuple((i, j) for i, j in alignment.pairs if cross[i, j] >= self.config.min_match_similarity)
        mass = sum(float(cross[i, j]) for i, j in pairs)
        recall, precision = mass / len(r), mass / len(g)
        beta2 = self.config.retention_beta**2
        identity = (1 + beta2) * recall * precision / (beta2 * precision + recall) if mass else 0.
        pattern, by_length = self._patterns(r, g, pairs, cross, transpose)
        structure, cosine, coverage = self._structure(r, g, pairs, cross)
        weighted = [(identity, self.config.identity_weight), (pattern, self.config.pattern_weight), (structure, self.config.structure_weight)]
        total_weight = sum(weight for value, weight in weighted if value is not None)
        score = sum(value * weight for value, weight in weighted if value is not None) / total_weight
        return PitchSimilarityResult(float(np.clip(score, 0, 1)), identity, pattern, structure, recall, precision,
                                     cosine, coverage, len(pairs), len(r), len(g), int(transpose), pairs, by_length)

    def evaluate_pitches(self, reference, generated):
        r, g = _pitches(reference, "reference"), _pitches(generated, "generated")
        if not len(r):
            raise ValueError("Reference has no pitched notes; similarity is not applicable")
        if max(len(r), len(g)) > self.config.max_notes:
            raise ValueError("Too many notes; split the corresponding canvases into explicit windows")
        if not len(g):
            return PitchSimilarityResult(0., 0., 0. if len(r) >= min(self.config.pattern_lengths) else None,
                                         None, 0., 0., None, 0., 0, len(r), 0, 0, (), {})
        offsets = [0]
        if self.config.allow_global_transposition:
            offsets += [offset for step in range(1, self.config.max_transpose + 1) for offset in (-step, step)]
        # Stable ties prefer no transposition, then the smallest absolute shift.
        best = None
        for offset in offsets:
            result = self._evaluate(r, g, offset)
            if best is None or result.score > best.score + 1e-12:
                best = result
        return best

    def pitches_from_tokens(self, tokens, *, context=None, valid_mask=None):
        """Read flat tokens/IDs. Pitch-only scoring is NOT a syntax validator.

        Prefixes and timing/velocity fields never enter the score. An optional
        context resolves the active instrument for a canvas beginning mid-event;
        context pitches are NEVER scored. Explicit padding masks remove pads.
        """
        if self.tokenizer is None:
            from improvnet.tokenizer.absolute import AbsTokenizer
            self.tokenizer = AbsTokenizer()
        t = self.tokenizer

        def decode(value):
            if isinstance(value, Integral) and not isinstance(value, bool):
                if int(value) not in t.id_to_tok:
                    raise ValueError("Unknown pitch-metric token ID")
                value = t.id_to_tok[int(value)]
            elif isinstance(value, list):
                value = tuple(value)
            if isinstance(value, tuple):
                if any(isinstance(field, bool) for field in value):
                    raise ValueError("Boolean fields are not musical values")
                if value and value[0] in ("note", "onset", "dur") and any(not isinstance(field, Integral) for field in value[1:]):
                    raise ValueError("Musical fields must be integers")
            try:
                known = value in t.tok_to_id
            except TypeError:
                known = False
            if not known or value in (t.pad_tok, t.mask_tok, t.sep_tok, t.blank_tok, t.unk_tok):
                raise ValueError("Pitch scoring expects known finalized tokens; mask out padding explicitly")
            return value

        instrument = None
        for value in _sequence(context if context is not None else [], "context"):
            token = decode(value)
            if isinstance(token, tuple) and token[0] == "instrument":
                instrument = token[1]
        values = _sequence(tokens, "tokens")
        if valid_mask is not None:
            mask = _sequence(valid_mask, "valid_mask")
            if len(mask) != len(values) or any(value not in (True, False, 0, 1) for value in mask):
                raise ValueError("valid_mask must match token positions")
            values = [value for value, keep in zip(values, mask) if keep]
        pitches = []
        for value in values:
            token = decode(value)
            if isinstance(token, tuple) and token[0] == "instrument":
                instrument = token[1]
            elif isinstance(token, tuple) and token[0] == "note":
                percussion = instrument == "drum" or "Drum" in str(instrument) or "Percuss" in str(instrument)
                if not percussion:
                    pitches.append(int(token[1]))
        return pitches

    def evaluate_tokens(self, reference, generated, *, reference_context=None, generated_context=None,
                        reference_mask=None, generated_mask=None):
        return self.evaluate_pitches(self.pitches_from_tokens(reference, context=reference_context, valid_mask=reference_mask),
                                     self.pitches_from_tokens(generated, context=generated_context, valid_mask=generated_mask))
