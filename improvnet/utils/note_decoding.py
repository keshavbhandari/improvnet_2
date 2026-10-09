"""Onset-gated, confidence-aware note assembly on a continuous fine-time grid.

This is inference postprocessing, not a change to the lossless MIDI tokenizer.
It uses predictions only; reference notes never enter the decoding decision.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class NoteDecoding:
    mode: str = "legacy"
    continuation_threshold: float = .5
    max_gap_ms: int = 20

    def __post_init__(self):
        if self.mode not in ("legacy", "held"):
            raise ValueError("Note decoding must be 'legacy' or 'held'")
        if not math.isfinite(self.continuation_threshold) or not 0 < self.continuation_threshold < 1:
            raise ValueError("Continuation threshold must be strictly between zero and one")
        if isinstance(self.max_gap_ms, bool) or not isinstance(self.max_gap_ms, int) or self.max_gap_ms < 0:
            raise ValueError("Maximum note gap must be a nonnegative integer in milliseconds")


def decode_held_notes(activity, onset, velocity, valid_mask, *, activity_threshold,
                      onset_threshold, continuation_threshold=.5, max_gap_ms=20,
                      fine_ms=10, allow_initial_carry=True):
    """Return canonical velocity/onset arrays, carrying state across ALL bins.

    New strikes require both onset and start-activity evidence. A held note uses
    the lower continuation threshold. Short internal dips may be bridged, but
    never a gap ending in a new onset: that represents an articulated release.
    Unbridged releases remain at the FIRST inactive frame, rather than being
    delayed by the lookahead allowance. Trailing gaps are never filled.

    Only frame zero can infer a carry without an onset. Every later bin uses
    existing state; an activity-only return after a confirmed release is silent.
    Attack velocity is retained until the next accepted strike.
    """
    cfg = NoteDecoding("held", continuation_threshold, max_gap_ms)
    if (not math.isfinite(activity_threshold) or not 0 < activity_threshold < 1
            or not math.isfinite(onset_threshold) or not 0 < onset_threshold < 1
            or cfg.continuation_threshold > activity_threshold):
        raise ValueError("Invalid start/onset thresholds or continuation exceeds start threshold")
    if isinstance(fine_ms, bool) or not isinstance(fine_ms, int) or fine_ms < 1:
        raise ValueError("Fine frame duration must be a positive integer")
    activity, onset, velocity = (np.asarray(x) for x in (activity, onset, velocity))
    valid_mask = np.asarray(valid_mask)
    if (activity.ndim != 3 or activity.shape[1] != 128 or onset.shape != activity.shape
            or velocity.shape != activity.shape or valid_mask.dtype != np.bool_
            or valid_mask.shape != (activity.shape[0], activity.shape[2])):
        raise ValueError("Expected [bins,128,frames] probabilities and a boolean [bins,frames] mask")
    valid = valid_mask.ravel()
    if np.any(np.diff(valid.astype(np.int8)) > 0):
        raise ValueError("Padding must be a trailing suffix")
    n = int(valid.sum())
    shape = activity.shape
    result = {k: np.zeros(shape, dtype=np.uint8) for k in ("velocity", "onset")}
    if not n:
        return result
    a, o, v = (x.transpose(1, 0, 2).reshape(128, -1)[:, :n] for x in (activity, onset, velocity))
    if any(not np.isfinite(x).all() or np.any((x < 0) | (x > 1)) for x in (a, o, v)):
        raise ValueError("Valid-frame probabilities/normalized velocities must be finite and within [0,1]")
    hits = (o >= onset_threshold) & (a >= activity_threshold)
    carry = (a[:, 0] >= activity_threshold) if allow_initial_carry else np.zeros(128, dtype=bool)
    out_v = np.zeros((128, len(valid)), dtype=np.uint8)
    out_o = np.zeros_like(out_v)
    gap_frames = max_gap_ms // fine_ms
    for pitch in np.flatnonzero(hits.any(axis=1) | carry):
        active = a[pitch] >= continuation_threshold
        edges = np.diff(np.r_[False, active, False].astype(np.int8))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        if not len(starts):
            continue
        # Merge only short INTERNAL uncertainty. A new onset at the right edge
        # preserves the real gap and still retriggers even at identical velocity.
        merged = []
        begin, finish = int(starts[0]), int(ends[0])
        for left, right in zip(starts[1:], ends[1:]):
            if left - finish <= gap_frames and not hits[pitch, left]:
                finish = int(right)
            else:
                merged.append((begin, finish)); begin, finish = int(left), int(right)
        merged.append((begin, finish))
        attacks = np.flatnonzero(hits[pitch])
        for begin, finish in merged:
            first, last = np.searchsorted(attacks, (begin, finish))
            strikes = attacks[first:last].tolist()
            if begin == 0 and carry[pitch] and (not strikes or strikes[0] != 0):
                strikes.insert(0, 0)
            for index, start in enumerate(strikes):
                end = strikes[index + 1] if index + 1 < len(strikes) else finish
                out_v[pitch, start:end] = np.uint8(np.clip(np.rint(v[pitch, start] * 127), 1, 127))
                out_o[pitch, start] = hits[pitch, start]
    for key, array in (("velocity", out_v), ("onset", out_o)):
        result[key] = array.reshape(128, shape[0], shape[2]).transpose(1, 0, 2).copy()
    return result


def confidence_arrays(reconstruction):
    """Capture full-precision confidences before hard gating/velocity masking."""
    return dict(activity=reconstruction.activity_logits.detach().float().sigmoid().cpu().numpy(),
                onset=reconstruction.onset_logits.detach().float().sigmoid().cpu().numpy(),
                velocity=reconstruction.velocity.detach().float().clamp(0, 1).cpu().numpy())


def apply_note_decoding(piece, indices, confidence, options, *, activity_threshold, onset_threshold):
    """Process each contiguous selected run; never borrow unselected reference state."""
    options = options if isinstance(options, NoteDecoding) else NoteDecoding(**(options or {}))
    if options.mode == "legacy":
        return piece
    ids = np.asarray(indices)
    if (ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(np.diff(ids) <= 0)):
        raise ValueError("Held-note decoding requires sorted, unique selected bins")
    breaks = np.r_[0, np.flatnonzero(np.diff(ids) != 1) + 1, len(ids)]
    for left, right in zip(breaks[:-1], breaks[1:]):
        selected = ids[left:right]
        arrays = decode_held_notes(*(confidence[k][left:right] for k in ("activity", "onset", "velocity")),
            piece.valid_mask[selected], activity_threshold=activity_threshold, onset_threshold=onset_threshold,
            continuation_threshold=options.continuation_threshold, max_gap_ms=options.max_gap_ms,
            fine_ms=piece.config.fine_ms)
        for key, value in arrays.items():
            getattr(piece, key)[selected] = value
    return piece
