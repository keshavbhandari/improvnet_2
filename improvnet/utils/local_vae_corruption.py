"""Conservative note-event corruption on the shared clock, before bin cropping.

Only encoder inputs change. Clean targets, pedal, padding and accompaniment stay
untouched. This is denoising supervision, not musical transposition augmentation.
"""
import math

import numpy as np
import torch

from improvnet.tokenizer.hierarchical import roll_notes


def target_note_rows(piece):
    """Normalized key-down events: pitch, velocity, start/end frame, source order.

    Extract once per cached piece, scanning across coarse-bin boundaries. These
    private training rows are not added to the tokenizer's stored NPZ format.
    """
    rows = roll_notes(piece.velocity, piece.onset, piece.n_frames,
                      piece.metadata["target_channel"], piece.metadata["target_track"],
                      piece.metadata["target_program"], strict=True)
    result = np.asarray([[n[3], n[4], n[5], n[6], n[7]] for n in rows], dtype=np.int64).reshape(-1, 5)
    return result[np.lexsort((result[:, 4], result[:, 2]))]


def render_bin(rows, index, frames_per_bin, valid_frames):
    """Crop complete events; attacks before the bin remain carries, not onsets."""
    result = np.zeros((2, 128, frames_per_bin), dtype=np.float32)
    start, stop = index * frames_per_bin, index * frames_per_bin + valid_frames
    for pitch, velocity, onset, end, _ in rows:
        left, right = max(int(onset), start), min(int(end), stop)
        if left < right:
            result[0, pitch, left - start:right - start] = velocity / 127
            if start <= onset < stop:
                result[1, pitch, onset - start] = 1
    return torch.from_numpy(result)


def semantic_probability(config, step=0, *, training=True):
    """Rate follows completed optimizer updates; validation always uses the maximum."""
    maximum = config.SEMANTIC_NOTE_PROB
    warmup = getattr(config, "SEMANTIC_CORRUPTION_WARMUP_STEPS", 0)
    if not training or not warmup:
        return maximum
    return maximum * min(1., max(0., step / warmup))


def corrupt_roll_bin(piece, index, config, rng, *, notes=None, probability=None):
    """Bernoulli-select notes; never force corruption in a sparse/silent bin.

    Each eligible note has SEMANTIC_NOTE_PROB chance of one proposed operation,
    not a guaranteed percentage per bin. Unsafe proposals are skipped rather
    than clipped/wrapped/retried. Original and added events never overlap another
    same-pitch event. Drums use timing/deletion only, preserving percussion IDs.
    """
    if not 0 <= index < len(piece.velocity):
        raise IndexError(index)
    notes = target_note_rows(piece) if notes is None else notes
    frames = piece.config.frames_per_bin
    start = index * frames
    stop = start + int(piece.valid_mask[index].sum())
    max_shift = int(math.floor(config.SEMANTIC_TIME_SHIFT_MS / piece.config.fine_ms))
    # Include neighbouring attacks that can enter this bin, plus complete carries.
    selected = (notes[:, 2] < stop + max_shift) & (notes[:, 3] > max(0, start - max_shift))
    nearby, outside = notes[selected], notes[~selected]
    probability = config.SEMANTIC_NOTE_PROB if probability is None else probability
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Corruption probability must be in [0,1]")
    cap = getattr(config, "SEMANTIC_MAX_NOTE_FRACTION", None)
    budget = None
    order = list(range(len(nearby)))
    if cap is not None:
        if not math.isfinite(cap) or not 0 <= cap <= 1:
            raise ValueError("SEMANTIC_MAX_NOTE_FRACTION must be in [0,1]")
        visible = int(((nearby[:, 2] < stop) & (nearby[:, 3] > start)).sum())
        budget = math.floor(cap * visible)
        # Randomize before imposing the cap, avoiding chronological selection bias.
        rng.shuffle(order)
    rows = [row.copy() for row in nearby]
    attacks = int(((nearby[:, 2] >= start) & (nearby[:, 2] < stop)).sum())
    operations = tuple(config.SEMANTIC_OPERATION_WEIGHTS)
    weights = tuple(config.SEMANTIC_OPERATION_WEIGHTS.values())
    is_drum = bool(piece.metadata.get("target_is_drum", piece.metadata["target_channel"] == 9))
    deleted = changed = 0

    def overlaps(candidate, except_index=None):
        if np.any((outside[:, 0] == candidate[0]) & (candidate[2] < outside[:, 3]) & (outside[:, 2] < candidate[3])):
            return True
        return any(other is not None and j != except_index and
                   other[0] == candidate[0] and candidate[2] < other[3] and other[2] < candidate[3]
                   for j, other in enumerate(rows))

    for i in order:
        if budget is not None and changed >= budget:
            break
        original = nearby[i]
        if rng.random() >= probability:
            continue
        operation = rng.choices(operations, weights=weights)[0]
        if is_drum and operation in ("pitch", "addition"):
            continue
        candidate = original.copy()
        if operation == "timing":
            if not max_shift:
                continue
            amount = rng.choice(tuple(range(-max_shift, 0)) + tuple(range(1, max_shift + 1)))
            candidate[2:4] += amount  # Preserve complete note duration.
            if candidate[2] < 0 or candidate[3] > piece.n_frames:
                continue
            # Do not remove the only local evidence of a note by shifting it out.
            if original[2] < stop and original[3] > start and not (candidate[2] < stop and candidate[3] > start):
                continue
        elif operation in ("pitch", "deletion"):
            # Carries and isolated notes have insufficient evidence for recovery.
            if not start <= original[2] < original[3] <= stop:
                continue
            if operation == "pitch":
                if attacks < 2 or not config.SEMANTIC_PITCH_SHIFT_MAX:
                    continue
                amount = rng.choice(tuple(range(-config.SEMANTIC_PITCH_SHIFT_MAX, 0)) +
                                    tuple(range(1, config.SEMANTIC_PITCH_SHIFT_MAX + 1)))
                candidate[0] += amount
            else:
                if attacks < config.SEMANTIC_MIN_NOTES_FOR_DELETION or deleted >= max(1, attacks // 10):
                    continue
                rows[i] = None
                deleted += 1
                changed += 1
                continue
        elif operation == "addition":
            # The original event survives, so even isolated-note bins retain it.
            candidate[0] += rng.choice((-1, 1))
        else:
            raise ValueError(f"Unknown semantic operation: {operation}")
        if not 0 <= candidate[0] <= 127 or overlaps(candidate, None if operation == "addition" else i):
            continue
        if operation == "addition":
            rows.append(candidate)
        else:
            rows[i] = candidate
        changed += 1
    return render_bin([row for row in rows if row is not None], index, frames, stop - start)


def latent_channel_std(posterior):
    """Detached per-channel clean-mean scale from the current minibatch.

    Fractionally valid columns contribute their valid weight. Padded columns do
    not contribute. There is no per-example normalization or learned noise scale.
    """
    mean = posterior.mean.detach().float()
    weights = posterior.weights.float().expand_as(mean)
    dims = (0, 2, 3)
    count = weights.sum(dims).clamp_min(1)
    average = (mean * weights).sum(dims) / count
    variance = ((mean - average[None, :, None, None]).square() * weights).sum(dims) / count
    return variance.clamp_min(0).sqrt()


def jitter_latent(latent, posterior, maximum, *, generator=None, fixed=False):
    """Add Gaussian jitter relative to clean channel std; keep padding zero.

    Training draws one amplitude uniformly in [0, maximum] per example. Fixed
    evaluation uses exactly maximum. Noise is additional to posterior sampling.
    """
    if not math.isfinite(maximum) or maximum < 0:
        raise ValueError("Latent jitter must be finite and nonnegative")
    shape = (len(latent), 1, 1, 1)
    amount = torch.full(shape, maximum, device=latent.device, dtype=torch.float32) if fixed else (
        torch.rand(shape, device=latent.device, generator=generator) * maximum)
    scale = latent_channel_std(posterior)[None, :, None, None]
    epsilon = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=generator)
    result = latent + amount * scale * epsilon
    return result.masked_fill(~(posterior.weights > 0), 0)
