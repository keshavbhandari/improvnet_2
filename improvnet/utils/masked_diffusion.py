"""Absorbing-mask training and typed, confidence-based iterative decoding."""
import math
import random

import torch
from torch.nn import functional as F

from improvnet.utils.masked_diffusion_data import expand_slots, NOTE_ATTRIBUTES


def corrupt(target, roles, valid, *, generator=None, levels=None, whole_note=None, mask_id=5, blank_id=6, pad_id=2):
    """Uniform masking rate with equally frequent token and whole-note views.

    Whole-note views give hidden real notes the same mask pattern as empty slots;
    otherwise four masks would itself become a shortcut for predicting blanks.
    Every eligible token still has marginal masking probability t.
    """
    b = len(target)
    if levels is None:
        levels = torch.rand(b, device=target.device, generator=generator).clamp_min(1e-4)
    if levels.shape != (b,) or not bool(((levels > 0) & (levels <= 1)).all()):
        raise ValueError("Mask levels must lie in (0,1]")
    # Structural tokens use independent draws at the same rate as attributes.
    # Padding never becomes a reconstruction target, even if mislabeled valid.
    eligible = valid & (target != pad_id)
    draws = torch.rand(target.shape, device=target.device, generator=generator)
    if whole_note is None:
        whole_note = torch.rand(b, device=target.device, generator=generator) < .5
    if whole_note.shape != (b,):
        raise ValueError("Whole-note mode must have one flag per example")
    # Vectorized group masking avoids a CPU/GPU synchronization for each note.
    rows, starts = (valid & (roles == 1)).nonzero(as_tuple=True)
    positions = starts[:, None] + torch.arange(NOTE_ATTRIBUTES, device=target.device)
    groups = target[rows[:, None], positions]
    blank = groups[:, 0] == blank_id
    if bool((blank[:, None] & (groups != blank_id)).any()):
        raise ValueError("Blank targets must occupy complete consecutive note groups")
    tied = blank | whole_note[rows]
    values = draws[rows[tied], starts[tied]].clone()
    draws[rows[tied, None], positions[tied]] = values[:, None]
    masked = (draws < levels[:, None]) & eligible
    return target.masked_fill(masked, mask_id), masked, levels


def masked_loss(logits, target, masked, valid, roles, levels, *, pad_id=2):
    """Masked-token CE / t, normalized by all eligible tokens, never only masks."""
    eligible = valid & (target != pad_id)
    denominator = eligible.sum(-1)
    if bool((denominator == 0).any()):
        raise ValueError("Every example needs at least one valid nonpadding target")
    rows, columns = (masked & eligible).nonzero(as_tuple=True)
    if not len(rows):
        return logits.sum() * 0
    losses = F.cross_entropy(logits[rows, columns].float(), target[rows, columns], reduction="none")
    return (losses / levels[rows] / denominator[rows]).sum() / len(target)


def batch_objective(model, batch, *, generator=None):
    noisy, masked, levels = corrupt(batch["target"], batch["roles"], batch["valid"], generator=generator)
    logits = model(noisy, batch["roles"], batch["valid"], batch["genre"], batch["multi_hot"],
                   batch["rhythm"], batch["elasticity"])
    loss = masked_loss(logits, batch["target"], masked, batch["valid"], batch["roles"], levels)
    with torch.no_grad():
        correct = ((logits.argmax(-1) == batch["target"]) & masked).sum()
    return loss, torch.stack((loss.detach() * len(noisy), loss.new_tensor(len(noisy)),
                             correct.float(), masked.sum().float()))


def prepare_canvas(units, processor, *, strategy, percentage, elasticity, seed):
    if strategy not in ("timing", "pitch-velocity", "notes") or not math.isfinite(percentage) or not 0 <= percentage <= 1:
        raise ValueError("Invalid strategy or mask percentage (must be in [0,1])")
    rng = random.Random(seed)
    tokens, roles, slots, added = expand_slots(units, elasticity, rng)
    original = [s for s in slots if s not in added]
    selected = rng.sample(original, math.ceil(percentage * len(original)))
    fields = {"timing": (2, 3), "pitch-velocity": (1,), "notes": (0, 1, 2, 3)}[strategy]
    editable = torch.zeros(len(tokens), dtype=torch.bool)
    for start in selected:
        editable[[start + field for field in fields]] = True
    for start in added:
        editable[start:start + NOTE_ATTRIBUTES] = True
    # Complete-note edits may delete at most the elasticity budget of source notes.
    deletion_candidates = set(rng.sample(selected, min(len(selected), math.floor(elasticity * len(original))))) if strategy == "notes" else set()
    allow_blank = set(added) | deletion_candidates
    target = processor.tokens_to_tensor(tokens)
    source = target.masked_fill(editable, processor.tokenizer.tok_to_id["<MASK>"])
    return dict(tokens=source, roles=torch.tensor(roles), editable=editable, slots=slots,
                added=added, allow_blank=allow_blank, original_notes=len(original))


def _draw(logits, allowed, temperature, top_k, generator):
    allowed = torch.as_tensor(allowed, device=logits.device)
    values = logits[allowed].float()
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("Nonfinite sampling logits")
    if temperature == 0:
        index = values.argmax()
        return int(allowed[index]), float(values.softmax(-1)[index])
    values = values / temperature
    if top_k:
        threshold = values.topk(min(top_k, len(values))).values[-1]
        values = values.masked_fill(values < threshold, -torch.inf)
    probabilities = values.softmax(-1)
    index = torch.multinomial(probabilities, 1, generator=generator)[0]
    return int(allowed[index]), float(probabilities[index])


@torch.inference_mode()
def denoise(model, canvas, processor, *, genre, rhythm, multi_hot, elasticity,
            steps=32, temperature=1., top_k=50, seed=42):
    if steps < 1 or not math.isfinite(temperature) or temperature < 0 or top_k < 0:
        raise ValueError("Invalid sampler settings")
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    tokens = canvas["tokens"].to(device).clone()
    roles = canvas["roles"].to(device)
    tokenizer = processor.tokenizer
    mask_id, blank_id = tokenizer.tok_to_id["<MASK>"], tokenizer.tok_to_id["<BLANK>"]
    allowed = {1: [tokenizer.tok_to_id[t] for t in tokenizer.instrument_tokens],
               2: [tokenizer.tok_to_id[t] for t in tokenizer.note_tokens if t[2] > 0],
               3: [tokenizer.tok_to_id[t] for t in tokenizer.onset_tokens],
               4: [tokenizer.tok_to_id[t] for t in tokenizer.dur_tokens if t[1] > 0]}
    # Preserve the source instrument palette, including in added slots. Besides
    # keeping the arrangement recognizable, this respects MIDI's channel limit.
    palette = {name for name, present in zip(processor.INSTRUMENT_CLASSES, multi_hot.tolist()) if present}
    if palette:
        allowed[1] = [tokenizer.tok_to_id[t] for t in tokenizer.instrument_tokens if t[1] in palette]
    drum_notes = [tokenizer.tok_to_id[t] for t in tokenizer.note_tokens if t[2] == tokenizer.config["drum_velocity"]]
    # Each note is a confidence unit. Whole-note masks commit all four attributes
    # together; partial edits commit only their requested fields.
    pending = [s for s in canvas["slots"] if bool((tokens[s:s + NOTE_ATTRIBUTES] == mask_id).any())]
    passes = 0
    for iteration in range(steps):
        if not pending:
            break
        context = torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
        with context:
            logits = model(tokens[None], roles[None], torch.ones_like(tokens[None], dtype=torch.bool),
                torch.tensor([genre], device=device), multi_hot[None].to(device), rhythm[None].to(device),
                torch.tensor([elasticity], device=device))[0]
        passes += 1
        candidates = []
        for start in pending:
            indices = (tokens[start:start + NOTE_ATTRIBUTES] == mask_id).nonzero().flatten().tolist()
            proposal = tokens[start:start + NOTE_ATTRIBUTES].clone()
            probabilities = []
            if 0 in indices:
                value, probability = _draw(logits[start], allowed[1] + ([blank_id] if start in canvas["allow_blank"] else []), temperature, top_k, generator)
                proposal[0] = value; probabilities.append(probability)
                if value == blank_id:
                    proposal[:] = blank_id
                    candidates.append((probability, start, proposal)); continue
            instrument = tokenizer.id_to_tok[int(proposal[0])][1]
            for field in indices:
                if field == 0:
                    continue
                choices = drum_notes if field == 1 and instrument == "drum" else allowed[field + 1]
                value, probability = _draw(logits[start + field], choices, temperature, top_k, generator)
                proposal[field] = value; probabilities.append(probability)
            confidence = math.exp(sum(math.log(max(1e-30, p)) for p in probabilities) / len(probabilities))
            candidates.append((confidence, start, proposal))
        # Re-predict low-confidence masks next pass; never change committed/source tokens.
        count = math.ceil(len(pending) / (steps - iteration))
        committed = set()
        for _, start, proposal in sorted(candidates, key=lambda c: c[0], reverse=True)[:count]:
            tokens[start:start + NOTE_ATTRIBUTES] = proposal; committed.add(start)
        pending = [s for s in pending if s not in committed]
    if bool((tokens == mask_id).any()):
        raise RuntimeError("Sampler left unresolved masks")
    # Remove whole blank groups; a mixed blank/note group is always an error.
    result, i, added_notes, removed_notes = [], 0, 0, 0
    starts = set(canvas["slots"])
    while i < len(tokens):
        if i in starts:
            group = tokens[i:i + NOTE_ATTRIBUTES]
            if bool((group == blank_id).all()):
                removed_notes += int(i not in canvas["added"])
            elif bool((group == blank_id).any()):
                raise RuntimeError("Partial blank note")
            else:
                result.extend(tokenizer.id_to_tok[int(t)] for t in group)
                added_notes += int(i in canvas["added"])
            i += NOTE_ATTRIBUTES
        else:
            result.append(tokenizer.id_to_tok[int(tokens[i])]); i += 1
    return result, dict(passes=passes, blank_slots=len(canvas["added"]),
                       added_notes=added_notes, removed_notes=removed_notes)
