"""Slot-constrained, two-stage style transfer with the existing SFT checkpoint.

Timing is refined first; joint pitch/velocity note tokens are refined second.
Only finalized blocks enter causal history. Instruments, metadata and structure
are immutable. This is an inference adaptation, not the full-vocabulary forward
process used in SFT. The model has NO explicit timestep argument.
"""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class SamplerConfig:
    steps: int = 48                 # PER STAGE, up to 96 passes per block.
    sampler: str = "flow"
    temperature_start: float = .8
    temperature_end: float = .4
    entropy_budget: float = .1
    stopping_entropy: float = .005
    adaptive_stopping: bool = True
    self_conditioning: bool = True

    def __post_init__(self):
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or not 1 <= self.steps <= 48:
            raise ValueError("steps must be an integer between 1 and 48 per stage")
        if self.sampler not in ("flow", "entropy"):
            raise ValueError("sampler must be flow or entropy")
        for name in ("temperature_start", "temperature_end", "entropy_budget", "stopping_entropy"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name.startswith("temperature") and value == 0):
                raise ValueError(f"Invalid {name}")


class SlotVocabulary:
    """Vocabulary types, not offsets modulo four: slices can split events.

    Drums use the same four-field format after ProcessData normalization. Note
    tokens jointly encode pitch/velocity, including drum pitch. Zero velocity
    and zero duration are excluded from editable candidates to avoid silence.
    """
    def __init__(self, tokenizer, device="cpu"):
        self.roles = torch.zeros(tokenizer.vocab_size, dtype=torch.long, device=device)
        self.candidates = torch.zeros(4, tokenizer.vocab_size, dtype=torch.bool, device=device)
        for index, token in tokenizer.id_to_tok.items():
            if isinstance(token, tuple) and token[0] in ("onset", "dur", "note"):
                role = {"onset": 1, "dur": 2, "note": 3}[token[0]]
                self.roles[index] = role
                sounding = token[0] == "onset" or token[-1] > 0
                self.candidates[role, index] = sounding
        if not bool(self.candidates[1:].any(-1).all()):
            raise ValueError("Need onset, positive-duration and sounding-note vocabulary")

    def constraints(self, anchor, valid_mask, stage):
        if anchor.dtype != torch.long or anchor.ndim != 2 or valid_mask.shape != anchor.shape or valid_mask.dtype != torch.bool:
            raise ValueError("Need long [batch,canvas] IDs and a matching boolean validity mask")
        if stage not in ("timing", "pitch-velocity"):
            raise ValueError("Unknown stage")
        roles = self.roles[anchor]
        editable = valid_mask & ((roles == 1) | (roles == 2) if stage == "timing" else roles == 3)
        allowed = self.candidates[roles] & editable[..., None]
        # Fixed positions, including padded positions, have a point mass at
        # their anchor ID. Clamp both sampling AND self-conditioning this way.
        allowed.scatter_(-1, anchor[..., None], allowed.gather(-1, anchor[..., None]) | ~editable[..., None])
        return editable, allowed


def typed_corrupt(anchor, editable, allowed, noise, generator):
    if not math.isfinite(noise) or not 0 <= noise <= 1:
        raise ValueError("noise must be in [0,1]")
    replacement = torch.rand(anchor.shape, device=anchor.device, generator=generator) < noise
    draws = torch.multinomial(allowed.flatten(0, 1).float(), 1, generator=generator).reshape_as(anchor)
    replaced = replacement & editable
    return torch.where(replaced, draws, anchor), replaced


def entropy_selection(entropies, editable, budget):
    """Gemma entropy bound: sum of the first k-1 entropies <= budget."""
    if entropies.shape != editable.shape or entropies.ndim != 2 or not math.isfinite(budget) or budget < 0:
        raise ValueError("Invalid entropy bound or mask")
    if not bool(torch.isfinite(entropies).all()) or bool((entropies < 0).any()):
        raise ValueError("Entropies must be finite and nonnegative")
    ranked = entropies.masked_fill(~editable, float("inf")).argsort(dim=-1, stable=True)
    ordered = entropies.gather(1, ranked)
    preceding = torch.cat((torch.zeros_like(ordered[:, :1]), ordered[:, :-1].cumsum(-1)), -1)
    return torch.zeros_like(editable).scatter(1, ranked, (preceding <= budget) & editable.gather(1, ranked))


def constrained_probabilities(logits, allowed, temperatures):
    if logits.shape != allowed.shape or logits.ndim != 3 or not bool(torch.isfinite(logits).all()):
        raise ValueError("Need finite [batch,canvas,vocab] logits and matching restrictions")
    temperatures = torch.as_tensor(temperatures, device=logits.device)
    if temperatures.shape != (len(logits),) or not bool(torch.isfinite(temperatures).all()) or bool((temperatures <= 0).any()):
        raise ValueError("Need a positive temperature per row")
    scores = (logits.float() / temperatures[:, None, None]).masked_fill(~allowed, -float("inf"))
    probabilities = scores.softmax(-1)
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("A slot has no allowed vocabulary candidates")
    entropies = -(probabilities * probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(-1)
    return scores, probabilities, entropies


@torch.inference_mode()
def refine_stage(model, cache, anchor, valid_mask, slots, *, stage, noise, config, generators, sampler_generator):
    """Independent row stopping; hard constraints at every forward/reverse step.

    t_edit decays linearly to zero. t_effective = t_edit * editable/valid,
    an approximate *whole-canvas replacement rate*, not an exact match to SFT's
    full-vocab corruption distribution. It scales temperature, not model inputs.
    Flow uses (s/t) delta(current) + (1-s/t) p(clean); entropy selection is an
    optional alternative that re-noises unselected editable positions.
    """
    editable, allowed = slots.constraints(anchor, valid_mask, stage)
    fraction = editable.sum(-1) / valid_mask.sum(-1).clamp_min(1)
    canvas = anchor.clone()
    corruptions = []
    for row, generator in enumerate(generators):
        canvas[row:row+1], replaced = typed_corrupt(anchor[row:row+1], editable[row:row+1], allowed[row:row+1], noise, generator)
        count = int(editable[row].sum())
        corruptions.append(dict(editable_positions=count, editable_fraction=float(fraction[row]),
                                replaced_positions=int(replaced.sum()), changed_positions=int((canvas[row] != anchor[row]).sum())))
    initial = canvas.clone()
    output = anchor.clone()
    active = editable.any(-1) & (noise > 0)
    traces = [[] for _ in range(len(canvas))]
    previous_scores = previous_prediction = None
    for index in range(config.steps):
        if not bool(active.any()):
            break
        t_edit = noise * (1 - index / config.steps)
        next_t_edit = noise * (1 - (index + 1) / config.steps)
        t_effective = fraction * t_edit
        temperatures = config.temperature_end + (config.temperature_start - config.temperature_end) * t_effective
        logits = model.denoise(canvas, cache=cache, attention_mask=valid_mask & active[:, None],
                               self_conditioning_logits=previous_scores if config.self_conditioning else None,
                               self_conditioning_temperature=1.)
        # Scores already contain each row's temperature; SC temperature=1 is
        # therefore exact, including one-hot fixed/padded positions.
        scores, probabilities, entropies = constrained_probabilities(logits, allowed, temperatures)
        prediction = scores.argmax(-1)
        mean_entropy = (entropies * editable).sum(-1) / editable.sum(-1).clamp_min(1)
        stable = torch.zeros_like(active) if previous_prediction is None else ((prediction == previous_prediction) | ~editable).all(-1)
        stop = active & stable & (mean_entropy <= config.stopping_entropy) & config.adaptive_stopping
        sampled = torch.multinomial(probabilities.flatten(0, 1), 1, generator=sampler_generator).reshape_as(canvas)
        # Keep temperature sampling at the final pass too; do not collapse
        # ambiguous note/timing distributions to argmax. Argmax is only the
        # stability diagnostic for adaptive stopping.
        finished = active & (stop | (index + 1 == config.steps))
        output = torch.where(finished[:, None], sampled, output)
        diagnostics = torch.stack((active.float(), t_effective, temperatures, mean_entropy, stable.float(), stop.float()), -1).cpu().tolist()
        for row, (running, effective, temperature, entropy, unchanged, stopped) in enumerate(diagnostics):
            if running:
                traces[row].append(dict(step=index+1, t_editable=t_edit, t_effective=effective,
                                        temperature=temperature, mean_entropy=entropy, stable=bool(unchanged), adaptive_stop=bool(stopped)))
        active &= ~finished
        if bool(active.any()):
            mask = editable & active[:, None]
            if config.sampler == "flow":
                selected = (torch.rand(canvas.shape, device=canvas.device, generator=sampler_generator) < 1-next_t_edit/t_edit) & mask
                updated = torch.where(selected, sampled, canvas)
            else:
                selected = entropy_selection(entropies, mask, config.entropy_budget)
                draws = torch.multinomial(allowed.flatten(0, 1).float(), 1, generator=sampler_generator).reshape_as(canvas)
                updated = torch.where(selected, sampled, draws)
            canvas = torch.where(mask, updated, canvas)
        previous_scores, previous_prediction = scores, prediction
    if not torch.equal(output[~editable], anchor[~editable]) or (noise > 0 and not bool(allowed.gather(-1, output[..., None]).all())):
        raise RuntimeError("Sampler violated a frozen slot or vocabulary constraint")
    return output, initial, traces, corruptions


def closed_source(tokens, max_tokens):
    """Close the SOURCE endpoint before inference, never trim generated output.

    Internal 256 boundaries remain exact, even inside events. Only the last
    source block can be partial. No synthetic E or future source suffix is used.
    """
    if max_tokens < 1:
        raise ValueError("Need a positive token cap")
    tokens = list(tokens)
    if "<E>" in tokens:
        tokens = tokens[:tokens.index("<E>")+1]
    end = min(len(tokens), max_tokens)
    while end and not (tokens[end-1] in ("<S>", "<T>", "<D>", "<E>") or
                       isinstance(tokens[end-1], tuple) and tokens[end-1][0] == "dur"):
        end -= 1
    if not end:
        raise ValueError("No complete source event in token cap")
    return tokens[:end]


def padded_ids(sequences, processor, device, width):
    ids = torch.full((len(sequences), width), processor.tokenizer.pad_id, device=device, dtype=torch.long)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        if len(sequence) > width:
            raise ValueError("Source exceeds canvas width")
        if sequence:
            ids[row, :len(sequence)] = processor.tokens_to_tensor(sequence).to(device)
            valid[row, :len(sequence)] = True
    return ids, valid


@torch.inference_mode()
def generate_batch(model, processor, examples, *, noise, config, max_blocks=8, seed=42, device=torch.device("cpu"), genre="jazz", strategy="timing-pitch"):
    if not examples or not 1 <= max_blocks <= 8 or model.config.canvas_size != 256:
        raise ValueError("This workflow uses 256-token blocks, at most eight")
    if strategy not in ("timing", "timing-pitch"):
        raise ValueError("strategy must be timing or timing-pitch")
    stages = ("timing",) if strategy == "timing" else ("timing", "pitch-velocity")
    slots = SlotVocabulary(processor.tokenizer, device)
    sources = [closed_source(e["tokens"], max_blocks*256) for e in examples]
    # No musical prompt on block 1. Conditioning controls are still available.
    empty = torch.empty(len(examples), 0, dtype=torch.long, device=device)
    cache = model.encode_context(empty, attention_mask=empty.bool(),
                                 genre=torch.full((len(examples),), processor.get_genre_id(genre), device=device),
                                 multi_hot=torch.stack([processor.get_instrument_multihot(s) for s in sources]).to(device),
                                 elasticity=torch.zeros(len(examples), device=device), return_logits=False).cache
    results = [dict(example_id=e["example_id"], initial_noise=noise, source_genre=e["source_genre"],
                    target_genre=genre, max_steps=config.steps, strategy=strategy,
                    generated_tokens=[], blocks=[], stop_reason="token_cap") for e in examples]
    for block in range(max_blocks):
        refs = [s[block*256:(block+1)*256] for s in sources]
        if not any(refs):
            break
        clean, valid = padded_ids(refs, processor, device, 256)
        snapshots, stage_records = {}, {}
        current = clean
        for stage_index, stage in enumerate(stages):
            generators = [torch.Generator(device=device).manual_seed(seed + e["sample_index"]*100003 + block*1009 + stage_index*7919) for e in examples]
            sampler_generator = torch.Generator(device=device).manual_seed(seed + block*1009 + stage_index*7919 + examples[0]["sample_index"]*100003 + 17)
            current, initial, traces, corruptions = refine_stage(model, cache, current, valid, slots, stage=stage,
                                                               noise=noise, config=config, generators=generators, sampler_generator=sampler_generator)
            snapshots[stage] = current.clone()
            stage_records[stage] = (initial, traces, corruptions)
        for row, ref in enumerate(refs):
            if not ref:
                continue
            tokens = processor.tokenizer.decode(current[row, :len(ref)].tolist())
            roles = slots.roles[clean[row, :len(ref)]]
            if any(a != b for a, b, r in zip(ref, tokens, roles.tolist()) if r == 0 or (strategy == "timing" and r == 3)):
                raise RuntimeError("A frozen instrument, structural or note slot changed")
            record = dict(block=block+1, source_start=block*256, source_end=block*256+len(ref),
                          timing_tokens=processor.tokenizer.decode(snapshots["timing"][row, :len(ref)].tolist()), stages={})
            for stage in stage_records:
                initial, traces, corruptions = stage_records[stage]
                record["stages"][stage] = dict(**corruptions[row], actual_steps=len(traces[row]), trace=traces[row],
                                              initial_token_ids=initial[row, :len(ref)].tolist())
            results[row]["blocks"].append(record)
            results[row]["generated_tokens"].extend(tokens)
            if "<E>" in ref:
                results[row]["stop_reason"] = "source_end_token"
            elif (block+1)*256 >= len(sources[row]):
                results[row]["stop_reason"] = "source_exhausted" if len(sources[row]) < max_blocks*256-3 else "token_cap"
        progress = "; ".join(f"{stage} passes={[len(t) for t in stage_records[stage][1] if t]}" for stage in stages)
        print(f"  block {block+1}/{max_blocks}: pieces={sum(bool(r) for r in refs)}; {progress}", flush=True)
        if block+1 < max_blocks and any(len(s) > (block+1)*256 for s in sources):
            cache = model.encode_context(current, cache=cache, attention_mask=valid, return_logits=False).cache
    return results
