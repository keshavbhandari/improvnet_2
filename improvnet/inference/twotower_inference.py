"""Cascaded prompt continuation with the TwoTower denoiser.

Two inference strategies are supported:

* ``full-mask`` generates a novel continuation from fully masked blocks.
* ``onset-duration`` keeps the source event scaffold and regenerates timing.

Each target block is refined through the same four-block staircase used during
training.  The confidence sampler and ADAS share the same fixed reveal quota;
ADAS changes only the order in which positions are committed.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
from collections import defaultdict
from typing import Any, Sequence

import torch

from improvnet.model.ar_model import ARContextModel
from improvnet.model.twotower_config import (
    AR_MODEL_PATH,
    BLOCK_SIZE,
    MASK_ID,
    NUM_DRAFTS,
    PAD_ID,
    PROMPT_MAX,
    SAVE_DIR,
    SEP_ID,
)
from improvnet.model.twotower_denoiser import TwoTowerDenoiser
from improvnet.utils.ar_utils import ProcessData


EVENT_CATEGORIES = frozenset({"instrument", "note", "onset", "dur"})
TIMING_CATEGORIES = frozenset({"onset", "dur"})
FULL_POOL = "allowed"


def token_category(token: Any) -> str:
    """Return the vocabulary/event category used for constrained decoding."""
    if isinstance(token, tuple) and token:
        category = token[0]
        if category in EVENT_CATEGORIES:
            return category
        if category == "prefix":
            return "prefix"
    return "structural"


class VocabularyConstraints:
    """Vocabulary pools used by scaffolded and unconstrained continuation."""

    _ALLOWED_STRUCTURAL = frozenset(
        {"<E>", "<D>", "<T>", "<PED_ON>", "<PED_OFF>", "<X>"}
    )

    def __init__(self, tokenizer):
        pools: dict[str, list[int]] = {category: [] for category in EVENT_CATEGORIES}
        allowed: list[int] = []

        for token_id in range(tokenizer.vocab_size):
            token = tokenizer.id_to_tok[token_id]
            category = token_category(token)
            if category in EVENT_CATEGORIES:
                pools[category].append(token_id)
                allowed.append(token_id)
            elif isinstance(token, str) and token in self._ALLOWED_STRUCTURAL:
                allowed.append(token_id)

        pools[FULL_POOL] = allowed
        for name, pool in pools.items():
            if not pool:
                raise ValueError(f"Vocabulary pool {name!r} is empty")
        self.pools = pools

    def ids(self, category: str, device: torch.device) -> torch.Tensor:
        if category not in self.pools:
            raise ValueError(f"No vocabulary constraint for category {category!r}")
        return torch.tensor(self.pools[category], dtype=torch.long, device=device)


def constrained_top_predictions(
    logits: torch.Tensor,
    categories: Sequence[str | None],
    constraints: VocabularyConstraints,
    temperature: float = 1.0,
    current_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find constrained argmax tokens, confidence, and current-token probability.

    Positions whose category is ``None`` are left with prediction ``-1`` and
    zero probability. Probabilities are normalized inside the relevant pool,
    rather than over tokens that inference will never permit at that position.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be [length, vocab], got {tuple(logits.shape)}")
    if len(categories) != logits.shape[0]:
        raise ValueError("categories must contain one entry per logit position")
    if temperature <= 0:
        raise ValueError("confidence_temperature must be greater than zero")
    if current_ids is not None and current_ids.shape != (logits.shape[0],):
        raise ValueError("current_ids must have shape [length]")

    predictions = torch.full(
        (logits.shape[0],), -1, dtype=torch.long, device=logits.device
    )
    confidence = torch.zeros(logits.shape[0], dtype=torch.float32, device=logits.device)
    current_probability = torch.zeros_like(confidence)

    grouped_positions: dict[str, list[int]] = defaultdict(list)
    for position, category in enumerate(categories):
        if category is not None:
            grouped_positions[category].append(position)

    for category, positions in grouped_positions.items():
        position_ids = torch.tensor(positions, dtype=torch.long, device=logits.device)
        pool_ids = constraints.ids(category, logits.device)
        selected_logits = logits.index_select(0, position_ids).index_select(1, pool_ids)
        probabilities = torch.softmax(selected_logits.float() / temperature, dim=-1)
        category_confidence, relative_prediction = probabilities.max(dim=-1)

        predictions[position_ids] = pool_ids[relative_prediction]
        confidence[position_ids] = category_confidence

        if current_ids is not None:
            relative_by_id = {
                token_id: relative_id
                for relative_id, token_id in enumerate(constraints.pools[category])
            }
            for row, position in enumerate(positions):
                relative_id = relative_by_id.get(int(current_ids[position].item()))
                if relative_id is not None:
                    current_probability[position] = probabilities[row, relative_id]

    return predictions, confidence, current_probability


def select_reveal_positions(
    confidence: torch.Tensor,
    candidate_mask: torch.Tensor,
    quota: int,
    sampler: str,
    attention: torch.Tensor | None = None,
    alpha: float = 40.0,
) -> list[int]:
    """Select positions using confidence or greedy attention-discounted ADAS."""
    if confidence.ndim != 1 or candidate_mask.shape != confidence.shape:
        raise ValueError("confidence and candidate_mask must be matching vectors")

    candidates = candidate_mask.nonzero(as_tuple=True)[0]
    quota = min(max(int(quota), 0), int(candidates.numel()))
    if quota == 0:
        return []

    if sampler == "confidence":
        order = torch.argsort(confidence[candidates], descending=True, stable=True)
        return candidates[order[:quota]].tolist()
    if sampler != "adas":
        raise ValueError(f"Unknown sampler {sampler!r}")
    if attention is None or attention.shape != (confidence.numel(), confidence.numel()):
        raise ValueError("ADAS requires a square current-draft attention matrix")

    remaining = candidates.clone()
    scores = confidence[remaining].float().clone()
    selected: list[int] = []
    for _ in range(quota):
        winner_offset = int(torch.argmax(scores).item())
        winner = int(remaining[winner_offset].item())
        selected.append(winner)

        keep = torch.ones(remaining.numel(), dtype=torch.bool, device=remaining.device)
        keep[winner_offset] = False
        remaining = remaining[keep]
        scores = scores[keep]
        if remaining.numel() == 0:
            break

        discount = alpha * attention[remaining, winner].float() * (
            1.0 - confidence[winner].float()
        )
        scores = scores - discount

    return selected


def build_trajectory(
    snapshots: Sequence[torch.Tensor],
    timesteps: Sequence[float],
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Join fixed-size draft snapshots using the training-time SEP layout."""
    if not snapshots or len(snapshots) != len(timesteps):
        raise ValueError("snapshots and timesteps must be nonempty and equally sized")
    if valid_mask.shape != (BLOCK_SIZE,):
        raise ValueError(f"valid_mask must have shape ({BLOCK_SIZE},)")

    trajectory: list[torch.Tensor] = []
    trajectory_times: list[torch.Tensor] = []
    trajectory_mask: list[torch.Tensor] = []
    for index, (snapshot, timestep) in enumerate(zip(snapshots, timesteps)):
        if snapshot.shape != (BLOCK_SIZE,):
            raise ValueError(f"snapshot must have shape ({BLOCK_SIZE},)")
        trajectory.append(snapshot)
        trajectory_times.append(
            torch.full(
                (BLOCK_SIZE,), float(timestep), dtype=torch.float32, device=snapshot.device
            )
        )
        trajectory_mask.append(valid_mask)
        if index < len(snapshots) - 1:
            trajectory.append(
                torch.tensor([SEP_ID], dtype=torch.long, device=snapshot.device)
            )
            trajectory_times.append(
                torch.tensor([float(timestep)], dtype=torch.float32, device=snapshot.device)
            )
            trajectory_mask.append(
                torch.ones(1, dtype=torch.bool, device=snapshot.device)
            )

    return (
        torch.cat(trajectory),
        torch.cat(trajectory_times),
        torch.cat(trajectory_mask),
    )


def truncate_to_complete_events(tokens: Sequence[Any]) -> list[Any]:
    """Drop a final partial scaffold event after a hard token-length cap."""
    last_complete = 0
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if isinstance(token, str):
            if token in {"<PED_ON>", "<PED_OFF>"}:
                if index + 1 >= len(tokens) or token_category(tokens[index + 1]) != "onset":
                    break
                index += 2
            else:
                index += 1
            last_complete = index
            continue

        if token_category(token) == "instrument":
            expected = ("note", "onset", "dur")
            if index + 3 >= len(tokens) or tuple(
                token_category(tokens[index + offset]) for offset in range(1, 4)
            ) != expected:
                break
            index += 4
            last_complete = index
            continue

        # Normalized MIDI sequences should not begin an event with a bare
        # note/onset/duration token. Stop rather than emit a partial event.
        break

    return list(tokens[:last_complete])


def _load_checkpoint_state(path: str) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint {path!r} does not contain a model state dictionary")
    if next(iter(state)).startswith("module."):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


class TwoTowerInference:
    def __init__(
        self,
        ar_model_path: str,
        twotower_model_path: str,
        device: str | torch.device | None = None,
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.processor = ProcessData()
        self.tokenizer = self.processor.tokenizer
        self.constraints = VocabularyConstraints(self.tokenizer)

        print(f"Loading Tower A from {ar_model_path}...")
        self.model_ar = ARContextModel()
        self.model_ar.load_state_dict(_load_checkpoint_state(ar_model_path))
        self.model_ar.to(self.device).eval()

        print(f"Loading Tower B from {twotower_model_path}...")
        self.model_denoiser = TwoTowerDenoiser()
        self.model_denoiser.load_state_dict(_load_checkpoint_state(twotower_model_path))
        self.model_denoiser.to(self.device).eval()

        if self.device.type == "cuda":
            major = torch.cuda.get_device_capability(self.device)[0]
            self.amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            self.amp_dtype = None

    def _autocast(self):
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    @torch.no_grad()
    def _tower_a_context(
        self,
        committed_ids: Sequence[int],
        genre: torch.Tensor,
        multi_hot: torch.Tensor,
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, int]:
        context_ids = list(committed_ids[-PROMPT_MAX:])
        if not context_ids:
            context_ids = [self.tokenizer.tok_to_id["<S>"]]
        context = torch.tensor(context_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        with self._autocast():
            _, past_kv = self.model_ar(
                target=context,
                genre=genre,
                multi_hot=multi_hot,
                use_cache=True,
                past_key_values=None,
                seq_offset=0,
            )
        context_mask = torch.ones(
            (1, len(context_ids) + 2), dtype=torch.bool, device=self.device
        )
        return past_kv, context_mask, len(context_ids) + 2

    def _prediction_categories(
        self,
        state: torch.Tensor,
        valid_mask: torch.Tensor,
        source_roles: Sequence[str],
        strategy: str,
        revision_mode: str,
    ) -> list[str | None]:
        categories: list[str | None] = [None] * BLOCK_SIZE
        for position in range(BLOCK_SIZE):
            if not bool(valid_mask[position]):
                continue
            if int(state[position].item()) == MASK_ID:
                categories[position] = (
                    FULL_POOL if strategy == "full-mask" else source_roles[position]
                )
                continue
            if revision_mode == "freeze":
                continue
            if revision_mode == "timing":
                category = (
                    token_category(self.tokenizer.id_to_tok[int(state[position].item())])
                    if strategy == "full-mask"
                    else source_roles[position]
                )
                if category in TIMING_CATEGORIES:
                    categories[position] = category
            elif revision_mode == "all":
                if strategy == "full-mask":
                    categories[position] = FULL_POOL
                elif source_roles[position] in EVENT_CATEGORIES:
                    categories[position] = source_roles[position]
            else:
                raise ValueError(f"Unknown revision mode {revision_mode!r}")
        return categories

    @torch.no_grad()
    def _decode_block(
        self,
        initial_state: torch.Tensor,
        valid_mask: torch.Tensor,
        source_roles: Sequence[str],
        strategy: str,
        sampler: str,
        adas_alpha: float,
        revision_mode: str,
        revision_margin: float,
        confidence_temperature: float,
        context_kv,
        context_attention_mask: torch.Tensor,
        seq_offset: int,
        genre: torch.Tensor,
        multi_hot: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        state = initial_state.clone()
        snapshots: list[torch.Tensor] = []
        snapshot_timesteps: list[float] = []
        total_revisions = 0
        valid_count = max(int(valid_mask.sum().item()), 1)

        for draft_index in range(NUM_DRAFTS):
            masked = (state == MASK_ID) & valid_mask
            timestep = float(masked.sum().item()) / valid_count
            snapshots.append(state.clone())
            snapshot_timesteps.append(timestep)
            trajectory, trajectory_times, trajectory_mask = build_trajectory(
                snapshots, snapshot_timesteps, valid_mask
            )

            need_attention = sampler == "adas" and bool(masked.any())
            with self._autocast():
                model_result = self.model_denoiser(
                    noisy_target=trajectory.unsqueeze(0),
                    timestep=trajectory_times.unsqueeze(0),
                    seq_offset=seq_offset,
                    context_kv_cache=context_kv,
                    attention_mask=trajectory_mask.unsqueeze(0),
                    context_attention_mask=context_attention_mask,
                    draft_size=BLOCK_SIZE + 1,
                    elasticity=torch.zeros(1, dtype=torch.float32, device=self.device),
                    genre=genre,
                    multi_hot=multi_hot,
                    return_last_attention=need_attention,
                )

            if need_attention:
                logits, last_attention = model_result
                if last_attention is None:
                    raise RuntimeError("Tower B did not return final-layer attention for ADAS")
            else:
                logits = model_result
                last_attention = None

            current_start = (len(snapshots) - 1) * (BLOCK_SIZE + 1)
            current_logits = logits[0, current_start : current_start + BLOCK_SIZE]
            categories = self._prediction_categories(
                state, valid_mask, source_roles, strategy, revision_mode
            )
            predictions, confidence, current_probability = constrained_top_predictions(
                current_logits,
                categories,
                self.constraints,
                temperature=confidence_temperature,
                current_ids=state,
            )

            remaining_drafts = NUM_DRAFTS - draft_index
            remaining_masks = int(masked.sum().item())
            quota = math.ceil(remaining_masks / remaining_drafts) if remaining_masks else 0

            current_attention = None
            if need_attention:
                prefix_length = context_attention_mask.shape[1]
                key_start = prefix_length + current_start
                key_end = key_start + BLOCK_SIZE
                if last_attention.shape[1] != BLOCK_SIZE or key_end > last_attention.shape[2]:
                    raise RuntimeError(
                        "Unexpected final attention shape: "
                        f"{tuple(last_attention.shape)} for key range {key_start}:{key_end}"
                    )
                current_attention = last_attention[0, :, key_start:key_end]

            selected = select_reveal_positions(
                confidence,
                masked,
                quota,
                sampler=sampler,
                attention=current_attention,
                alpha=adas_alpha,
            )

            # Revisions use the pre-update state. Newly revealed positions are
            # therefore not eligible until a later draft, and every update in
            # this pass is applied simultaneously.
            revision_positions: list[int] = []
            for position, category in enumerate(categories):
                if category is None or bool(masked[position]):
                    continue
                predicted = int(predictions[position].item())
                current = int(state[position].item())
                if predicted == current:
                    continue
                improvement = float(
                    (confidence[position] - current_probability[position]).item()
                )
                if improvement >= revision_margin:
                    revision_positions.append(position)

            updated = state.clone()
            if selected:
                selected_tensor = torch.tensor(selected, dtype=torch.long, device=self.device)
                updated[selected_tensor] = predictions[selected_tensor]
            if revision_positions:
                revision_tensor = torch.tensor(
                    revision_positions, dtype=torch.long, device=self.device
                )
                updated[revision_tensor] = predictions[revision_tensor]
            state = updated
            total_revisions += len(revision_positions)

            print(
                f"    draft {draft_index + 1}/{NUM_DRAFTS}: "
                f"t={timestep:.3f}, revealed={len(selected)}, "
                f"revised={len(revision_positions)}, "
                f"remaining_masks={int(((state == MASK_ID) & valid_mask).sum().item())}"
            )

        unresolved = (state == MASK_ID) & valid_mask
        if bool(unresolved.any()):
            raise RuntimeError(f"{int(unresolved.sum().item())} masks remain after final draft")
        return state, total_revisions

    @torch.no_grad()
    def generate(
        self,
        source_tokens: Sequence[Any],
        genre_str: str = "jazz",
        strategy: str = "full-mask",
        prompt_len: int = 256,
        max_tokens: int = 2048,
        sampler: str = "adas",
        adas_alpha: float = 40.0,
        revision_mode: str = "timing",
        revision_margin: float = 0.10,
        confidence_temperature: float = 1.0,
    ) -> list[Any]:
        if strategy not in {"full-mask", "onset-duration"}:
            raise ValueError(f"Unknown generation strategy {strategy!r}")
        if sampler not in {"adas", "confidence"}:
            raise ValueError(f"Unknown sampler {sampler!r}")
        if revision_mode not in {"freeze", "timing", "all"}:
            raise ValueError(f"Unknown revision mode {revision_mode!r}")
        if prompt_len < 1:
            raise ValueError("prompt_len must be at least one token")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least one token")
        if revision_margin < 0:
            raise ValueError("revision_margin cannot be negative")

        normalized_source = self.processor.normalize_token_sequence(list(source_tokens))
        if not normalized_source:
            raise ValueError("The input MIDI produced no tokens")

        actual_prompt_len = min(prompt_len, len(normalized_source), max_tokens)
        committed = list(normalized_source[:actual_prompt_len])
        eos_token = "<E>"
        if eos_token in committed:
            return committed[: committed.index(eos_token) + 1]

        conditioning_tokens = (
            committed if strategy == "full-mask" else normalized_source
        )
        multi_hot = self.processor.get_instrument_multihot(conditioning_tokens)
        multi_hot = multi_hot.unsqueeze(0).to(self.device)
        genre = torch.tensor(
            [self.processor.get_genre_id(genre_str)], dtype=torch.long, device=self.device
        )
        source_roles = [token_category(token) for token in normalized_source]
        source_cursor = actual_prompt_len
        block_number = 0

        while len(committed) < max_tokens:
            remaining_budget = max_tokens - len(committed)
            if strategy == "full-mask":
                valid_length = min(BLOCK_SIZE, remaining_budget)
                block_tokens: list[Any] = []
                block_roles = [FULL_POOL] * valid_length
                initial_ids = torch.full(
                    (BLOCK_SIZE,), PAD_ID, dtype=torch.long, device=self.device
                )
                initial_ids[:valid_length] = MASK_ID
            else:
                available = len(normalized_source) - source_cursor
                if available <= 0:
                    break
                valid_length = min(BLOCK_SIZE, remaining_budget, available)
                block_tokens = list(
                    normalized_source[source_cursor : source_cursor + valid_length]
                )
                block_roles = source_roles[source_cursor : source_cursor + valid_length]
                initial_ids = torch.full(
                    (BLOCK_SIZE,), PAD_ID, dtype=torch.long, device=self.device
                )
                source_ids = self.processor.tokens_to_tensor(block_tokens).to(self.device)
                initial_ids[:valid_length] = source_ids
                for position, category in enumerate(block_roles):
                    if category in TIMING_CATEGORIES:
                        initial_ids[position] = MASK_ID

            valid_mask = torch.zeros(BLOCK_SIZE, dtype=torch.bool, device=self.device)
            valid_mask[:valid_length] = True
            padded_roles = list(block_roles) + ["structural"] * (BLOCK_SIZE - valid_length)

            committed_ids = self.processor.tokens_to_tensor(committed).tolist()
            context_kv, context_attention_mask, seq_offset = self._tower_a_context(
                committed_ids, genre, multi_hot
            )
            block_number += 1
            print(
                f"Block {block_number}: strategy={strategy}, valid_tokens={valid_length}, "
                f"history={min(len(committed), PROMPT_MAX)}"
            )
            final_ids, revision_count = self._decode_block(
                initial_state=initial_ids,
                valid_mask=valid_mask,
                source_roles=padded_roles,
                strategy=strategy,
                sampler=sampler,
                adas_alpha=adas_alpha,
                revision_mode=revision_mode,
                revision_margin=revision_margin,
                confidence_temperature=confidence_temperature,
                context_kv=context_kv,
                context_attention_mask=context_attention_mask,
                seq_offset=seq_offset,
                genre=genre,
                multi_hot=multi_hot,
            )

            generated = [
                self.tokenizer.id_to_tok[int(token_id)]
                for token_id in final_ids[:valid_length].tolist()
            ]
            if eos_token in generated:
                generated = generated[: generated.index(eos_token) + 1]
                committed.extend(generated)
                print(
                    f"Block {block_number} finalized {len(generated)} tokens with "
                    f"{revision_count} revisions and produced <E>."
                )
                break

            committed.extend(generated)
            if strategy == "onset-duration":
                source_cursor += valid_length
            print(
                f"Block {block_number} finalized {len(generated)} tokens with "
                f"{revision_count} revisions."
            )

            if strategy == "onset-duration" and source_cursor >= len(normalized_source):
                break

        # This is a no-op for a complete scaffold and also handles the case in
        # which max_tokens lands exactly on a 256-token block boundary inside
        # a four-token event.
        if strategy == "onset-duration":
            committed = truncate_to_complete_events(committed)
        return committed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input MIDI used for the prompt/scaffold")
    parser.add_argument(
        "--output", default="generated_twotower.mid", help="Generated MIDI output path"
    )
    parser.add_argument("--ar-model", default=AR_MODEL_PATH)
    parser.add_argument(
        "--twotower-model", default=os.path.join(SAVE_DIR, "latest_checkpoint.pt")
    )
    parser.add_argument("--genre", default="jazz")
    parser.add_argument(
        "--strategy", choices=("full-mask", "onset-duration"), default="full-mask"
    )
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--sampler", choices=("adas", "confidence"), default="adas")
    parser.add_argument("--adas-alpha", type=float, default=40.0)
    parser.add_argument(
        "--revision-mode", choices=("freeze", "timing", "all"), default="timing"
    )
    parser.add_argument("--revision-margin", type=float, default=0.10)
    parser.add_argument("--confidence-temperature", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    inference = TwoTowerInference(args.ar_model, args.twotower_model)

    print(f"Reading {args.input}...")
    source_midi = inference.processor.read_midi(args.input)
    source_tokens = inference.processor.midi_to_tokens(source_midi)
    generated_tokens = inference.generate(
        source_tokens=source_tokens,
        genre_str=args.genre,
        strategy=args.strategy,
        prompt_len=args.prompt_len,
        max_tokens=args.max_tokens,
        sampler=args.sampler,
        adas_alpha=args.adas_alpha,
        revision_mode=args.revision_mode,
        revision_margin=args.revision_margin,
        confidence_temperature=args.confidence_temperature,
    )

    output_directory = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_directory, exist_ok=True)
    print(f"Saving {len(generated_tokens)} tokens to {args.output}...")
    output_midi = inference.processor.tokens_to_midi(generated_tokens)
    inference.processor.save_midi(output_midi, args.output)
    print("Generation complete.")


if __name__ == "__main__":
    main()
