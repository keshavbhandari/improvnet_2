import math
import random

import torch

from improvnet.model.twotower_config import (
    BLOCK_SIZE,
    BOUNDARY_TOKEN_MASK_PROBS,
    ELASTICITY_MAX_RATIO,
    ELASTICITY_MIN_RATIO,
    FULL_VOCAB_CORRUPTION_PROB,
    MASK_ID,
    NON_MASK_CORRUPTION_PROBS,
    PAD_ID,
)


class TwoTowerCorruptionStrategy:
    """Structured target augmentation and draft corruption for Tower B."""

    def __init__(self, processor):
        self.processor = processor
        self._corruption_pools = None
        self._full_corruption_pool = None

    def instrument_associations(self, tokens):
        """Return the instrument owning each token in a normalized sequence."""
        associations = []
        current_instrument = None

        for token in tokens:
            instrument = None
            if isinstance(token, tuple) and token:
                token_type = token[0]
                if (
                    token_type == "instrument"
                    and len(token) >= 2
                    and token[1] in self.processor.INSTRUMENT_CLASSES
                ):
                    current_instrument = token[1]
                    instrument = current_instrument
                elif token_type in ("note", "onset", "dur"):
                    instrument = current_instrument
                    if token_type == "dur":
                        current_instrument = None
                elif token_type in self.processor.INSTRUMENT_CLASSES:
                    # Legacy compact note token; onset and duration follow it.
                    current_instrument = token_type
                    instrument = current_instrument
                else:
                    current_instrument = None
            else:
                current_instrument = None
            associations.append(instrument)

        return associations

    def insert_elastic_blanks(self, tokens, associations, enabled):
        """Reserve a conditioned 10-20% of the target for blank event slots."""
        tokens = list(tokens[:BLOCK_SIZE])
        associations = list(associations[:BLOCK_SIZE])
        if not enabled or len(tokens) < 4:
            return tokens, associations, 0.0

        # Blanks represent a complete four-token note event. Sampling a number
        # of groups keeps the realized fraction variable while preserving event
        # alignment and remaining inside the configured ratio range.
        candidates = []
        for num_groups in range(1, (BLOCK_SIZE // 4) + 1):
            num_blanks = 4 * num_groups
            min_final_length = math.ceil(num_blanks / ELASTICITY_MAX_RATIO)
            max_final_length = math.floor(num_blanks / ELASTICITY_MIN_RATIO)
            final_length = min(BLOCK_SIZE, len(tokens) + num_blanks, max_final_length)
            content_budget = final_length - num_blanks
            if min_final_length <= final_length and 0 < content_budget <= len(tokens):
                candidates.append((num_groups, final_length))

        # Very short fragments cannot fit a complete four-token blank event
        # without exceeding the requested maximum ratio; keep those rigid.
        if not candidates:
            return tokens, associations, 0.0

        num_groups, final_length = random.choice(candidates)
        num_blanks = 4 * num_groups
        content_budget = final_length - num_blanks

        # If a full block needs room, remove complete musical events rather
        # than truncating the tail, which could silently discard <D> or <E>.
        remove_count = max(0, len(tokens) - content_budget)
        remove_indices = set()
        event_spans = []
        for idx in range(max(0, len(tokens) - 3)):
            fields = tokens[idx:idx + 4]
            if (
                isinstance(fields[0], tuple)
                and fields[0][0] == "instrument"
                and all(isinstance(field, tuple) for field in fields)
                and [field[0] for field in fields] == ["instrument", "note", "onset", "dur"]
            ):
                event_spans.append(tuple(range(idx, idx + 4)))

        random.shuffle(event_spans)
        for span in event_spans:
            if len(remove_indices) + 4 > remove_count:
                break
            if not any(idx in remove_indices for idx in span):
                remove_indices.update(span)

        # Defensive fallback for cropped/malformed fragments. Boundary markers
        # remain protected even when a full event decomposition is impossible.
        if len(remove_indices) < remove_count:
            for idx in range(len(tokens) - 1, -1, -1):
                if len(remove_indices) >= remove_count:
                    break
                if idx not in remove_indices and tokens[idx] not in ("<D>", "<E>"):
                    remove_indices.add(idx)

        tokens = [token for idx, token in enumerate(tokens) if idx not in remove_indices]
        associations = [
            instrument for idx, instrument in enumerate(associations)
            if idx not in remove_indices
        ]
        tokens = tokens[:content_budget]
        associations = associations[:content_budget]

        # Insert only at event boundaries so a blank group never splits the
        # instrument/note/onset/duration fields of a surviving event.
        boundaries = [
            idx for idx, token in enumerate(tokens)
            if isinstance(token, tuple) and token and token[0] == "instrument"
        ]
        boundaries.append(len(tokens))
        insertion_points = sorted(
            (random.choice(boundaries) for _ in range(num_groups)), reverse=True
        )
        for idx in insertion_points:
            tokens[idx:idx] = ["<BLANK>"] * 4
            associations[idx:idx] = [None] * 4

        elasticity = num_blanks / max(1, len(tokens))
        return tokens, associations, elasticity

    def _corruption_category(self, token):
        if not isinstance(token, tuple) or not token:
            return "special"
        token_type = token[0]
        if token_type in ("instrument", "note", "onset", "dur"):
            return token_type
        if token_type in self.processor.INSTRUMENT_CLASSES:
            return "note"
        return f"tuple:{token_type}"

    def _get_corruption_pools(self):
        if self._corruption_pools is not None:
            return self._corruption_pools, self._full_corruption_pool

        pools = {}
        full_pool = []
        excluded_ids = {PAD_ID, MASK_ID}
        for token_id, token in self.processor.tokenizer.id_to_tok.items():
            if token_id in excluded_ids:
                continue
            pools.setdefault(self._corruption_category(token), []).append(token_id)
            full_pool.append(token_id)

        self._corruption_pools = pools
        self._full_corruption_pool = full_pool
        return pools, full_pool

    @staticmethod
    def _sample_different(pool, original_id):
        if len(pool) <= 1:
            return original_id
        replacement = random.choice(pool)
        while replacement == original_id:
            replacement = random.choice(pool)
        return replacement

    def corrupt_visible_tokens(
        self,
        draft_input,
        target_tensor,
        valid_indices,
        draft_idx,
        protected_visible_mask=None,
    ):
        """Replace a scheduled fraction of currently visible, non-mask tokens."""
        corruption_mask = torch.zeros_like(draft_input, dtype=torch.bool)
        if draft_idx >= len(NON_MASK_CORRUPTION_PROBS):
            return corruption_mask

        corruption_prob = float(NON_MASK_CORRUPTION_PROBS[draft_idx])
        if corruption_prob <= 0.0:
            return corruption_mask

        is_visible = draft_input[valid_indices] != MASK_ID
        if protected_visible_mask is not None:
            is_visible &= ~protected_visible_mask[valid_indices]
        visible_indices = valid_indices[is_visible]
        if len(visible_indices) == 0:
            return corruption_mask

        selected = visible_indices[torch.rand(len(visible_indices)) < corruption_prob]
        pools, full_pool = self._get_corruption_pools()
        for position in selected.tolist():
            original_id = target_tensor[position].item()
            original_token = self.processor.tokenizer.id_to_tok[original_id]
            if random.random() < FULL_VOCAB_CORRUPTION_PROB:
                pool = full_pool
            else:
                pool = pools.get(self._corruption_category(original_token), full_pool)
            replacement_id = self._sample_different(pool, original_id)
            if replacement_id != original_id:
                draft_input[position] = replacement_id
                corruption_mask[position] = True

        return corruption_mask

    def mask_boundary_tokens(self, draft_input, target_tensor, valid_indices, draft_idx):
        """Give <D>/<E> explicit reconstruction practice in every draft stage."""
        protected_visible_mask = torch.zeros_like(draft_input, dtype=torch.bool)
        if draft_idx >= len(BOUNDARY_TOKEN_MASK_PROBS):
            return protected_visible_mask
        mask_prob = float(BOUNDARY_TOKEN_MASK_PROBS[draft_idx])

        dim_id = self.processor.tokenizer.tok_to_id.get("<D>")
        end_id = self.processor.tokenizer.tok_to_id.get("<E>")
        if dim_id is None and end_id is None:
            return protected_visible_mask

        target_values = target_tensor[valid_indices]
        dim_positions = valid_indices[target_values == dim_id] if dim_id is not None else valid_indices[:0]
        end_positions = valid_indices[target_values == end_id] if end_id is not None else valid_indices[:0]

        # When the pair is available, train both conditional directions:
        # visible <D> -> recover <E>, or visible <E> -> recover <D>.
        if len(dim_positions) > 0 and len(end_positions) > 0:
            if random.random() < 0.5:
                visible_positions, masked_positions = dim_positions, end_positions
            else:
                visible_positions, masked_positions = end_positions, dim_positions
            draft_input[visible_positions] = target_tensor[visible_positions]
            draft_input[masked_positions] = MASK_ID
            protected_visible_mask[visible_positions] = True
            return protected_visible_mask

        boundary_positions = dim_positions if len(dim_positions) > 0 else end_positions
        if len(boundary_positions) == 0 or mask_prob <= 0.0:
            return protected_visible_mask

        still_visible = draft_input[boundary_positions] != MASK_ID
        selected = torch.rand(len(boundary_positions)) < mask_prob
        draft_input[boundary_positions[still_visible & selected]] = MASK_ID
        return protected_visible_mask
