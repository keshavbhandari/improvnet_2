"""Cosine compatibility for finalized musical excerpts; no probability claim."""

from numbers import Integral
import torch

from improvnet.tokenizer.absolute import AbsTokenizer
from improvnet.metrics.genre_classifier.tokenization import vocabulary_digest
from improvnet.metrics.continuity.continuity_model import ContinuityConfig, ContinuityEncoder


class ContinuityMetric:
    def __init__(self, model, tokenizer):
        if tokenizer.vocab_size != model.config.vocab_size or tokenizer.pad_id != model.config.pad_id:
            raise ValueError("Continuity model/tokenizer vocabulary mismatch")
        self.model, self.tokenizer = model.eval(), tokenizer
        self.device = next(model.parameters()).device

    @classmethod
    def from_checkpoint(cls, path, *, device="cpu", tokenizer=None):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format") != "continuity_encoder_v1" or state.get("trained_steps", 0) < 1:
            raise ValueError("Continuity rewards require a trained continuity checkpoint")
        tokenizer = tokenizer or AbsTokenizer()
        if tokenizer.config != state["tokenizer_config"] or vocabulary_digest(tokenizer) != state["vocabulary_digest"]:
            raise ValueError("Continuity checkpoint/tokenizer mismatch")
        model = ContinuityEncoder(ContinuityConfig(**state["config"]))
        model.load_state_dict(state["model_state_dict"], strict=True)
        return cls(model.to(device), tokenizer)

    def _ids(self, sequence, valid_mask=None):
        if isinstance(sequence, torch.Tensor) and sequence.ndim != 1:
            raise ValueError("Continuity scoring expects a one-dimensional excerpt")
        values = sequence.detach().cpu().tolist() if isinstance(sequence, torch.Tensor) else list(sequence)
        if any(isinstance(value, bool) for value in values):
            raise ValueError("Boolean values are not music token IDs")
        mask = [True] * len(values) if valid_mask is None else (valid_mask.detach().cpu().tolist() if isinstance(valid_mask, torch.Tensor) else list(valid_mask))
        if len(mask) != len(values) or any(value not in (0, 1, False, True) for value in mask):
            raise ValueError("valid_mask must match the excerpt")
        ids, has_note = [], False
        for value, keep in zip(values, mask):
            if not keep:
                continue
            if isinstance(value, Integral):
                token = self.tokenizer.id_to_tok.get(int(value))
            else:
                token = tuple(value) if isinstance(value, list) else value
            if isinstance(token, tuple) and token and token[0] == "prefix":
                continue
            if isinstance(token, tuple):
                if any(isinstance(field, bool) for field in token):
                    raise ValueError("Boolean values are not musical fields")
                if token and token[0] in ("note", "onset", "dur") and any(not isinstance(field, Integral) for field in token[1:]):
                    raise ValueError("Pitch, velocity and timing fields must be integers")
            if token in (self.tokenizer.pad_tok, self.tokenizer.mask_tok, self.tokenizer.blank_tok, self.tokenizer.sep_tok, self.tokenizer.unk_tok):
                raise ValueError("Continuity scoring expects finalized music; exclude padding with valid_mask")
            try:
                known = token in self.tokenizer.tok_to_id
            except TypeError:
                known = False
            if not known:
                raise ValueError(f"Unknown continuity token: {token!r}")
            ids.append(self.tokenizer.tok_to_id[token])
            has_note |= isinstance(token, tuple) and bool(token) and token[0] == "note"
        if not ids or not has_note:
            raise ValueError("Continuity scoring needs note evidence, not empty/terminal-only output")
        return ids

    @torch.inference_mode()
    def _score_ids(self, previous, current):
        if not 1 <= len(previous) <= self.model.config.max_length or not 1 <= len(current) <= self.model.config.max_length:
            raise ValueError("Excerpt exceeds continuity encoder limit; use score_boundary or explicit windows")
        a = torch.tensor(previous, device=self.device, dtype=torch.long)[None]
        b = torch.tensor(current, device=self.device, dtype=torch.long)[None]
        first, second = self.model(a, None, b, None)
        return float((first * second).sum(-1).clamp(-1, 1)[0])

    def score(self, previous, current, *, previous_mask=None, current_mask=None):
        """Cosine in [-1,1]. Not calibrated; a negative value is possible, not guaranteed."""
        return self._score_ids(self._ids(previous, previous_mask), self._ids(current, current_mask))

    def score_boundary(self, history, canvas, *, window_size=256, history_mask=None, canvas_mask=None):
        """Compare the clean history's tail to the generated canvas's head."""
        if not isinstance(window_size, int) or isinstance(window_size, bool) or not 1 <= window_size <= self.model.config.max_length:
            raise ValueError("Invalid boundary window size")
        previous = self._ids(history, history_mask)[-window_size:]
        current = self._ids(canvas, canvas_mask)[:window_size]
        # Ensure the actual selected windows, not just the full inputs, contain notes.
        return self.score(previous, current)
