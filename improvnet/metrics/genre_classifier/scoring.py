"""Genre probabilities from a trained checkpoint, not an untrained reward."""

import argparse
import json
import math
from numbers import Integral

import torch

from improvnet.tokenizer.absolute import AbsTokenizer
from improvnet.tokenizer.midi import MidiDict
from .audit import normalize_genre
from .genre_classifier_model import GenreClassifier, GenreClassifierConfig
from .tokenization import vocabulary_digest


class GenreClassifierMetric:
    def __init__(self, model, tokenizer, labels, *, temperature=1., prior_correction=None):
        if any(normalize_genre(label) is None for label in labels):
            raise ValueError("Classifier supports known genres only; old other-label checkpoints are unsupported")
        if len(labels) != model.config.num_genres or len(set(labels)) != len(labels):
            raise ValueError("Checkpoint genre mapping does not match the model")
        if tokenizer.vocab_size != model.config.vocab_size or tokenizer.pad_id != model.config.pad_id:
            raise ValueError("Classifier/tokenizer vocabulary dimensions do not match")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be finite and positive")
        self.model, self.tokenizer, self.labels = model.eval(), tokenizer, list(labels)
        self.temperature = temperature
        self.device = next(model.parameters()).device
        self.correction = torch.zeros(len(labels), device=self.device) if prior_correction is None else torch.as_tensor(prior_correction, device=self.device)
        if self.correction.shape != (len(labels),) or not bool(torch.isfinite(self.correction).all()):
            raise ValueError("Invalid checkpoint class-prior correction")
        self.note_ids = {i for i, token in tokenizer.id_to_tok.items() if isinstance(token, tuple) and token[0] == "note"}

    @classmethod
    def from_checkpoint(cls, path, *, device="cpu", tokenizer=None):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("trained_steps", 0) < 1:
            raise ValueError("Genre rewards require a trained classifier checkpoint")
        if any(normalize_genre(label) is None for label in state.get("labels", [])):
            raise ValueError("Only known-genre checkpoints are supported; remove/retrain old other-label heads")
        tokenizer = tokenizer or AbsTokenizer()
        if state.get("vocabulary_digest") != vocabulary_digest(tokenizer):
            raise ValueError("Classifier tokenizer vocabulary differs from the checkpoint")
        model = GenreClassifier(GenreClassifierConfig(**state["config"]))
        model.load_state_dict(state["model_state_dict"], strict=True)
        return cls(model.to(device), tokenizer, state["labels"], temperature=state.get("temperature", 1.),
                   prior_correction=state.get("prior_correction"))

    def _prepare(self, sequence, valid_mask):
        if hasattr(sequence, "detach"):
            sequence = sequence.detach().cpu()
        if hasattr(sequence, "ndim") and sequence.ndim != 1:
            raise ValueError("Use a 1-D sequence for genre scoring")
        if hasattr(sequence, "tolist"):
            sequence = sequence.tolist()
        if isinstance(sequence, (str, bytes)):
            raise ValueError("Expected a token/ID sequence")
        try:
            values = list(sequence)
        except TypeError as error:
            raise ValueError("Expected a 1-D token/ID sequence") from error
        if valid_mask is None:
            mask = [True] * len(values)
        else:
            mask = valid_mask.detach().cpu().tolist() if hasattr(valid_mask, "detach") else list(valid_mask)
            if len(mask) != len(values) or any(x not in (0, 1, False, True) for x in mask):
                raise ValueError("valid_mask must match the sequence")
        ids = []
        for value, keep in zip(values, mask):
            if not keep:
                continue
            if isinstance(value, Integral) and not isinstance(value, bool):
                if int(value) not in self.tokenizer.id_to_tok:
                    raise ValueError("Unknown token ID in genre scoring")
                token = self.tokenizer.id_to_tok[int(value)]
            else:
                token = tuple(value) if isinstance(value, list) else value
            if isinstance(token, tuple):
                if any(isinstance(field, bool) for field in token):
                    raise ValueError("Boolean values are not musical fields")
                if token and token[0] in ("note", "onset", "dur") and any(not isinstance(field, Integral) for field in token[1:]):
                    raise ValueError("Pitch, velocity and timing fields must be integers")
            if isinstance(token, tuple) and token and token[0] == "prefix":
                continue
            if token in (self.tokenizer.pad_tok, self.tokenizer.mask_tok, self.tokenizer.blank_tok,
                         self.tokenizer.sep_tok, self.tokenizer.unk_tok):
                raise ValueError("Genre scoring expects finalized tokens; exclude padding with valid_mask")
            try:
                known = token in self.tokenizer.tok_to_id
            except TypeError:
                known = False
            if not known:
                raise ValueError(f"Unknown classifier token: {token!r}")
            ids.append(self.tokenizer.tok_to_id[token])
        if not any(i in self.note_ids for i in ids):
            raise ValueError("No note evidence for genre scoring (empty/terminal-only/all-rest input)")
        return ids

    @torch.inference_mode()
    def probabilities_batch(self, sequences, *, valid_masks=None, window_batch_size=16):
        if window_batch_size < 1:
            raise ValueError("window_batch_size must be positive")
        rows = list(sequences)
        if valid_masks is not None and len(valid_masks) != len(rows):
            raise ValueError("valid_masks must have one entry per sequence")
        if not rows:
            return []
        size = self.model.config.max_length
        windows = []
        for owner, sequence in enumerate(rows):
            ids = self._prepare(sequence, None if valid_masks is None else valid_masks[owner])
            for start in range(0, len(ids), size):
                window = ids[start:start + size]
                if any(i in self.note_ids for i in window):
                    windows.append((owner, window))
        windows.sort(key=lambda pair: len(pair[1]), reverse=True)
        totals = torch.zeros(len(rows), len(self.labels), device=self.device)
        counts = torch.zeros(len(rows), device=self.device)
        for start in range(0, len(windows), window_batch_size):
            selected = windows[start:start + window_batch_size]
            owners = torch.tensor([owner for owner, _ in selected], device=self.device)
            lengths = torch.tensor([len(row) for _, row in selected], device=self.device)
            batch = torch.full((len(selected), int(lengths.max())), self.tokenizer.pad_id, device=self.device, dtype=torch.long)
            for index, (_, row) in enumerate(selected):
                batch[index, :len(row)] = torch.tensor(row, device=self.device)
            mask = torch.arange(batch.shape[1], device=self.device)[None, :] < lengths[:, None]
            dtype = torch.bfloat16 if self.device.type != "cuda" or torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=self.device.type == "cuda"):
                logits = self.model(batch, mask).float() - self.correction
            probabilities = (logits / self.temperature).softmax(-1)
            totals.index_add_(0, owners, probabilities * lengths[:, None])
            counts.index_add_(0, owners, lengths.float())
        return [dict(zip(self.labels, row)) for row in (totals / counts[:, None]).cpu().tolist()]

    def probabilities(self, sequence, *, valid_mask=None, window_batch_size=16):
        return self.probabilities_batch([sequence], valid_masks=None if valid_mask is None else [valid_mask],
                                        window_batch_size=window_batch_size)[0]

    def score(self, sequence, *, target_genre="jazz", valid_mask=None):
        target_genre = normalize_genre(target_genre)
        if target_genre not in self.labels:
            raise ValueError(f"Unknown target genre; expected one of {self.labels}")
        return self.probabilities(sequence, valid_mask=valid_mask)[target_genre]

    def score_batch(self, sequences, *, target_genre="jazz", valid_masks=None, window_batch_size=16):
        target_genre = normalize_genre(target_genre)
        if target_genre not in self.labels:
            raise ValueError(f"Unknown target genre; expected one of {self.labels}")
        return [probabilities[target_genre] for probabilities in
                self.probabilities_batch(sequences, valid_masks=valid_masks, window_batch_size=window_batch_size)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="MIDI file to classify")
    parser.add_argument("--genre", default="jazz")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    metric = GenreClassifierMetric.from_checkpoint(args.checkpoint, device=args.device)
    tokens = metric.tokenizer.tokenize(MidiDict.from_midi(args.input))
    scores = metric.probabilities(tokens)
    target = normalize_genre(args.genre)
    if target not in scores:
        raise ValueError(f"Unknown target genre {args.genre!r}")
    print(json.dumps({"target_genre": target, "target_score": scores[target], "probabilities": scores}, indent=2))


if __name__ == "__main__":
    main()
