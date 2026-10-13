"""AR tokens, whole-note elasticity slots and abstract rhythm conditions."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from improvnet.utils.ar_utils import ProcessData, load_split_byte_index
from improvnet.metrics.genre_classifier.audit import read_metadata

NOTE_ATTRIBUTES = 4
RHYTHM_DIM = 14


def note_units(tokens):
    """Preserve structure; reject incomplete notes rather than silently dropping them."""
    result, i = [], 0
    while i < len(tokens):
        token = tokens[i]
        if isinstance(token, tuple) and token[0] == "instrument":
            unit = tokens[i:i + NOTE_ATTRIBUTES]
            if len(unit) != NOTE_ATTRIBUTES or [t[0] if isinstance(t, tuple) else None for t in unit] != ["instrument", "note", "onset", "dur"]:
                raise ValueError(f"Incomplete/malformed AR note at token {i}")
            result.append(list(unit)); i += NOTE_ATTRIBUTES
        elif token in ("<PED_ON>", "<PED_OFF>"):
            if i + 1 >= len(tokens) or not isinstance(tokens[i + 1], tuple) or tokens[i + 1][0] != "onset":
                raise ValueError("Malformed pedal event")
            result.append(list(tokens[i:i + 2])); i += 2
        elif token in ("<MASK>", "<BLANK>", "<P>"):
            raise ValueError("Source tokens contain diffusion/padding tokens")
        else:
            result.append([token]); i += 1
    return result


def is_note(unit):
    return len(unit) == NOTE_ATTRIBUTES and isinstance(unit[0], tuple) and unit[0][0] == "instrument"


def rhythm_features(tokens, tokenizer):
    """14 pooled features: two six-bin histograms, chord and long-note shares.

    All times are divided by median positive inter-onset spacing. No pitch,
    velocity, instrument, note count, absolute tempo or ordered template is sent.
    Fractions are rounded to two decimals to avoid exact sample fingerprints.
    """
    onsets, durations, origin = [], [], 0
    for unit in note_units(tokens):
        if unit == ["<T>"]:
            origin += tokenizer.abs_time_step_ms
        elif is_note(unit):
            onsets.append(origin + unit[2][1]); durations.append(unit[3][1])
    if not onsets:
        return torch.zeros(RHYTHM_DIM)
    unique, counts = np.unique(onsets, return_counts=True)
    intervals = np.diff(unique)
    scale = float(np.median(intervals)) if len(intervals) else float(np.median(durations))
    scale = max(float(tokenizer.time_step_ms), scale)
    def histogram(values):
        if not len(values):
            return np.zeros(6)
        counts = np.bincount(np.searchsorted([.25, .5, 1., 2., 4.], np.asarray(values) / scale, side="right"), minlength=6)
        return counts / counts.sum()
    features = np.concatenate((histogram(intervals), histogram(durations),
        [counts[counts > 1].sum() / len(onsets), np.mean(np.asarray(durations) > scale)]))
    return torch.tensor(np.round(features, 2), dtype=torch.float32)


def blank_budget(units, elasticity):
    if not math.isfinite(elasticity) or not 0 <= elasticity <= 1:
        raise ValueError("Elasticity must lie in [0,1]")
    return math.floor(elasticity * sum(is_note(u) for u in units))


def split_windows(units, seq_len, elasticity):
    """Bounded windows end at complete event boundaries, reserving blank capacity."""
    windows, current, size, notes = [], [], 0, 0
    for unit in units:
        next_notes = notes + int(is_note(unit))
        if size + len(unit) + NOTE_ATTRIBUTES * math.floor(elasticity * next_notes) > seq_len:
            if not current:
                raise ValueError("SEQ_LEN is too small for a complete event and elasticity slots")
            windows.append(current); current, size, notes = [], 0, 0
            next_notes = int(is_note(unit))
        if len(unit) + NOTE_ATTRIBUTES * math.floor(elasticity * next_notes) > seq_len:
            raise ValueError("SEQ_LEN is too small")
        current.append(unit); size += len(unit); notes = next_notes
    if current:
        windows.append(current)
    return windows


def expand_slots(units, elasticity, rng):
    """Insert consecutive four-blank targets only at legal note boundaries."""
    count = blank_budget(units, elasticity)
    gaps = [i for i, unit in enumerate(units) if is_note(unit)]
    if gaps:
        gaps.append(gaps[-1] + 1)
    insertions = {}
    for _ in range(count):
        index = rng.choice(gaps)
        insertions[index] = insertions.get(index, 0) + 1
    tokens, roles, slots, added = [], [], [], []
    for i in range(len(units) + 1):
        for _ in range(insertions.get(i, 0)):
            slots.append(len(tokens)); added.append(len(tokens))
            tokens.extend(["<BLANK>"] * NOTE_ATTRIBUTES); roles.extend(range(1, NOTE_ATTRIBUTES + 1))
        if i < len(units):
            unit = units[i]
            if is_note(unit):
                slots.append(len(tokens)); roles.extend(range(1, NOTE_ATTRIBUTES + 1))
            else:
                roles.extend([0] * len(unit))
            tokens.extend(unit)
    return tokens, roles, slots, added


def _genre_weights(path, offsets, processor, config):
    """One-time exact metadata scan, cached with source/split/weight identity."""
    stat = os.stat(path)
    signature = (stat.st_size, stat.st_mtime_ns, tuple(offsets), config.JAZZ_BLUES_WEIGHT)
    key = hashlib.sha256(repr(signature).encode()).hexdigest()[:16]
    cache = Path(path).with_suffix(f".masked-weights-{key}.pkl")
    with open(str(cache) + ".lock", "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cache.exists():
            with cache.open("rb") as handle:
                return pickle.load(handle)
        weights = []
        with open(path, "rb") as handle:
            for offset in offsets:
                # Headers precede tokens in the AR JSONL format. Avoid parsing
                # hundreds of GB of note arrays merely to obtain genre labels.
                entry = read_metadata(handle.fileno(), offset)
                genre = processor.get_genre_id(entry.get("genre", "unknown"))
                weights.append(config.JAZZ_BLUES_WEIGHT if processor.genres[genre] in ("jazz", "blues") else 1.)
        pending = Path(str(cache) + f".tmp.{os.getpid()}")
        try:
            with pending.open("wb") as handle:
                pickle.dump(weights, handle)
            os.replace(pending, cache)
        finally:
            pending.unlink(missing_ok=True)
        return weights


class MaskedDiffusionDataset(Dataset):
    def __init__(self, config, split):
        self.config = SimpleNamespace(**{k: v for k, v in vars(config).items() if k.isupper()})
        self.split, self.epoch = split, 0
        self.processor, self.indices, weights = ProcessData(), [], []
        self.handles, self.pid = {}, os.getpid()
        for path in config.JSONL_FILES:
            offsets = load_split_byte_index(path, split, config=config)
            self.indices.extend((path, offset) for offset in offsets)
            if split == "train":
                weights.extend(_genre_weights(path, offsets, self.processor, config))
        if not self.indices:
            raise ValueError(f"No {split} examples")
        self.weights = torch.tensor(weights, dtype=torch.double) if weights else None

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getstate__(self):
        return {**self.__dict__, "handles": {}}

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def __del__(self):
        if hasattr(self, "handles"):
            self.close()

    def __getitem__(self, index):
        draw = 0
        if isinstance(index, tuple):
            index, draw = index
        if self.pid != os.getpid():
            self.close()
            self.handles, self.pid = {}, os.getpid()
        path, offset = self.indices[index]
        if path not in self.handles:
            self.handles[path] = open(path, "rb")
        handle = self.handles[path]; handle.seek(offset)
        entry = json.loads(handle.readline())
        c = self.config
        rng = random.Random(c.SEED + 1000003 * (self.epoch if self.split == "train" else 0) + index + 7000003 * draw)
        tokens = self.processor.serialized_tokens_to_tokens(entry.get("tokens", []))
        if self.split == "train":
            shift, current = rng.randint(-7, 7), None
            augmented = []
            for token in tokens:
                if isinstance(token, tuple) and token[0] == "instrument":
                    current = token[1]
                if isinstance(token, tuple) and token[0] == "note" and current != "drum":
                    token = ("note", max(0, min(127, token[1] + shift)), token[2])
                augmented.append(token)
            tokens = augmented
        elasticity = rng.uniform(0, c.ELASTICITY_MAX)
        units = note_units(tokens)
        # Choose a complete note boundary, then form one bounded contiguous excerpt.
        starts = [i for i, u in enumerate(units) if is_note(u)]
        if not starts:
            raise ValueError(f"No complete notes at {path}:{offset}")
        start = rng.choice(starts) if self.split == "train" and len(tokens) > c.SEQ_LEN else 0
        units = split_windows(units[start:], c.SEQ_LEN, elasticity)[0]
        raw = [token for unit in units for token in unit]
        rhythm = rhythm_features(raw, self.processor.tokenizer)
        if self.split == "train" and rng.random() < c.RHYTHM_DROP_PROB:
            rhythm.zero_()
        expanded, roles, _, _ = expand_slots(units, elasticity, rng)
        return dict(target=self.processor.tokens_to_tensor(expanded), roles=torch.tensor(roles),
            genre=torch.tensor(self.processor.get_genre_id(entry.get("genre", "unknown"))),
            multi_hot=self.processor.get_instrument_multihot(raw), rhythm=rhythm,
            elasticity=torch.tensor(elasticity, dtype=torch.float32))


def collate_sequences(examples):
    maximum = max(len(example["target"]) for example in examples)
    target = torch.full((len(examples), maximum), 2, dtype=torch.long)
    roles = torch.zeros_like(target); valid = torch.zeros_like(target, dtype=torch.bool)
    for i, example in enumerate(examples):
        length = len(example["target"])
        target[i, :length] = example["target"]; roles[i, :length] = example["roles"]; valid[i, :length] = True
    return dict(target=target, roles=roles, valid=valid,
                **{k: torch.stack([e[k] for e in examples]) for k in ("genre", "multi_hot", "rhythm", "elasticity")})


class GenreSampler(Sampler):
    """A shared weighted draw is sharded across ranks, including deterministic resume."""
    def __init__(self, weights, batch_size, seed, rank=0, world=1):
        self.weights, self.seed, self.rank, self.world, self.epoch = weights, seed, rank, world, 0
        self.size = len(weights) // (batch_size * world) * batch_size
        self.start = 0
        if self.size < 1:
            raise ValueError("Not enough examples for a distributed batch")

    def __len__(self):
        return self.size - self.start

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.size * self.world, replacement=True, generator=generator)
        draws = range(self.rank + self.start * self.world, self.size * self.world, self.world)
        return iter(zip(indices[self.rank::self.world][self.start:].tolist(), draws))
