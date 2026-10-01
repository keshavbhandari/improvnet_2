"""Labeled byte-offset examples; deterministic piece folds and variable crops."""

import json
import hashlib
from collections import Counter, defaultdict
import math
import os
from pathlib import Path
import random
import sqlite3

import torch
from torch.utils.data import Dataset, Sampler

from improvnet.tokenizer.absolute import AbsTokenizer
from improvnet.metrics.genre_classifier.audit import source_signature
from improvnet.metrics.genre_classifier.tokenization import classifier_tokens, transpose_melodic_notes, vocabulary_digest


def load_manifest(path, *, verify_sources=True):
    if not Path(path).is_file():
        prefix = str(Path(path).with_suffix(""))
        raise FileNotFoundError(
            f"Genre manifest not found: {path}. Build it before training with "
            f"python -m improvnet.metrics.genre_classifier.audit --output-prefix {prefix!r} "
            "(use --jsonl for custom dataset paths)."
        )
    with open(path) as handle:
        report = json.load(handle)
    if report.get("schema_version") != 1 or not report.get("complete") or len(report.get("labels", [])) < 2:
        raise ValueError("Need a complete genre audit with at least two labels")
    if report["labels"] != sorted(set(report["labels"])):
        raise ValueError("Manifest labels must be unique and sorted")
    if "other" in report["labels"]:
        raise ValueError("Known genres only: rebuild the obsolete other-label manifest")
    if verify_sources:
        for source in report["sources"]:
            actual = source_signature(source["path"])
            if any(actual[key] != source[key] for key in actual):
                raise ValueError(f"Stale genre manifest: {source['path']} changed")
            if "index" in source and source_signature(source["index"]["path"]) != source["index"]:
                raise ValueError(f"Stale genre byte index: {source['index']['path']} changed")
    if not Path(report["database"]).is_file():
        raise FileNotFoundError(report["database"])
    return report


class GenreDataset(Dataset):
    def __init__(self, manifest, split, *, tokenizer=None, min_length=128, max_length=4096,
                 eval_length=256, seed=42, class_caps=None, pitch_augmentation=True,
                 max_semitone_shift=7):
        if split not in ("train", "validation", "test") or not 1 <= min_length <= max_length or not 1 <= eval_length <= max_length:
            raise ValueError("Invalid split or crop lengths")
        self.report = load_manifest(manifest)
        if not isinstance(pitch_augmentation, bool) or not isinstance(max_semitone_shift, int) or isinstance(max_semitone_shift, bool) or max_semitone_shift < 0:
            raise ValueError("Invalid pitch augmentation settings")
        self.pitch_augmentation, self.max_semitone_shift = pitch_augmentation, max_semitone_shift
        self.labels = self.report["labels"]
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}
        uri = Path(self.report["database"]).as_uri() + "?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as connection:
            self.rows = connection.execute("SELECT source, offset, genre FROM examples WHERE split=? ORDER BY source, offset", (split,)).fetchall()
        if not self.rows:
            raise ValueError(f"No {split} examples in manifest")
        if class_caps:
            if split != "train":
                raise ValueError("Class pool caps must not rebalance held-out data")
            if any(not isinstance(cap, int) or isinstance(cap, bool) or cap < 1 for cap in class_caps.values()):
                raise ValueError("Class pool caps must be positive integers")
            grouped = defaultdict(list)
            for row in self.rows:
                grouped[row[2]].append(row)
            selected = []
            for label, rows in sorted(grouped.items()):
                cap = class_caps.get(label, len(rows))
                selected.extend(random.Random(f"{seed}:pool:{label}").sample(rows, cap) if len(rows) > cap else rows)
            self.rows = sorted(selected)
        self.class_counts = Counter(row[2] for row in self.rows)
        fingerprint = hashlib.sha256(json.dumps(self.report, sort_keys=True).encode())
        for source, offset, label in self.rows:
            fingerprint.update(f"\n{source}:{offset}:{label}".encode())
        self.data_fingerprint = fingerprint.hexdigest()
        self.tokenizer = tokenizer or AbsTokenizer()
        self.vocab_digest = vocabulary_digest(self.tokenizer)
        self.split, self.seed, self.epoch = split, seed, 0
        self.max_length, self.eval_length = max_length, eval_length
        lengths = {min_length, max_length}
        length = 1
        while length <= max_length:
            if length >= min_length:
                lengths.add(length)
            length *= 2
        self.crop_lengths = sorted(lengths)
        self.handles, self.handle_pid = {}, os.getpid()

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def __getstate__(self):
        return {**self.__dict__, "handles": {}}

    def __getitem__(self, index):
        # Replacement sampling should produce fresh excerpts of rare pieces,
        # not the same crop twenty times per epoch. The draw ID is global and
        # deterministic across ranks/workers and mid-epoch resumes.
        draw = 0
        if isinstance(index, tuple):
            index, draw = index
        if self.handle_pid != os.getpid():
            self.close()
            self.handle_pid = os.getpid()
        source, offset, genre = self.rows[index]
        path = self.report["sources"][source]["path"]
        if source not in self.handles:
            self.handles[source] = open(path, "rb")
        handle = self.handles[source]
        handle.seek(offset)
        record = json.loads(handle.readline())
        try:
            tokens = classifier_tokens(record["tokens"], self.tokenizer)
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Invalid labeled music at {path}:{offset}: {error}") from error
        rng = random.Random(self.seed + 1000003 * (self.epoch if self.split == "train" else 0) + 9176 * index + 7000003 * draw)
        desired = rng.choice(self.crop_lengths) if self.split == "train" else self.eval_length
        length = min(desired, len(tokens))
        # Anchor on an actual note; all-rest crops have no useful genre evidence.
        note_indices = [i for i, token in enumerate(tokens) if isinstance(token, tuple) and token[0] == "note"]
        anchor = rng.choice(note_indices) if self.split == "train" else note_indices[len(note_indices) // 2]
        low, high = max(0, anchor - length + 1), min(anchor, len(tokens) - length)
        start = rng.randint(low, high) if self.split == "train" else (low + high) // 2
        if self.split == "train" and self.pitch_augmentation:
            # Before slicing, so even a crop beginning on a drum note retains
            # its instrument role. One shift preserves melodic intervals.
            tokens = transpose_melodic_notes(tokens, rng, self.max_semitone_shift)
        ids = torch.tensor([self.tokenizer.tok_to_id[token] for token in tokens[start:start + length]], dtype=torch.long)
        return {"ids": ids, "label": self.label_to_id[genre]}


class GenreCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, rows):
        if not rows or any(row["ids"].ndim != 1 or not len(row["ids"]) for row in rows):
            raise ValueError("Need nonempty 1-D classifier examples")
        lengths = torch.tensor([len(row["ids"]) for row in rows])
        ids = torch.full((len(rows), int(lengths.max())), self.pad_id, dtype=torch.long)
        for index, row in enumerate(rows):
            ids[index, :len(row["ids"])] = row["ids"]
        return {"ids": ids, "attention_mask": torch.arange(ids.shape[1])[None, :] < lengths[:, None],
                "labels": torch.tensor([row["label"] for row in rows])}


class BalancedGenreSampler(Sampler):
    """Deterministic global class-balanced draws, sharded without DDP padding.

    Explicit class probabilities avoid reading millions of unknown records or
    majority-class dominance. Rare blues examples are oversampled; seeded
    (epoch,index) crops still vary over epochs. Each rank gets equal work.
    """
    def __init__(self, dataset, num_samples, *, focus_genres=("jazz", "blues"),
                 focus_weight=4., seed=42, rank=0, world_size=1):
        if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size < 1:
            raise ValueError("world_size must be a positive integer")
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples < world_size or not isinstance(rank, int) or not 0 <= rank < world_size:
            raise ValueError("Invalid distributed sample budget/rank")
        if not math.isfinite(focus_weight) or focus_weight <= 0:
            raise ValueError("Focus sampling weight must be finite and positive")
        if any(dataset.class_counts.get(label, 0) == 0 for label in dataset.labels):
            raise ValueError("Every audited genre needs positive training support")
        self.dataset, self.seed, self.epoch = dataset, seed, 0
        self.rank, self.world_size = rank, world_size
        self.num_samples = num_samples // world_size
        mass = torch.tensor([focus_weight if label in focus_genres else 1. for label in dataset.labels], dtype=torch.double)
        self.class_probabilities = mass / mass.sum()
        counts = torch.tensor([dataset.class_counts[label] for label in dataset.labels], dtype=torch.double)
        weights = (self.class_probabilities / counts).tolist()
        self.weights = torch.tensor([weights[dataset.label_to_id[row[2]]] for row in dataset.rows], dtype=torch.double)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        draws = torch.multinomial(self.weights, self.num_samples * self.world_size, replacement=True, generator=generator)
        return iter((index, draw) for index, draw in zip(draws[self.rank::self.world_size].tolist(), range(self.rank, len(draws), self.world_size)))
