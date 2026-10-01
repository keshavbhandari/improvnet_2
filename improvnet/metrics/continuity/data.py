"""Deterministic non-overlapping performance pairs, worker-safe byte reads."""

from bisect import bisect_left
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3

import torch
from torch.utils.data import Sampler

from improvnet.tokenizer.absolute import AbsTokenizer
from improvnet.metrics.genre_classifier.audit import source_signature
from improvnet.metrics.genre_classifier.tokenization import classifier_tokens, vocabulary_digest


def load_manifest(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing continuity manifest: {path}. Run python -m improvnet.metrics.continuity.audit --output-prefix {str(Path(path).with_suffix(''))!r}")
    with open(path) as handle:
        report = json.load(handle)
    if report.get("format") != "continuity_pairs_v1" or not report.get("complete") or report.get("schema_version") != 1:
        raise ValueError("Need a complete continuity manifest (not the genre manifest)")
    for source in report["sources"]:
        actual = source_signature(source["path"])
        if any(actual[key] != source[key] for key in actual) or source_signature(source["index"]["path"]) != source["index"]:
            raise ValueError("Stale continuity manifest: JSONL source or byte index changed")
    if not Path(report["database"]).is_file():
        raise FileNotFoundError(report["database"])
    return report


def sample_pair(tokens, rng, min_length, max_length, eval_length=None):
    """Return chronological, disjoint note-containing crops or None if unsuitable."""
    minimum = eval_length if eval_length is not None else min_length
    if len(tokens) < 2 * minimum:
        return None
    notes = [i for i, token in enumerate(tokens) if isinstance(token, tuple) and token[0] == "note"]
    if len(notes) < 2:
        return None
    a_length = eval_length or rng.randint(min_length, min(max_length, len(tokens) - min_length))
    b_length = eval_length or rng.randint(min_length, min(max_length, len(tokens) - a_length))
    # Note-centered anchors avoid terminal-only/all-rest positives. Disjoint
    # ranges eliminate trivial overlapping-token matches. Validation has a
    # fixed RNG and exactly eval_length real tokens in each retained excerpt.
    for _ in range(32):
        anchor = rng.choice(notes)
        low, high = max(0, anchor - a_length + 1), min(anchor, len(tokens) - a_length - b_length)
        if low > high:
            continue
        a_start = rng.randint(low, high)
        candidates = notes[bisect_left(notes, a_start + a_length):]
        if not candidates:
            continue
        anchor_b = rng.choice(candidates)
        low_b, high_b = max(a_start + a_length, anchor_b - b_length + 1), min(anchor_b, len(tokens) - b_length)
        if low_b <= high_b:
            b_start = rng.randint(low_b, high_b)
            return (a_start, a_start + a_length), (b_start, b_start + b_length)
    return None


def augment_excerpt(tokens, start, end, rng, max_shift=7):
    """Transpose one view independently, respecting its pitches and drum roles.

    Resolve instruments before the crop, but bound the shift using only this
    excerpt so a distant extreme note cannot prevent its augmentation.
    """
    result, melodic, instrument = list(tokens[start:end]), [], None
    for index, token in enumerate(tokens[:end]):
        if isinstance(token, tuple) and token[0] == "instrument":
            instrument = token[1]
        elif start <= index and isinstance(token, tuple) and token[0] == "note":
            percussion = instrument == "drum" or "Drum" in str(instrument) or "Percuss" in str(instrument)
            if instrument is not None and not percussion:
                melodic.append((index - start, token))
    if melodic:
        pitches = [token[1] for _, token in melodic]
        shift = rng.randint(max(-max_shift, -min(pitches)), min(max_shift, 127 - max(pitches)))
        for index, token in melodic:
            result[index] = ("note", token[1] + shift, token[2])
    return result


class ContinuityDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, split, *, min_length=128, max_length=512, eval_length=256,
                 seed=42, pitch_augmentation=True, max_semitone_shift=7, tokenizer=None):
        if split not in ("train", "validation", "test") or not 1 <= min_length <= max_length or not 1 <= eval_length <= max_length:
            raise ValueError("Invalid continuity split/crop lengths")
        if not isinstance(pitch_augmentation, bool) or not isinstance(max_semitone_shift, int) or isinstance(max_semitone_shift, bool) or max_semitone_shift < 0:
            raise ValueError("Invalid augmentation settings")
        self.report = load_manifest(manifest)
        with sqlite3.connect(Path(self.report["database"]).as_uri() + "?mode=ro&immutable=1", uri=True) as connection:
            # Assign stable, collision-free integer identities within the fold.
            self.rows = connection.execute("SELECT source,offset,DENSE_RANK() OVER (ORDER BY piece)-1 FROM examples WHERE split=? ORDER BY source,offset", (split,)).fetchall()
        if len(self.rows) < 2 or len({row[2] for row in self.rows}) < 2:
            raise ValueError(f"Need at least two piece identities in {split}")
        self.fingerprint = hashlib.sha256(json.dumps(self.report, sort_keys=True).encode()).hexdigest()
        self.tokenizer = tokenizer or AbsTokenizer()
        self.vocab_digest = vocabulary_digest(self.tokenizer)
        self.split, self.seed, self.epoch = split, seed, 0
        self.min_length, self.max_length, self.eval_length = min_length, max_length, eval_length
        self.pitch_augmentation, self.max_semitone_shift = pitch_augmentation, max_semitone_shift
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
        draw = 0
        if isinstance(index, tuple):
            index, draw = index
        epoch = self.epoch if self.split == "train" else 0
        rng = random.Random(self.seed + 1000003 * epoch + 9176 * index + 7000003 * draw)
        if self.handle_pid != os.getpid():
            self.close()
            self.handle_pid = os.getpid()
        # The header-only index cannot know flattened token lengths. Retry
        # short/rest-only pieces deterministically within the SAME fold, never
        # overlap/pad a positive just to manufacture a pair.
        for attempt in range(64):
            actual_index = index if not attempt else rng.randrange(len(self.rows))
            source, offset, piece = self.rows[actual_index]
            path = self.report["sources"][source]["path"]
            if source not in self.handles:
                self.handles[source] = open(path, "rb")
            handle = self.handles[source]
            handle.seek(offset)
            record = json.loads(handle.readline())
            try:
                tokens = classifier_tokens(record["tokens"], self.tokenizer)
            except ValueError as error:
                if "contains no musical note" in str(error):
                    continue
                raise ValueError(f"Invalid continuity music at {path}:{offset}: {error}") from error
            ranges = sample_pair(tokens, rng, self.min_length, self.max_length,
                                 self.eval_length if self.split != "train" else None)
            if ranges is None:
                continue
            views = []
            for start, end in ranges:
                # Independent RNG draws for BOTH positive views. Resolve drum
                # roles on the full piece before cropping, not on isolated notes.
                excerpt = augment_excerpt(tokens, start, end, rng, self.max_semitone_shift) if self.split == "train" and self.pitch_augmentation else tokens[start:end]
                views.append(torch.tensor([self.tokenizer.tok_to_id[token] for token in excerpt], dtype=torch.long))
            return {"a": views[0], "b": views[1], "piece_id": piece,
                    "ranges": ranges, "record_index": actual_index}
        raise ValueError("No eligible note-containing disjoint pairs found after 64 attempts; check dataset/crop lengths")


class PairCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, rows):
        if not rows:
            raise ValueError("Need a nonempty pair batch")
        result = {"piece_ids": torch.tensor([row["piece_id"] for row in rows], dtype=torch.long)}
        for view in ("a", "b"):
            lengths = torch.tensor([len(row[view]) for row in rows])
            if bool((lengths < 1).any()):
                raise ValueError("Empty continuity excerpt")
            ids = torch.full((len(rows), int(lengths.max())), self.pad_id, dtype=torch.long)
            for i, row in enumerate(rows):
                ids[i, :len(row[view])] = row[view]
            result[view] = ids
            result[view + "_mask"] = torch.arange(ids.shape[1])[None] < lengths[:, None]
        return result


class PairBatchSampler(Sampler):
    """Equal DDP work; unique record draws per global batch, resumable mid-epoch.

    Same-piece records/retries can still duplicate an identity: loss masks these
    as negatives. Uniform draws include unknown genres without class balancing.
    """
    def __init__(self, size, samples, batch_size, *, seed=42, rank=0, world_size=1):
        if not 0 <= rank < world_size or batch_size < 2 or size < batch_size * world_size:
            raise ValueError("Need >=2 pairs/rank and enough records for one global batch")
        self.size, self.samples, self.batch_size = size, samples, batch_size
        self.seed, self.rank, self.world_size = seed, rank, world_size
        self.epoch, self.start_batch = 0, 0
        if not len(self):
            raise ValueError("Global sample budget too small")

    def __len__(self):
        return self.samples // (self.batch_size * self.world_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 1000003)
        global_size = self.batch_size * self.world_size
        for batch in range(len(self)):
            selected = rng.sample(range(self.size), global_size)
            if batch >= self.start_batch:
                low = self.rank * self.batch_size
                yield [(index, batch * global_size + low + i) for i, index in enumerate(selected[low:low + self.batch_size])]
