"""Aligned multi-bin MIDI windows and bounded full/rhythm accompaniment conditioning."""
from collections import defaultdict
import json
import os
import random
import sqlite3
from pathlib import Path

import numpy as np
import torch
from mido import KeySignatureError

from improvnet.utils.local_vae_data import LocalVAEDataset, seed_for

MODES = {"none": 0, "full": 1, "rhythm": 2}


def genre_labels(index):
    """Freeze all supplied training genre tags, without inventing an 'other' genre."""
    with sqlite3.connect(Path(index).resolve().as_uri() + "?mode=ro", uri=True) as db:
        labels = {tag for (value,) in db.execute("SELECT DISTINCT genres FROM files WHERE split='train'")
                  for tag in json.loads(value)}
    return ("<unconditioned>", *sorted(labels))


def budget_events(rows, slots, starts, origin, frames, max_events):
    """Round-robin track/time buckets: dense chords cannot monopolize the window."""
    if max_events < 1:
        raise ValueError("Event budget must be positive")
    if len(rows) <= max_events:
        return np.arange(len(rows))
    groups = defaultdict(list)
    for i, (slot, start) in enumerate(zip(slots, starts)):
        groups[int(slot), max(0, int((start - origin) // frames))].append(i)
    queues = list(groups.values())
    # Spread within each bucket too, rather than always preferring lower pitches.
    for q in queues:
        q[:] = [q[i] for i in np.argsort(np.minimum(np.arange(len(q)), len(q) - 1 - np.arange(len(q))), kind="stable")]
    chosen, depth = [], 0
    while len(chosen) < max_events:
        for q in queues:
            if depth < len(q):
                chosen.append(q[depth])
                if len(chosen) == max_events:
                    break
        depth += 1
    return np.sort(chosen)


def accompaniment(events, *, origin_bin, bins, frames, valid_mask, mode="full", max_tracks=3,
                  max_events=256, rng=None):
    """At most three instrument parts (track/channel/program), not merely MIDI containers.

    Event positions remain signed: carried notes have a negative start. Rhythm
    curves use ALL selected notes before event pruning, and do not contain pitch.
    """
    if mode not in MODES or not 1 <= max_tracks <= 3:
        raise ValueError("Unknown mode or accompaniment track limit outside [1,3]")
    if valid_mask.shape != (bins, frames):
        raise ValueError("Invalid accompaniment clock mask")
    starts, ends = events[:, 6] * frames + events[:, 7], events[:, 8] * frames + events[:, 9]
    origin, finish = origin_bin * frames, (origin_bin + bins) * frames
    ids = np.flatnonzero((starts < finish) & (ends > origin))
    parts = defaultdict(list)
    for i in ids:
        parts[tuple(map(int, events[i, :3]))].append(int(i))
    keys = sorted(parts, key=lambda k: (-len(parts[k]), k))
    if rng is not None:
        rng.shuffle(keys)
    keys = keys[:max_tracks] if mode != "none" else []
    selected = sorted((i for key in keys for i in parts[key]), key=lambda i: (starts[i], int(events[i, 10])))
    rows = events[selected]
    slot_for = {key: i for i, key in enumerate(keys)}
    slots = np.array([slot_for[tuple(map(int, r[:3]))] for r in rows], dtype=np.int64)
    curves = np.zeros((max_tracks, bins * frames, 3), np.float32)
    for row, slot, start, end in zip(rows, slots, starts[selected], ends[selected]):
        local_start, local_end = int(start - origin), int(end - origin)
        curves[slot, max(0, local_start):min(bins * frames, local_end), 1] += 1
        if 0 <= local_start < bins * frames:
            curves[slot, local_start, 0] += 1
            curves[slot, local_start, 2] += row[5] / 127
    # Attack velocity = mean velocity at that frame, not a pitch-dependent feature.
    curves[..., 2] /= np.maximum(curves[..., 0], 1)
    curves[..., :2] = np.log1p(curves[..., :2])
    curves *= valid_mask.reshape(1, bins * frames, 1)
    programs, drums, track_mask = np.zeros(max_tracks, np.int64), np.zeros(max_tracks, np.int64), np.zeros(max_tracks, bool)
    for slot, key in enumerate(keys):
        programs[slot], drums[slot], track_mask[slot] = key[2], int(key[1] == 9), True
    keep = budget_events(rows, slots, starts[selected], origin, frames, max_events)
    # Full mode has no rhythm inputs; rhythm mode has no pitch-bearing event inputs.
    features = np.stack((slots, rows[:, 2], rows[:, 3], rows[:, 4], rows[:, 5]), axis=1) if len(rows) else np.zeros((0, 5), np.int64)
    positions = np.stack((starts[selected] - origin, ends[selected] - origin), axis=1).astype(np.float32)
    return dict(events=torch.from_numpy(features[keep] if mode == "full" else np.zeros((0, 5), np.int64)),
                event_positions=torch.from_numpy(positions[keep] if mode == "full" else np.zeros((0, 2), np.float32)),
                rhythm=torch.from_numpy(curves.reshape(max_tracks, bins, frames, 3) if mode == "rhythm" else np.zeros((max_tracks, bins, frames, 3), np.float32)),
                track_mask=torch.from_numpy(track_mask), accomp_programs=torch.from_numpy(programs),
                accomp_drums=torch.from_numpy(drums), mode=torch.tensor(MODES[mode]),
                accompaniment_info=dict(parts=[list(k) for k in keys], available_events=len(rows),
                    retained_events=len(keep) if mode == "full" else 0,
                    dropped_events=len(rows) - len(keep) if mode == "full" else 0))


def window_tensors(piece, start, config, *, history_count=None, mode="full", events=None, rng=None):
    """Fixed-size window: left-padded optional history, nonempty target, trailing padding."""
    h, t, f = config.HISTORY_BINS, config.TARGET_BINS, piece.config.frames_per_bin
    if not 0 <= start < len(piece.velocity):
        raise ValueError("Target start is outside the source canvas")
    history_count = min(h, start) if history_count is None else history_count
    if not 0 <= history_count <= min(h, start):
        raise ValueError("Invalid clean history length")
    length = h + t
    item = dict(roll=torch.zeros(length, 2, 128, f), pedal=torch.zeros(length, 1, f),
                valid_mask=torch.zeros(length, f, dtype=torch.bool), known=torch.zeros(length, dtype=torch.bool),
                pitch=torch.zeros(length, 128), pitch_present=torch.zeros(length, dtype=torch.bool),
                genre=torch.tensor(0), program=torch.tensor(piece.metadata["target_program"]),
                is_drum=torch.tensor(bool(piece.metadata.get("target_is_drum"))),
                pedal_enabled=torch.tensor(bool(piece.metadata["pedal_enabled"])))
    for src in range(start - history_count, min(start + t, len(piece.velocity))):
        dst = src - start + h
        item["roll"][dst] = torch.from_numpy(np.stack((piece.velocity[src].astype(np.float32) / 127, piece.onset[src])))
        item["pedal"][dst, 0] = torch.from_numpy(piece.pedal[src].astype(np.float32) / 127)
        item["valid_mask"][dst] = torch.from_numpy(piece.valid_mask[src].copy())
        item["known"][dst] = src < start
        item["pitch"][dst] = torch.from_numpy(piece.velocity[src].any(-1).astype(np.float32))
        item["pitch_present"][dst] = True
    item.update(accompaniment(piece.events if events is None else events, origin_bin=start - h,
        bins=length, frames=f, valid_mask=item["valid_mask"].numpy(), mode=mode,
        max_tracks=config.MAX_ACCOMPANIMENT_TRACKS, max_events=config.MAX_ACCOMPANIMENT_EVENTS, rng=rng))
    return item


def transpose_window(item, shift):
    """One consistent shift for all target/history bins and pitched accompaniment."""
    if not shift:
        return item
    for name, dim in (("roll", 2), ("pitch", 1)):
        source = item[name]
        output = torch.zeros_like(source)
        left, right = [slice(None)] * source.ndim, [slice(None)] * source.ndim
        left[dim], right[dim] = (slice(None, -shift), slice(shift, None)) if shift > 0 else (slice(-shift, None), slice(None, shift))
        output[tuple(right)] = source[tuple(left)]
        item[name] = output
    pitched = item["events"][:, 2] == 0
    item["events"][pitched, 3] += shift
    return item


def collate_windows(items):
    batch = {key: torch.stack([item[key] for item in items]) for key, value in items[0].items()
             if isinstance(value, torch.Tensor) and key not in ("events", "event_positions")}
    # One masked dummy token keeps all-empty batches well defined.
    length = max(1, max(len(item["events"]) for item in items))
    batch.update(events=torch.zeros(len(items), length, 5, dtype=torch.long),
                 event_positions=torch.zeros(len(items), length, 2),
                 event_mask=torch.zeros(len(items), length, dtype=torch.bool))
    for i, item in enumerate(items):
        n = len(item["events"])
        batch["events"][i, :n], batch["event_positions"][i, :n], batch["event_mask"][i, :n] = item["events"], item["event_positions"], True
    batch["metadata"] = [item.get("metadata", {}) for item in items]
    batch["accompaniment_info"] = [item["accompaniment_info"] for item in items]
    return batch


class LatentDiffusionDataset(LocalVAEDataset):
    """Reuse the MIDI index/splits and fixed drum subset; sample windows rather than bins."""
    def __init__(self, config, split="train", *, labels=None, error_dir=None):
        super().__init__(config, split, error_dir=error_dir)
        self.labels = tuple(labels or genre_labels(config.INDEX_PATH))
        self.genre_ids = {label: i for i, label in enumerate(self.labels)}

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        epoch = self.epoch if self.split == "train" else 0
        rng = random.Random(seed_for(self.config.SEED, self.split, epoch, index, "stage2"))
        c = self.config
        for attempt in range(c.MAX_DATA_RETRIES):
            collection = rng.choices(self.names, self.weights)[0]
            ids = self.groups[collection]
            row_id = int(ids[rng.randrange(len(ids))])
            path, tags = self.connect().execute("SELECT path,genres FROM files WHERE id=?", (row_id,)).fetchone()
            target_seed = seed_for(c.SEED, epoch, row_id)
            key = row_id, target_seed
            try:
                if key in self.cache:
                    piece, size = self.cache.pop(key)
                    self.cache[key] = piece, size
                else:
                    piece = self.tokenizer.encode(path, seed=target_seed, recover=True)
                    if not piece.metadata["target_present"]:
                        raise ValueError("No valid target notes")
                    size = sum(getattr(piece, k).nbytes for k in ("velocity", "onset", "pedal", "valid_mask", "events"))
                    budget = int(c.CACHE_MB_PER_WORKER * 1024**2)
                    while self.cache and self.cache_bytes + size > budget:
                        _, (_, removed) = self.cache.popitem(last=False)
                        self.cache_bytes -= removed
                    if size <= budget:
                        self.cache[key] = piece, size
                        self.cache_bytes += size
                active = np.flatnonzero(piece.velocity.any(axis=(1, 2)))
                if not len(active):
                    raise ValueError("No active target bins")
                if rng.random() < c.PROMPTLESS_PROB:
                    start = 0  # Actual beginnings, not only random windows without context.
                elif rng.random() < c.ACTIVE_BIN_PROB:
                    anchor = int(rng.choice(active))
                    start = max(0, anchor - rng.randrange(c.TARGET_BINS))
                else:
                    start = rng.randrange(len(piece.velocity))
                modes = list(c.ACCOMPANIMENT_MODE_PROBS)
                mode = rng.choices(modes, [c.ACCOMPANIMENT_MODE_PROBS[m] for m in modes])[0]
                item = window_tensors(piece, start, c, mode=mode, rng=rng)
                choices = [self.genre_ids[g] for g in json.loads(tags) if g in self.genre_ids]
                item["genre"] = torch.tensor(rng.choice(choices) if choices and rng.random() >= c.GENRE_DROP_PROB else 0)
                # Absent reference is explicit and cannot masquerade as a silent bin.
                if rng.random() < c.PITCH_DROP_PROB:
                    item["pitch"].zero_(); item["pitch_present"].zero_()
                elif self.split == "train" and c.PITCH_NOTE_DROP_PROB:
                    generator = torch.Generator().manual_seed(seed_for(c.SEED, epoch, index, attempt, "pitch-guide") % 2**63)
                    item["pitch"] *= torch.rand(item["pitch"].shape, generator=generator) >= c.PITCH_NOTE_DROP_PROB
                if self.split == "train" and c.PITCH_AUGMENTATION and not bool(item["is_drum"]):
                    pitches = (item["roll"][:, 0].any((0, 2))).nonzero().flatten()
                    # Include every selected accompaniment note, even notes omitted by the token budget.
                    selected = {tuple(p) for p in item["accompaniment_info"]["parts"]}
                    extra = [int(r[4]) for r in piece.events if tuple(map(int, r[:3])) in selected and not r[3]]
                    pitches = torch.cat((pitches, torch.tensor(extra, dtype=torch.long)))
                    low, high = c.PITCH_SHIFT_RANGE
                    if len(pitches):
                        low, high = max(low, -int(pitches.min())), min(high, 127 - int(pitches.max()))
                    item = transpose_window(item, rng.randint(low, high))
                item["retries"] = torch.tensor(attempt)
                item["metadata"] = dict(path=path, collection=collection, start_bin=start)
                return item
            except (OSError, EOFError, ValueError, IndexError, KeyError, KeySignatureError) as error:
                if c.LOG_DATA_ERRORS and self.error_dir and path not in self.logged:
                    self.error_dir.mkdir(parents=True, exist_ok=True)
                    with (self.error_dir / f"data_errors.{os.getpid()}.jsonl").open("a") as handle:
                        handle.write(json.dumps(dict(path=path, error=str(error))) + "\n")
                    self.logged.add(path)
        raise RuntimeError(f"No usable {self.split} MIDI after {c.MAX_DATA_RETRIES} attempts; last={path}")
