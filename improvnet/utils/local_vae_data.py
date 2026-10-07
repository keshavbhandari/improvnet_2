"""MIDI-only SQLite index, local-bin tensors and reproducible sampling.

Genre is metadata, not a VAE input. Accompaniment tensors retain the existing
typed event rows; a flattened event vocabulary/conditioner is deliberately later.
"""
import ast
from collections import Counter, OrderedDict
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import shutil
import tempfile

import numpy as np
import torch
from mido import KeySignatureError
from torch.utils.data import Dataset

from improvnet.tokenizer.hierarchical import EVENT_FIELDS, GridConfig, HierarchicalTokenizer
from improvnet.utils.local_vae_corruption import corrupt_roll_bin, target_note_rows

INDEX_FORMAT = "local_vae_midi_index_v1"


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256("\0".join(map(str, parts)).encode()).digest()[:8], "big")


def genres_from(value):
    if isinstance(value, dict):
        value = value.get("genre", value.get("genres", []))
    if isinstance(value, str):
        if value[:1] in ("[", "{"):
            try:
                return genres_from(ast.literal_eval(value))
            except (ValueError, SyntaxError):
                return []
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return sorted({v.strip().lower() for v in value if isinstance(v, str) and
                   v.strip().lower() not in ("", "unknown", "none", "nan")})


def split_for(group, seed, ratios):
    value = seed_for(seed, group) / 2**64
    threshold = 0
    for name in ("train", "validation", "test"):
        threshold += ratios[name]
        if value < threshold:
            return name
    return "test"


def index_settings(config):
    result = dict(datasets={k: str(Path(v).resolve()) for k, v in config.DATASETS.items()},
                split_seed=config.DATA_SPLIT_SEED, ratios=config.DATA_SPLIT_RATIOS,
                aria_metadata=config.ARIA_METADATA, gigamidi_metadata=config.GIGAMIDI_METADATA)
    if config.GIGAMIDI_METADATA:
        result["gigamidi_genre_fields"] = list(getattr(config, "GIGAMIDI_GENRE_FIELDS", ("music_styles_curated",)))
    return result


def load_gigamidi_genres(path, fields=("music_styles_curated",)):
    """Read supplied genre fields, not loop/instrument or noisy scraped style labels.

    Actual extended CSV has very large serialized loop fields. Restore the global
    CSV limit afterward. Empty labels remain unconditioned; duplicates union tags.
    Legacy genre-column CSVs remain supported when curated columns are absent.
    """
    result, counts, rows = {}, Counter(), 0
    previous_limit = csv.field_size_limit()
    csv.field_size_limit(max(previous_limit, 64 * 1024**2))
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            selected = tuple(fields)
            if not all(f in columns for f in selected):
                legacy = tuple(f for f in ("genre", "genres", "genres_scraped", "genres_discogs", "genres_tagtraum", "genres_lastfm") if f in columns)
                if selected == ("music_styles_curated",) and legacy:
                    selected = legacy
                else:
                    raise ValueError(f"Missing GigaMIDI genre fields: {set(selected)-set(columns)}")
            for row in reader:
                rows += 1
                key = row.get("md5") or row.get("MD5") or row.get("file_id") or row.get("file_path")
                tags = set()
                for field in selected:
                    raw = row.get(field) or ""
                    tags.update(genres_from(raw if raw.startswith(("[", "{")) else raw.split(",")))
                if key and tags:
                    key = Path(key).stem.lower()
                    result[key] = sorted(set(result.get(key, ())) | tags)
        for tags in result.values():
            counts.update(tags)
        return result, dict(rows=rows, labelled_hashes=len(result), fields=list(selected), genres=dict(sorted(counts.items())))
    finally:
        csv.field_size_limit(previous_limit)


def midi_walk(root):
    """Use readdir's cached file types, avoiding millions of remote lstat calls."""
    pending = [root]
    while pending:
        parent = pending.pop()
        folders, files = [], []
        with os.scandir(parent) as entries:
            for entry in entries:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if not entry.name.startswith(".") and entry.name != "__MACOSX":
                        folders.append(Path(entry.path))
                elif not entry.name.startswith("._") and entry.name.lower().endswith((".mid", ".midi")):
                    files.append(entry.name)
        yield parent, sorted(files)
        pending.extend(sorted(folders, reverse=True))


def build_midi_index(config, *, max_files_per_collection=None):
    """Index paths/labels without expensive bulk MIDI parsing or legacy JSONL.

    New deterministic grouped splits, independent of old AR splits. Giga subsets
    with the same filename hash share a split; ARIA audio segments share file_id.
    SymphonyNet contemporary stays unknown (never inferred to be classical).
    """
    ratios = config.DATA_SPLIT_RATIOS
    if set(ratios) != {"train", "validation", "test"} or any(v < 0 for v in ratios.values()) or not np.isclose(sum(ratios.values()), 1):
        raise ValueError("Split ratios must be nonnegative train/validation/test values summing to 1")
    if max_files_per_collection is not None and max_files_per_collection < 1:
        raise ValueError("Index limit must be positive")
    output = Path(config.INDEX_PATH)
    if output.exists():
        raise FileExistsError(f"Index already exists; choose another INDEX_PATH: {output}")
    for directory in config.DATASETS.values():
        if not Path(directory).is_dir():
            raise FileNotFoundError(directory)
    aria = {}
    if config.ARIA_METADATA:
        with open(config.ARIA_METADATA) as handle:
            aria = json.load(handle)
    giga = {}
    if config.GIGAMIDI_METADATA:
        giga, _ = load_gigamidi_genres(config.GIGAMIDI_METADATA,
                                      getattr(config, "GIGAMIDI_GENRE_FIELDS", ("music_styles_curated",)))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".building.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    # Millions of tiny B-tree writes are prohibitively slow on parallel scratch
    # storage. Build privately on node-local disk, then publish one complete DB.
    local = tempfile.TemporaryDirectory(prefix="improvnet-midi-index-")
    connection = sqlite3.connect(Path(local.name) / "index.sqlite")
    try:
        connection.executescript("""
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA cache_size=-65536;
            PRAGMA temp_store=MEMORY;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE files (id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL,
                collection TEXT NOT NULL, genre TEXT NOT NULL, genres TEXT NOT NULL,
                group_id TEXT NOT NULL, split TEXT NOT NULL);
        """)
        counts, labels = {}, {}
        for collection, directory in config.DATASETS.items():
            root = Path(directory).resolve()
            count, genre_counts = 0, Counter()
            stop = False
            for parent, files in midi_walk(root):
                for filename in files:
                    path = Path(parent) / filename
                    relative = path.relative_to(root)
                    if collection == "aria":
                        # Prefix is audio/file_id, not the segment number.
                        key = path.stem.split("_", 1)[0]
                        raw = aria.get(key, {}).get("metadata", {})
                        tags = genres_from(raw.get("genre"))
                    elif collection == "gigamidi":
                        key = path.stem.lower()
                        tags = giga.get(key, [])
                    else:
                        key = relative.as_posix()
                        if collection in ("pijama", "doug"):
                            tags = ["jazz"]
                        elif collection in ("classical_archives", "maestro"):
                            tags = ["classical"]
                        elif collection == "symphonynet" and "classical" in {p.lower() for p in relative.parts[:-1]} and "contemporary" not in {p.lower() for p in relative.parts[:-1]}:
                            tags = ["classical"]
                        else:
                            tags = []
                    group = collection + ":" + key
                    # Do not arbitrarily choose a conditioning genre for multi-label files.
                    primary = tags[0] if len(tags) == 1 else "multi" if tags else "unknown"
                    connection.execute("INSERT INTO files(path,collection,genre,genres,group_id,split) VALUES(?,?,?,?,?,?)",
                                       (str(path), collection, primary, json.dumps(tags), group,
                                        split_for(group, config.DATA_SPLIT_SEED, ratios)))
                    genre_counts[primary] += 1
                    count += 1
                    if count % 25000 == 0:
                        connection.commit()
                        print(f"Index {collection}: {count:,} MIDI files", flush=True)
                    if max_files_per_collection is not None and count >= max_files_per_collection:
                        stop = True
                        break
                if stop:
                    break
            counts[collection], labels[collection] = count, dict(genre_counts)
            connection.commit()
            print(f"Indexed {collection}: {count:,}; genres={dict(genre_counts)}", flush=True)
        connection.execute("CREATE INDEX files_split_collection ON files(split,collection,id)")
        connection.execute("CREATE INDEX files_group ON files(group_id)")
        split_counts = dict(connection.execute("SELECT split,COUNT(*) FROM files GROUP BY split"))
        report = dict(format=INDEX_FORMAT, settings=index_settings(config), collections=counts,
                      genres=labels, splits=split_counts, total=sum(counts.values()),
                      gigamidi_genres_available=bool(giga), limited=max_files_per_collection is not None,
                      max_files_per_collection=max_files_per_collection)
        connection.execute("INSERT INTO metadata VALUES('report',?)", (json.dumps(report),))
        connection.commit()
        connection.close()
        shutil.copyfile(Path(local.name) / "index.sqlite", temporary)
        os.replace(temporary, output)
        print(f"MIDI index ready: {output}; {report['total']:,} files", flush=True)
        return report
    finally:
        connection.close()
        local.cleanup()
        # A failed build is kept at its explicit .building path for inspection;
        # never publish an incomplete database or overwrite a working index.


def bin_tensors(piece, index, *, validate=True):
    """One aligned coarse bin, containing ALL fine frames, plus carried events."""
    if validate:
        piece.validate()
    if not 0 <= index < len(piece.velocity):
        raise IndexError(index)
    f = piece.config.frames_per_bin
    events = piece.context_for_bin(index).copy()
    positions = np.stack((events[:, 6] * f + events[:, 7] - index * f,
                          events[:, 8] * f + events[:, 9] - index * f), axis=-1)
    previous = piece.velocity[index - 1, :, -1] if index else np.zeros(128, np.uint8)
    # Actual stored pedal has no pitch dimension.
    return dict(roll=torch.from_numpy(np.stack((piece.velocity[index].astype(np.float32) / 127,
                                               piece.onset[index].astype(np.float32)))),
                pedal=torch.from_numpy(piece.pedal[index].astype(np.float32)[None] / 127),
                valid_mask=torch.from_numpy(piece.valid_mask[index].copy()),
                pedal_enabled=torch.tensor(piece.metadata.get("pedal_enabled", piece.config.include_pedal), dtype=torch.bool),
                target_program=torch.tensor(piece.metadata["target_program"], dtype=torch.long),
                target_channel=torch.tensor(piece.metadata["target_channel"], dtype=torch.long),
                is_drum=torch.tensor(piece.metadata.get("target_is_drum", piece.metadata["target_channel"] == 9)),
                bin_index=torch.tensor(index), previous_velocity=torch.from_numpy(previous.copy()).float() / 127,
                events=torch.from_numpy(events), event_positions=torch.from_numpy(positions),
                coarse_velocity=torch.from_numpy(piece.coarse_velocity[index].copy()).float() / 127,
                coarse_onsets=torch.from_numpy(piece.coarse_onsets[index].copy()).long(),
                coarse_pedal=torch.tensor(int(piece.coarse_pedal[index]) / 127))


def transpose_bin(item, shift):
    """Coherently transpose pitched target and context, without wraparound."""
    if bool(item["is_drum"]) or not shift:
        return item
    active = (item["roll"].any(dim=0).any(dim=-1) | (item["previous_velocity"] > 0)).nonzero().flatten()
    if "corrupted_roll" in item:
        active = torch.cat((active, item["corrupted_roll"].any(0).any(-1).nonzero().flatten()))
    context_pitches = item["events"][item["events"][:, 3] == 0, 4]
    pitches = torch.cat((active, context_pitches))
    if len(pitches) and (int(pitches.min()) + shift < 0 or int(pitches.max()) + shift > 127):
        raise ValueError("Pitch transposition would leave MIDI range")
    fields = [("roll", 1), ("previous_velocity", 0)]
    if "corrupted_roll" in item:
        fields.append(("corrupted_roll", 1))
    for name, dim in fields:
        array = item[name]
        shifted = torch.zeros_like(array)
        src, dst = [slice(None)] * array.ndim, [slice(None)] * array.ndim
        src[dim], dst[dim] = (slice(None, -shift), slice(shift, None)) if shift > 0 else (slice(-shift, None), slice(None, shift))
        shifted[tuple(dst)] = array[tuple(src)]
        item[name] = shifted
    pitched = item["events"][:, 3] == 0
    item["events"][pitched, 4] += shift
    item["coarse_velocity"] = item["roll"][0].amax(-1)
    item["coarse_onsets"] = item["roll"][1].sum(-1).long()
    return item


def collate_bins(items):
    result = {k: torch.stack([item[k] for item in items]) for k, v in items[0].items()
              if isinstance(v, torch.Tensor) and k not in ("events", "event_positions")}
    size = max(len(item["events"]) for item in items)
    result["events"] = torch.zeros((len(items), size, len(EVENT_FIELDS)), dtype=torch.long)
    result["event_positions"] = torch.zeros((len(items), size, 2), dtype=torch.long)
    result["event_mask"] = torch.zeros((len(items), size), dtype=torch.bool)
    for i, item in enumerate(items):
        n = len(item["events"])
        result["events"][i, :n], result["event_positions"][i, :n] = item["events"], item["event_positions"]
        result["event_mask"][i, :n] = True
    result["metadata"] = [item.get("metadata", {}) for item in items]
    return result


class LocalVAEDataset(Dataset):
    """Virtual, collection-balanced epoch; randomness depends on epoch/item only.

    Read raw MIDI lazily. Invalid/empty files are retried with a bounded budget
    and logged, not silently counted as empty training targets. No target JSONL.
    """
    def __init__(self, config, split="train", *, error_dir=None):
        self.config, self.split, self.epoch = config, split, 0
        self.error_dir = Path(error_dir) if error_dir else None
        self.connection, self.pid = None, None
        self.cache, self.cache_bytes, self.logged = OrderedDict(), 0, set()
        self.tokenizer = HierarchicalTokenizer(GridConfig(coarse_ms=config.COARSE_MS, fine_ms=config.FINE_MS,
                                                          piano_programs=tuple(config.PIANO_PROGRAMS), max_frames=config.MAX_FRAMES))
        with self.connect() as db:
            self.report = json.loads(db.execute("SELECT value FROM metadata WHERE key='report'").fetchone()[0])
            if self.report["format"] != INDEX_FORMAT or self.report["settings"] != index_settings(config):
                raise ValueError("MIDI index settings differ; build a new index for these roots/splits/metadata")
            self.groups = {name: np.fromiter((r[0] for r in db.execute("SELECT id FROM files WHERE split=? AND collection=? ORDER BY id", (split, name))), np.int64)
                           for name in config.DATASETS}
            fraction = float(getattr(config, "GIGAMIDI_DRUM_ONLY_FRACTION", .05))
            if not np.isfinite(fraction) or not 0 <= fraction <= 1:
                raise ValueError("GIGAMIDI_DRUM_ONLY_FRACTION must be between 0 and 1")
            self.drum_subset = None
            if "gigamidi" in self.groups:
                # This selects the upstream drums-only folder subset, not all
                # mixtures containing drums and not a filename-based GM guess.
                drums = np.fromiter((r[0] for r in db.execute(
                    "SELECT id FROM files WHERE split=? AND collection='gigamidi' "
                    "AND lower(path) LIKE '%/drums-only/%' ORDER BY id", (split,))), np.int64)
                count = int(np.floor(len(drums) * fraction))
                rng = np.random.default_rng(seed_for(config.DATA_SPLIT_SEED, "gigamidi-drums-only", split))
                retained = rng.choice(drums, size=count, replace=False) if count else np.empty(0, np.int64)
                other = self.groups["gigamidi"][~np.isin(self.groups["gigamidi"], drums, assume_unique=True)]
                self.groups["gigamidi"] = np.sort(np.concatenate((other, retained)))
                self.drum_subset = dict(total=len(drums), retained=count, fraction=fraction)
        self.groups = {k: v for k, v in self.groups.items() if len(v)}
        if not self.groups:
            raise ValueError(f"MIDI index contains no {split} files")
        self.names = sorted(self.groups)
        self.weights = [float(config.COLLECTION_WEIGHTS.get(n, 1)) for n in self.names]
        if any(not np.isfinite(w) or w <= 0 for w in self.weights):
            raise ValueError("Collection weights must be finite and positive")

    def connect(self):
        if self.pid != os.getpid():
            if self.connection is not None:
                self.connection.close()
            uri = Path(self.config.INDEX_PATH).resolve().as_uri() + "?mode=ro"
            self.connection = sqlite3.connect(uri, uri=True)
            self.pid = os.getpid()
        return self.connection

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(connection=None, pid=None, cache=OrderedDict(), cache_bytes=0, logged=set())
        return state

    def __len__(self):
        return self.config.SAMPLES_PER_EPOCH if self.split == "train" else min(self.config.VALIDATION_SAMPLES, sum(map(len, self.groups.values())))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        epoch = self.epoch if self.split == "train" else 0
        rng = random.Random(seed_for(self.config.SEED, self.split, epoch, index))
        for attempt in range(self.config.MAX_DATA_RETRIES):
            collection = rng.choices(self.names, self.weights)[0]
            ids = self.groups[collection]
            row_id = int(ids[rng.randrange(len(ids))])
            row = self.connect().execute("SELECT path,genre,genres FROM files WHERE id=?", (row_id,)).fetchone()
            path, genre, genres = row
            target_seed = seed_for(self.config.SEED, epoch, row_id)
            key = (row_id, target_seed)
            try:
                if key in self.cache:
                    piece, size, notes = self.cache.pop(key)
                    self.cache[key] = (piece, size, notes)
                else:
                    piece = self.tokenizer.encode(path, seed=target_seed, recover=True)
                    if not piece.metadata["target_present"]:
                        raise ValueError("No valid target notes")
                    size = sum(getattr(piece, k).nbytes for k in ("velocity", "onset", "pedal", "valid_mask", "events"))
                    notes = target_note_rows(piece) if getattr(self.config, "SEMANTIC_CORRUPTION", False) else None
                    if notes is not None:
                        size += notes.nbytes
                    budget = int(self.config.CACHE_MB_PER_WORKER * 1024**2)
                    while self.cache and self.cache_bytes + size > budget:
                        _, (_, removed, _) = self.cache.popitem(last=False)
                        self.cache_bytes -= removed
                    if size <= budget:
                        self.cache[key] = piece, size, notes
                        self.cache_bytes += size
                active_bins = np.flatnonzero(piece.velocity.any(axis=(1, 2)))
                if not len(active_bins):
                    raise ValueError("Target roll is empty after normalization")
                if self.split == "train" and rng.random() >= self.config.ACTIVE_BIN_PROB:
                    bin_index = rng.randrange(len(piece.velocity))
                else:
                    bin_index = int(active_bins[rng.randrange(len(active_bins))])
                # encode() already validated this whole piece. Re-validating
                # every frame on each cache hit is unnecessary for one bin.
                item = bin_tensors(piece, bin_index, validate=False)
                if getattr(self.config, "SEMANTIC_CORRUPTION", False):
                    # Independent stream: enabling denoising does not change file,
                    # target-part or bin selection. Validation's
                    # corruptions are fixed; neither split changes clean targets.
                    semantic_rng = random.Random(seed_for(self.config.SEED, self.split, epoch, index, "semantic"))
                    item["corrupted_roll"] = corrupt_roll_bin(piece, bin_index, self.config, semantic_rng, notes=notes)
                if self.split == "train" and self.config.PITCH_AUGMENTATION and not bool(item["is_drum"]):
                    pitches = torch.cat(((item["roll"].any(0).any(-1) | (item["previous_velocity"] > 0)).nonzero().flatten(),
                                         item["events"][item["events"][:, 3] == 0, 4]))
                    if "corrupted_roll" in item:
                        pitches = torch.cat((pitches, item["corrupted_roll"].any(0).any(-1).nonzero().flatten()))
                    low, high = self.config.PITCH_SHIFT_RANGE
                    if len(pitches):
                        low, high = max(low, -int(pitches.min())), min(high, 127 - int(pitches.max()))
                    item = transpose_bin(item, rng.randint(low, high))
                item["retries"] = torch.tensor(attempt)
                item["metadata"] = dict(path=path, genre=genre, genres=json.loads(genres), collection=collection)
                return item
            # Mido's malformed-key-signature exception derives directly from
            # Exception, not ValueError/KeyError. Treat it as bad source data,
            # not a worker failure; keep retries bounded and log the source.
            except (OSError, EOFError, ValueError, KeyError, IndexError, KeySignatureError) as error:
                if getattr(self.config, "LOG_DATA_ERRORS", False) and self.error_dir and path not in self.logged:
                    self.error_dir.mkdir(parents=True, exist_ok=True)
                    with (self.error_dir / f"data_errors.{os.getpid()}.jsonl").open("a") as handle:
                        handle.write(json.dumps(dict(path=path, error=str(error))) + "\n")
                    self.logged.add(path)
        raise RuntimeError(f"Could not find usable {self.split} MIDI after {self.config.MAX_DATA_RETRIES} tries; last path: {path}")
