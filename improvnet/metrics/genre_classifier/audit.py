"""Scan ALL indexed metadata headers without reading the large token arrays.

Outputs a compact JSON report and a SQLite byte-offset manifest of labeled
examples only; unknown/null/other labels are ignored. Existing artifacts are never
overwritten. No model imports/jobs.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sqlite3
import time
import unicodedata

from .genre_classifier_config import ARTIFACT_DIR, JSONL_FILES


UNKNOWN_GENRES = {"", "unknown", "unk", "none", "null", "n/a", "na", "unlabeled", "unlabelled", "other"}


def normalize_genre(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Expected a single string genre label or null")
    value = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    return None if value in UNKNOWN_GENRES else value


def source_signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def piece_key(metadata):
    # Explicit identity wins. Otherwise use filename identity across directories
    # and sources, not record offset or crop. Same piece's windows share a fold.
    for field in ("piece_id", "midi_hash"):
        if metadata.get(field) is not None:
            return f"{field}:{metadata[field]}"
    path = metadata.get("midi_filepath")
    if not isinstance(path, str) or not path:
        raise ValueError("Missing MIDI identity; cannot make a piece-grouped split")
    return "filename:" + Path(path.replace("\\", "/")).stem.casefold()


def piece_split(key, seed=42, train_fraction=.9, validation_fraction=.05):
    if not math.isfinite(train_fraction) or not math.isfinite(validation_fraction) or train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Train/validation/test fractions must be positive and sum to one")
    value = int.from_bytes(hashlib.sha256(f"{seed}\0{key}".encode()).digest()[:8], "big") / 2**64
    return "train" if value < train_fraction else "validation" if value < train_fraction + validation_fraction else "test"


def read_metadata(fd, offset, initial_bytes=512, max_bytes=65536):
    """Parse JSON keys/values before `tokens`, respecting escaped strings.

    Metadata after the token array is deliberately unsupported: fail explicitly
    rather than silently pretending such records have unknown genres.
    """
    decoder = json.JSONDecoder()
    length = initial_bytes
    while length <= max_bytes:
        raw = os.pread(fd, length, offset)
        try:
            text = raw.decode("utf-8")
            if not text.lstrip().startswith("{"):
                raise ValueError("Indexed record does not start with a JSON object")
            position = text.index("{") + 1
            metadata = {}
            while True:
                while position < len(text) and text[position].isspace():
                    position += 1
                if position < len(text) and text[position] == "}":
                    raise ValueError("Record has no tokens field")
                key, position = decoder.raw_decode(text, position)
                if not isinstance(key, str):
                    raise ValueError("Non-string JSON object key")
                while position < len(text) and text[position].isspace():
                    position += 1
                if position >= len(text):
                    raise json.JSONDecodeError("Incomplete header", text, position)
                if text[position] != ":":
                    raise ValueError("Invalid JSON key separator")
                position += 1
                if key == "tokens":
                    if "genre" not in metadata or "midi_filepath" not in metadata:
                        raise ValueError("genre/midi_filepath must precede tokens in the JSONL header")
                    return metadata
                while position < len(text) and text[position].isspace():
                    position += 1
                value, position = decoder.raw_decode(text, position)
                metadata[key] = value
                while position < len(text) and text[position].isspace():
                    position += 1
                if position >= len(text):
                    raise json.JSONDecodeError("Incomplete header", text, position)
                if text[position] == ",":
                    position += 1
                elif text[position] == "}":
                    raise ValueError("Record has no tokens field")
                else:
                    raise ValueError("Invalid JSON field separator")
        except (UnicodeDecodeError, json.JSONDecodeError):
            length *= 2
    raise ValueError(f"Could not parse metadata within {max_bytes} bytes at offset {offset}")


def audit_dataset(paths, output_prefix, *, seed=42, train_fraction=.9, validation_fraction=.05,
                  progress_every=100000):
    piece_split("check", seed, train_fraction, validation_fraction)
    prefix = Path(output_prefix).resolve()
    report_path, database_path = Path(str(prefix) + ".json"), Path(str(prefix) + ".sqlite")
    if report_path.exists() or database_path.exists():
        raise FileExistsError("Audit artifacts exist; choose a new output prefix (no overwrite)")
    signatures = [source_signature(path) for path in paths]
    if not signatures or len({s["path"] for s in signatures}) != len(signatures):
        raise ValueError("Need distinct existing JSONL input files")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.execute("CREATE TABLE examples(source INTEGER, offset INTEGER, genre TEXT, piece TEXT, split TEXT, PRIMARY KEY(source, offset))")
    counts, raw_counts, per_source, pending = Counter(), Counter(), [], []
    started = time.monotonic()
    try:
        for source, signature in enumerate(signatures):
            path = signature["path"]
            index_path = Path(path).with_name(Path(path).stem + "_index.pkl")
            index_signature = source_signature(index_path)
            # These are the user's existing trusted local byte indexes.
            with index_path.open("rb") as handle:
                offsets = pickle.load(handle)
            file_counts, unknown = Counter(), 0
            last_offset = -1
            with open(path, "rb") as handle:
                for number, offset in enumerate(offsets, 1):
                    if not isinstance(offset, int) or isinstance(offset, bool) or not last_offset < offset < signature["size"]:
                        raise ValueError(f"Invalid/unsorted byte index at record {number}: {path}")
                    last_offset = offset
                    metadata = read_metadata(handle.fileno(), offset)
                    label = normalize_genre(metadata["genre"])
                    raw_counts[json.dumps(metadata["genre"], ensure_ascii=False)] += 1
                    if label is None:
                        unknown += 1
                    else:
                        key = piece_key(metadata)
                        split = piece_split(key, seed, train_fraction, validation_fraction)
                        pending.append((source, offset, label, key, split))
                        counts[label] += 1
                        file_counts[label] += 1
                    if len(pending) >= 10000:
                        connection.executemany("INSERT INTO examples VALUES (?, ?, ?, ?, ?)", pending)
                        connection.commit()
                        pending.clear()
                    if progress_every and number % progress_every == 0:
                        print(f"{Path(path).name}: {number:,}/{len(offsets):,} headers; labeled={sum(file_counts.values()):,}; elapsed={time.monotonic()-started:.0f}s", flush=True)
            if source_signature(path) != signature:
                raise ValueError(f"Dataset changed during audit: {path}")
            if source_signature(index_path) != index_signature:
                raise ValueError(f"Byte index changed during audit: {index_path}")
            per_source.append({**signature, "index": index_signature, "records": len(offsets),
                               "unknown": unknown, "genre_counts": dict(sorted(file_counts.items()))})
        if pending:
            connection.executemany("INSERT INTO examples VALUES (?, ?, ?, ?, ?)", pending)
        connection.commit()
        connection.execute("CREATE INDEX by_split ON examples(split, source, offset)")
        connection.commit()
        split_counts = {split: dict(connection.execute("SELECT genre, COUNT(*) FROM examples WHERE split=? GROUP BY genre", (split,)))
                        for split in ("train", "validation", "test")}
        group_count = connection.execute("SELECT COUNT(DISTINCT piece) FROM examples").fetchone()[0]
        conflicting_groups = connection.execute("SELECT COUNT(*) FROM (SELECT piece FROM examples GROUP BY piece HAVING COUNT(DISTINCT genre)>1)").fetchone()[0]
        manifest_counts = Counter(dict(connection.execute("SELECT genre, COUNT(*) FROM examples GROUP BY genre")))
        report = {"schema_version": 1, "complete": True, "sources": per_source,
                  "records": sum(s["records"] for s in per_source), "unknown": sum(s["unknown"] for s in per_source),
                  "labeled": sum(counts.values()), "labels": sorted(manifest_counts), "genre_counts": dict(sorted(counts.items())),
                  "manifest_genre_counts": dict(sorted(manifest_counts.items())),
                  "raw_genre_counts": dict(sorted(raw_counts.items())), "split_counts": split_counts,
                  "piece_groups": group_count, "groups_with_multiple_genre_labels": conflicting_groups,
                  "split_seed": seed, "split_fractions": [train_fraction, validation_fraction, 1-train_fraction-validation_fraction],
                  "grouping": "piece_id/midi_hash if present; otherwise casefolded MIDI filename stem across sources",
                  "database": str(database_path), "elapsed_seconds": time.monotonic()-started}
        with report_path.open("x") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(json.dumps({k: report[k] for k in ("records", "unknown", "labeled", "genre_counts", "split_counts", "groups_with_multiple_genre_labels")}, indent=2), flush=True)
        return report
    finally:
        connection.close()
    # A failure intentionally leaves a partial SQLite artifact, without a
    # completed report. Dataset loading requires the report's complete flag.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", nargs="+", default=list(JSONL_FILES))
    parser.add_argument("--output-prefix", default=str(Path(ARTIFACT_DIR) / "dataset_manifest"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=.9)
    parser.add_argument("--validation-fraction", type=float, default=.05)
    args = parser.parse_args()
    audit_dataset(args.jsonl, args.output_prefix, seed=args.seed, train_fraction=args.train_fraction,
                  validation_fraction=args.validation_fraction)


if __name__ == "__main__":
    main()
