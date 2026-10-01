"""Build continuity-only byte-offset metadata indexes, including unknown genres.

No model imports or token-array scans. Group duplicate identities across folds
and negatives, but positive excerpts always come from one performance/record.
"""

import argparse
import json
from pathlib import Path
import pickle
import sqlite3
import time

from improvnet.metrics.genre_classifier.audit import piece_key, piece_split, read_metadata, source_signature
from improvnet.metrics.continuity.continuity_config import ARTIFACT_DIR, JSONL_FILES, MIDI_FILEPATH_EXCLUDE_SUBSTRINGS


def audit_dataset(paths, output_prefix, *, seed=42, train_fraction=.9, validation_fraction=.05,
                  exclusions=None, progress_every=100000):
    piece_split("check", seed, train_fraction, validation_fraction)
    prefix = Path(output_prefix).resolve()
    report_path, database = Path(str(prefix) + ".json"), Path(str(prefix) + ".sqlite")
    if report_path.exists() or database.exists():
        raise FileExistsError("Continuity artifacts exist; choose a fresh audit prefix (no overwrite)")
    signatures = [source_signature(path) for path in paths]
    if not signatures or len({s["path"] for s in signatures}) != len(signatures):
        raise ValueError("Need distinct existing JSONL sources")
    exclusions = {str(Path(path).resolve()): list(values) for path, values in (exclusions or {}).items()}
    prefix.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE examples(source INTEGER, offset INTEGER, piece TEXT, split TEXT, PRIMARY KEY(source,offset))")
    pending, sources = [], []
    started = time.monotonic()
    try:
        for source, signature in enumerate(signatures):
            path = Path(signature["path"])
            index = path.with_name(path.stem + "_index.pkl")
            index_signature = source_signature(index)
            # User-provided trusted local indexes, never arbitrary remote pickle.
            with index.open("rb") as handle:
                offsets = pickle.load(handle)
            excluded, previous = 0, -1
            with path.open("rb") as handle:
                for number, offset in enumerate(offsets, 1):
                    if not isinstance(offset, int) or isinstance(offset, bool) or not previous < offset < signature["size"]:
                        raise ValueError(f"Invalid byte index at {path}:{offset}")
                    previous = offset
                    metadata = read_metadata(handle.fileno(), offset)
                    if any(value in metadata["midi_filepath"] for value in exclusions.get(str(path), ())):
                        excluded += 1
                    else:
                        key = piece_key(metadata)
                        pending.append((source, offset, key, piece_split(key, seed, train_fraction, validation_fraction)))
                    if len(pending) >= 10000:
                        connection.executemany("INSERT INTO examples VALUES (?,?,?,?)", pending)
                        connection.commit()
                        pending.clear()
                    if progress_every and number % progress_every == 0:
                        print(f"{path.name}: {number:,}/{len(offsets):,} headers; elapsed={time.monotonic()-started:.0f}s", flush=True)
            if source_signature(path) != signature or source_signature(index) != index_signature:
                raise ValueError("Source/index changed while auditing")
            sources.append({**signature, "index": index_signature, "records": len(offsets), "excluded": excluded})
        if pending:
            connection.executemany("INSERT INTO examples VALUES (?,?,?,?)", pending)
        connection.execute("CREATE INDEX by_split ON examples(split,piece,source,offset)")
        connection.commit()
        report = {
            "schema_version": 1, "format": "continuity_pairs_v1", "complete": True,
            "sources": sources, "exclusions": exclusions,
            "split_seed": seed, "split_fractions": [train_fraction, validation_fraction, 1-train_fraction-validation_fraction],
            "split_counts": dict(connection.execute("SELECT split,COUNT(*) FROM examples GROUP BY split")),
            "piece_groups": connection.execute("SELECT COUNT(DISTINCT piece) FROM examples").fetchone()[0],
            "records": sum(s["records"] for s in sources), "excluded": sum(s["excluded"] for s in sources),
            "database": str(database), "elapsed_seconds": time.monotonic()-started,
            "grouping": "piece_id/midi_hash or casefolded MIDI filename stem; positives from the same record",
        }
        with report_path.open("x") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(json.dumps({key: report[key] for key in ("records", "excluded", "piece_groups", "split_counts")}, indent=2), flush=True)
        return report
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", nargs="+", default=list(JSONL_FILES))
    parser.add_argument("--output-prefix", default=str(Path(ARTIFACT_DIR) / "dataset_manifest"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-drums-only", action="store_true")
    args = parser.parse_args()
    audit_dataset(args.jsonl, args.output_prefix, seed=args.seed,
                  exclusions={} if args.include_drums_only else MIDI_FILEPATH_EXCLUDE_SUBSTRINGS)


if __name__ == "__main__":
    main()
