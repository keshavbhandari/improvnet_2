"""Publish a separately labelled MIDI index, preserving every old path/ID/group/split.

Never edits the source index, MIDI files or checkpoints. Does not start training.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from types import SimpleNamespace

from improvnet.model import local_vae_config
from improvnet.utils.local_vae_data import index_settings, load_gigamidi_genres


def refresh(source, config):
    source, output = Path(source).resolve(), Path(config.INDEX_PATH).resolve()
    if source == output or output.exists():
        raise FileExistsError("Use a new output index; existing indexes are never overwritten")
    tags, csv_report = load_gigamidi_genres(config.GIGAMIDI_METADATA, config.GIGAMIDI_GENRE_FIELDS)
    print(f"Metadata: {csv_report}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="improvnet-midi-genres-") as local:
        path = Path(local) / "index.sqlite"
        with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as old, sqlite3.connect(path) as db:
            report = json.loads(old.execute("SELECT value FROM metadata WHERE key='report'").fetchone()[0])
            expected = index_settings(config)
            if any(report["settings"][k] != expected[k] for k in ("datasets", "split_seed", "ratios", "aria_metadata")):
                raise ValueError("Source index differs in paths/splits/ARIA metadata")
            old.backup(db)
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("CREATE TABLE giga_labels (hash TEXT PRIMARY KEY, genre TEXT, genres TEXT)")
            db.executemany("INSERT INTO giga_labels VALUES(?,?,?)", ((key, values[0] if len(values) == 1 else "multi", json.dumps(values)) for key, values in tags.items()))
            db.execute("UPDATE files SET genre=COALESCE((SELECT genre FROM giga_labels WHERE hash=substr(files.group_id,10)), 'unknown'), "
                       "genres=COALESCE((SELECT genres FROM giga_labels WHERE hash=substr(files.group_id,10)), '[]') WHERE collection='gigamidi'")
            matched = db.execute("SELECT COUNT(*) FROM files WHERE collection='gigamidi' AND genres!='[]'").fetchone()[0]
            report["genres"]["gigamidi"] = dict(db.execute("SELECT genre,COUNT(*) FROM files WHERE collection='gigamidi' GROUP BY genre"))
            report.update(settings=expected, gigamidi_genres_available=bool(matched),
                          gigamidi_metadata_report=csv_report, gigamidi_labelled_paths=matched)
            db.execute("UPDATE metadata SET value=? WHERE key='report'", (json.dumps(report),))
            db.execute("DROP TABLE giga_labels")
            db.commit()
            labels = sorted({tag for (raw,) in db.execute("SELECT DISTINCT genres FROM files WHERE split='train'") for tag in json.loads(raw)})
        pending = output.with_name(output.name + f".building.{os.getpid()}")
        if pending.exists() or output.exists():
            raise FileExistsError("Index publication target already exists")
        shutil.copyfile(path, pending)
        # A hard link publishes without replacing a concurrently created index.
        os.link(pending, output)
        pending.unlink()
    print(f"Ready: {output}; labelled GigaMIDI paths={matched:,}; {len(labels)} training genres: {labels}", flush=True)
    return dict(index=str(output), labelled_gigamidi_paths=matched, genres=labels, metadata=csv_report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", default=local_vae_config.INDEX_PATH)
    parser.add_argument("--config", default="improvnet.model.latent_diffusion_config")
    args = parser.parse_args()
    module = importlib.import_module(args.config)
    config = SimpleNamespace(**{k: v for k, v in vars(module).items() if k.isupper()})
    refresh(args.source_index, config)


if __name__ == "__main__":
    main()
