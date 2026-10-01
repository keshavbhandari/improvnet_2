"""Pure metadata tests; no torch, model, or production dataset reads."""

import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest

from improvnet.metrics.genre_classifier.audit import (
    audit_dataset, normalize_genre, piece_key, piece_split, read_metadata,
)


def write_records(path, records):
    offsets = []
    with path.open("wb") as handle:
        for record in records:
            offsets.append(handle.tell())
            handle.write((json.dumps(record, ensure_ascii=False) + "\n").encode())
    with path.with_name(path.stem + "_index.pkl").open("wb") as handle:
        pickle.dump(offsets, handle)
    return offsets


def tiny_records():
    tokens = ["<S>"] + [["instrument", "Acoustic Piano"], ["note", 60, 80], ["onset", 0], ["dur", 100]] * 20 + ["<E>"]
    rows = []
    for genre in ("classical", "jazz"):
        for split in ("train", "validation", "test"):
            for number in range(3):
                salt = 0
                while True:
                    name = f"{genre}_{split}_{number}_{salt}"
                    if piece_split("filename:" + name) == split:
                        break
                    salt += 1
                rows.append({"midi_filepath": f"/music/{name}.mid", "genre": genre, "tokens": tokens})
    rows.append({"midi_filepath": "/music/unknown.mid", "genre": None, "tokens": tokens})
    return rows


class GenreAuditTests(unittest.TestCase):
    def test_label_normalization_does_not_collapse_real_genres(self):
        self.assertEqual(normalize_genre("  Heavy   Metal "), "heavy metal")
        self.assertEqual(normalize_genre("ＡＭＢＩＥＮＴ"), "ambient")
        for value in (None, "", " UNKNOWN ", "n/a", "null"):
            self.assertIsNone(normalize_genre(value))
        self.assertIsNone(normalize_genre("other"))
        with self.assertRaises(ValueError):
            normalize_genre(["jazz", "blues"])

    def test_metadata_handles_escaped_strings_unicode_and_long_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            record = {"midi_filepath": '/music/a"tokens"genre' + "é" * 700 + ".mid",
                      "genre": "jazz", "composer": {"name": "x"}, "tokens": ["<S>", "<E>"]}
            write_records(path, [record])
            with path.open("rb") as handle:
                metadata = read_metadata(handle.fileno(), 0, initial_bytes=128)
            self.assertEqual(metadata, {k: v for k, v in record.items() if k != "tokens"})

    def test_metadata_after_tokens_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            write_records(path, [{"tokens": [], "genre": "jazz", "midi_filepath": "x.mid"}])
            with path.open("rb") as handle, self.assertRaises(ValueError):
                read_metadata(handle.fileno(), 0)

    def test_grouping_and_fold_assignment_are_stable(self):
        one = piece_key({"midi_filepath": "/train/ABC.mid"})
        two = piece_key({"midi_filepath": "C:\\test\\abc.mid"})
        self.assertEqual(one, two)
        self.assertEqual(piece_split(one), piece_split(two))
        self.assertEqual(piece_key({"piece_id": 12}), "piece_id:12")
        with self.assertRaises(ValueError):
            piece_key({})
        for values in ((0., .1), (.9, .1), (.5, float("nan"))):
            with self.assertRaises(ValueError):
                piece_split("x", train_fraction=values[0], validation_fraction=values[1])

    def test_full_audit_counts_unknowns_and_preserves_duplicate_piece_fold(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.jsonl"
            second = Path(directory) / "b.jsonl"
            records = tiny_records()
            write_records(path, records)
            write_records(second, [records[0]])
            prefix = Path(directory) / "manifest"
            report = audit_dataset([path, second], prefix, progress_every=0)
            self.assertTrue(report["complete"])
            self.assertEqual((report["records"], report["labeled"], report["unknown"]), (20, 19, 1))
            self.assertEqual(report["labels"], ["classical", "jazz"])
            with sqlite3.connect(report["database"]) as connection:
                leakage = connection.execute("SELECT piece FROM examples GROUP BY piece HAVING COUNT(DISTINCT split)>1").fetchall()
            self.assertEqual(leakage, [])
            for split in ("train", "validation", "test"):
                self.assertGreater(report["split_counts"][split]["jazz"], 0)
            with self.assertRaises(FileExistsError):
                audit_dataset([path], prefix, progress_every=0)

    def test_unknown_and_other_annotations_are_excluded_in_all_folds(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            rows = tiny_records()
            rows += [{**rows[0], "genre": value, "midi_filepath": f"/music/ignored_{index}.mid"}
                     for index, value in enumerate((None, "unknown", "other", " Other "))]
            write_records(path, rows)
            report = audit_dataset([path], Path(directory) / "manifest", progress_every=0)
            self.assertEqual(report["labels"], ["classical", "jazz"])
            self.assertEqual(report["unknown"], 5)
            with sqlite3.connect(report["database"]) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM examples").fetchone()[0], 18)
                self.assertEqual(connection.execute("SELECT DISTINCT genre FROM examples ORDER BY genre").fetchall(),
                                 [("classical",), ("jazz",)])


if __name__ == "__main__":
    unittest.main()
