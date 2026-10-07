"""Metadata refresh tests: source index/IDs/splits remain untouched."""
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest

from improvnet.utils.local_vae_data import index_settings, load_gigamidi_genres
from improvnet.scripts.update_midi_genres import refresh


class GenreMetadataTests(unittest.TestCase):
    def test_extended_columns_large_fields_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("md5", "music_styles_curated", "music_style_scraped", "loops"))
                writer.writerow(("abc", "Jazz,Blues", "renaissance", "x" * 150000))
                writer.writerow(("abc", "pop", "language", ""))
                writer.writerow(("def", "", "jazz", ""))
            limit = csv.field_size_limit()
            tags, report = load_gigamidi_genres(path)
            self.assertEqual(tags, {"abc": ["blues", "jazz", "pop"]})
            self.assertEqual(report["rows"], 3)
            self.assertEqual(csv.field_size_limit(), limit)

    def test_legacy_columns_and_missing_field_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.csv"
            path.write_text("md5,genre\nabc,classical\n")
            self.assertEqual(load_gigamidi_genres(path)[0], {"abc": ["classical"]})
            with self.assertRaises(ValueError):
                load_gigamidi_genres(path, ("bad_column",))

    def test_refresh_preserves_source_and_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target, metadata = root / "old.sqlite", root / "new.sqlite", root / "genres.csv"
            metadata.write_text("md5,music_styles_curated\nabc,jazz\n")
            c = SimpleNamespace(DATASETS={"gigamidi": str(root)}, DATA_SPLIT_SEED=42,
                DATA_SPLIT_RATIOS={"train": .97, "validation": .02, "test": .01}, ARIA_METADATA=None,
                GIGAMIDI_METADATA=None, INDEX_PATH=str(source))
            with sqlite3.connect(source) as db:
                db.executescript("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT); "
                    "CREATE TABLE files(id INTEGER PRIMARY KEY,path TEXT,collection TEXT,genre TEXT,genres TEXT,group_id TEXT,split TEXT);")
                db.execute("INSERT INTO metadata VALUES('report',?)", (json.dumps(dict(settings=index_settings(c),genres={"gigamidi":{"unknown":2}})),))
                db.executemany("INSERT INTO files VALUES(?,?,?,?,?,?,?)", (
                    (11,"abc.mid","gigamidi","unknown","[]","gigamidi:abc","train"),
                    (22,"def.mid","gigamidi","unknown","[]","gigamidi:def","validation")))
            before = source.read_bytes()
            c.GIGAMIDI_METADATA, c.GIGAMIDI_GENRE_FIELDS, c.INDEX_PATH = str(metadata), ("music_styles_curated",), str(target)
            with redirect_stdout(io.StringIO()):
                report = refresh(source, c)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(report["genres"], ["jazz"])
            with sqlite3.connect(target) as db:
                self.assertEqual(db.execute("SELECT id,genre,split FROM files ORDER BY id").fetchall(),
                                 [(11,"jazz","train"),(22,"unknown","validation")])
            with self.assertRaises(FileExistsError):
                refresh(source, c)


if __name__ == "__main__":
    unittest.main()
