"""Small synthetic archives; no access to the actual MIDI corpus."""
import io
from pathlib import Path
import stat
import tarfile
import tempfile
import unittest
import zipfile

from improvnet.scripts.extract_midi_archives import Budget, extract_archive, safe_target


class ExtractionTests(unittest.TestCase):
    def test_nested_zip_and_resume(self):
        nested = io.BytesIO()
        with zipfile.ZipFile(nested, "w") as z:
            z.writestr("piano/a.mid", b"midi-example")
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source, output = root / "outer.zip", root / "output"
            with zipfile.ZipFile(source, "w") as z:
                z.writestr("inner.zip", nested.getvalue())
                z.writestr("second.MIDI", b"other-midi")
            result = extract_archive(source, output, Budget(1_000_000))
            self.assertEqual(result["midi_files"], 2)
            self.assertEqual(result["nested_archives"], 1)
            self.assertEqual((output / "inner.zip.unpacked/piano/a.mid").read_bytes(), b"midi-example")
            self.assertEqual(result, extract_archive(source, output, Budget(1_000_000)))
            self.assertTrue(source.exists())

    def test_tar_links_skipped_and_regular_file_extracted(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source, output = root / "notes.tar.gz", root / "output"
            with tarfile.open(source, "w:gz") as t:
                member = tarfile.TarInfo("a.mid")
                member.size = 4
                t.addfile(member, io.BytesIO(b"MThd"))
                link = tarfile.TarInfo("escape")
                link.type, link.linkname = tarfile.SYMTYPE, "../../outside"
                t.addfile(link)
            result = extract_archive(source, output, Budget(1_000_000))
            self.assertEqual(result["midi_files"], 1)
            self.assertEqual(result["skipped_links"], 1)
            self.assertFalse((output / "escape").exists())

    def test_escaping_windows_and_absolute_paths_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ("../escape.mid", "/escape.mid", "C:\\escape.mid", "folder\\..\\escape.mid"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    safe_target(Path(root), name)

    def test_zip_symlinks_skipped_and_existing_conflicts_not_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source, output = root / "notes.zip", root / "output"
            output.mkdir()
            (output / "a.mid").write_bytes(b"user-content")
            with zipfile.ZipFile(source, "w") as z:
                link = zipfile.ZipInfo("link")
                link.create_system = 3
                link.external_attr = (stat.S_IFLNK | 0o777) << 16
                z.writestr(link, "../escape")
                z.writestr("a.mid", b"other-data")
            with self.assertRaises(FileExistsError):
                extract_archive(source, output, Budget(1_000_000))
            self.assertEqual((output / "a.mid").read_bytes(), b"user-content")
            self.assertFalse((output / "link").exists())

    def test_byte_budget_stops_before_large_member_is_written(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source, output = root / "notes.zip", root / "output"
            with zipfile.ZipFile(source, "w") as z:
                z.writestr("a.mid", b"x" * 100)
            with self.assertRaisesRegex(ValueError, "limit"):
                extract_archive(source, output, Budget(10))
            self.assertFalse((output / "a.mid").exists())

    def test_parallel_nested_archives_have_independent_destinations(self):
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as z:
            z.writestr("same.mid", b"MThd")
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source, output = root / "outer.zip", root / "output"
            with zipfile.ZipFile(source, "w") as z:
                z.writestr("a.zip", inner.getvalue())
                z.writestr("b.zip", inner.getvalue())
            result = extract_archive(source, output, Budget(1_000_000), nested_workers=2)
            self.assertEqual(result["midi_files"], 2)
            self.assertEqual(result["nested_archives"], 2)
            self.assertTrue((output / "a.zip.unpacked/same.mid").exists())
            self.assertTrue((output / "b.zip.unpacked/same.mid").exists())

    def test_incomplete_extraction_resumes_without_overwriting_existing_content(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source, output = root / "outer.zip", root / "output"
            with zipfile.ZipFile(source, "w") as z:
                z.writestr("a.mid", b"MThd")
                z.writestr("b.mid", b"MThd")
            output.mkdir()
            (output / "a.mid").write_bytes(b"MThd")
            (output / "b.mid.improvnet-partial").write_bytes(b"M")
            result = extract_archive(source, output, Budget(1_000_000))
            self.assertEqual(result["midi_files"], 2)
            self.assertEqual((output / "b.mid").read_bytes(), b"MThd")
            self.assertFalse((output / "b.mid.improvnet-partial").exists())

    def test_cached_parent_checks_still_reject_preexisting_symlinks(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            output = root / "output"
            output.mkdir()
            (output / "folder").symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                safe_target(output, "folder/escape.mid", {output})

    def test_appledouble_zip_metadata_is_not_a_nested_archive(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source, output = root / "outer.zip", root / "output"
            with zipfile.ZipFile(source, "w") as z:
                z.writestr("__MACOSX/._inner.zip", b"not a ZIP: resource-fork metadata")
                z.writestr("a.mid", b"MThd")
            result = extract_archive(source, output, Budget(1_000_000), nested_workers=2)
            self.assertEqual(result["midi_files"], 1)
            self.assertEqual(result["nested_archives"], 0)
            self.assertTrue((output / "__MACOSX/._inner.zip").exists())


if __name__ == "__main__":
    unittest.main()
