"""Safely unpack MIDI collections, recursively including nested archives.

Original archives are retained. Only ordinary files/directories are extracted;
links, special files and escaping paths are rejected. Completed collections are
resumable. No shell commands or archive-supplied code are executed.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import threading
import time
import zipfile

SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".txz", ".tar", ".zip")
MARKER = ".improvnet-extraction.json"


def is_archive(path):
    # macOS AppleDouble resource forks can be named ._something.zip, but they
    # are metadata, not ZIP files. Retain them as ordinary files, never recurse.
    return not Path(path).name.startswith("._") and str(path).lower().endswith(SUFFIXES)


def safe_target(root, name, validated_parents=None):
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or any(":" in p for p in path.parts):
        raise ValueError(f"Unsafe archive path: {name!r}")
    if not path.parts or path.name == MARKER:
        raise ValueError(f"Reserved/empty archive path: {name!r}")
    target = root.joinpath(*path.parts)
    if validated_parents is None:
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Archive path escapes destination: {name!r}")
        parents = [p for p in target.parents if p != root.parent]
    else:
        # root was resolved/verified once. Archive names cannot escape it, and
        # this extractor never creates links. Cache directory checks, NOT file
        # checks, to avoid dozens of network stat calls for every tiny MIDI.
        parents = []
        for parent in target.parents:
            if parent == root:
                break
            if parent not in validated_parents:
                parents.append(parent)
    if target.is_symlink() or any(p.is_symlink() for p in parents):
        raise ValueError(f"Destination contains a link: {name!r}")
    if validated_parents is not None:
        validated_parents.update(parents)
    return target


class Budget:
    def __init__(self, max_bytes):
        self.max_bytes, self.bytes, self.members = max_bytes, 0, 0
        self.lock = threading.Lock()

    def reserve(self, size):
        with self.lock:
            if size < 0 or self.bytes + size > self.max_bytes or self.members >= 5_000_000:
                raise ValueError("Extraction byte/member safety limit exceeded")
            self.bytes += size
            self.members += 1


def copy_member(stream, target, size, budget, created_parents=None):
    budget.reserve(size)
    if created_parents is None or target.parent not in created_parents:
        target.parent.mkdir(parents=True, exist_ok=True)
        if created_parents is not None:
            created_parents.add(target.parent)
    if target.exists():
        # Compare content rather than trusting sizes or overwriting user data.
        if not target.is_file() or target.stat().st_size != size:
            raise FileExistsError(f"Conflicting existing member: {target}")
        with target.open("rb") as previous:
            while chunk := stream.read(1024 * 1024):
                if previous.read(len(chunk)) != chunk:
                    raise FileExistsError(f"Conflicting existing member: {target}")
        return
    temporary = target.with_name(target.name + ".improvnet-partial")
    if temporary.is_symlink():
        raise ValueError(f"Unsafe partial file: {temporary}")
    count = 0
    try:
        with temporary.open("wb") as handle:
            while chunk := stream.read(1024 * 1024):
                count += len(chunk)
                if count > size:
                    raise ValueError(f"Member exceeds declared size: {target}")
                handle.write(chunk)
        if count != size:
            raise ValueError(f"Truncated archive member: {target}")
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def extract_archive(source, destination, budget, depth=0, nested_workers=1):
    if depth > 8:
        raise ValueError("Nested archive depth exceeds 8")
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError("Destination must not be a symlink")
    destination = destination.resolve()
    identity = dict(path=str(source.resolve()), bytes=source.stat().st_size, mtime_ns=source.stat().st_mtime_ns)
    marker = destination / MARKER
    if marker.exists():
        completed = json.loads(marker.read_text())
        if completed["source"] != identity:
            raise FileExistsError(f"Archive changed since previous extraction: {source}")
        print(f"Already complete: {source.name}", flush=True)
        return completed
    print(f"Unpacking: {source.name} → {destination}", flush=True)
    counts = dict(files=0, midi_files=0, bytes=0, nested_archives=0, skipped_links=0)
    nested, seen = [], set()
    validated_parents, created_parents = {destination}, {destination}
    last_progress = time.monotonic()

    def accept(name, size, stream, target):
        nonlocal last_progress
        if target in seen:
            raise ValueError(f"Duplicate archive member: {name}")
        seen.add(target)
        if size > 10 * 1024**2 and shutil.disk_usage(destination).free < size + 1024**3:
            raise OSError("Insufficient free space; preserving 1 GiB reserve")
        copy_member(stream, target, size, budget, created_parents)
        counts["files"] += 1
        counts["bytes"] += size
        counts["midi_files"] += target.suffix.lower() in (".mid", ".midi")
        if is_archive(target):
            nested.append(target)
        if time.monotonic() - last_progress >= 20:
            if shutil.disk_usage(destination).free < 1024**3:
                raise OSError("Insufficient free space; preserving 1 GiB reserve")
            print(f"  {source.name}: {counts['files']:,} files; {counts['midi_files']:,} MIDI", flush=True)
            last_progress = time.monotonic()

    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            for entry in archive.infolist():
                target = safe_target(destination, entry.filename, validated_parents)
                mode = entry.external_attr >> 16
                if stat.S_ISLNK(mode):
                    counts["skipped_links"] += 1
                    continue
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    created_parents.add(target)
                    continue
                with archive.open(entry) as stream:
                    accept(entry.filename, entry.file_size, stream, target)
    else:
        with tarfile.open(source, "r|*") as archive:
            for entry in archive:
                # Common tar root entries don't contain a file.
                if entry.name in (".", "./") and entry.isdir():
                    continue
                target = safe_target(destination, entry.name, validated_parents)
                if entry.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    created_parents.add(target)
                elif entry.isfile():
                    with archive.extractfile(entry) as stream:
                        accept(entry.name, entry.size, stream, target)
                else:
                    counts["skipped_links"] += 1
    def child_job(child):
        return extract_archive(child, child.with_name(child.name + ".unpacked"), budget, depth + 1, nested_workers)

    if nested_workers > 1 and len(nested) > 1:
        with ThreadPoolExecutor(max_workers=min(nested_workers, len(nested))) as pool:
            child_results = list(pool.map(child_job, nested))
    else:
        child_results = [child_job(child) for child in nested]
    for child_result in child_results:
        for key in counts:
            counts[key] += child_result[key]
        counts["nested_archives"] += 1
    completed = dict(source=identity, **counts)
    # Marker is written only after all nested collections finish successfully.
    pending_marker = marker.with_name(MARKER + ".partial")
    with pending_marker.open("w") as handle:
        json.dump(completed, handle, indent=2)
    os.replace(pending_marker, marker)
    print(f"Complete: {source.name}: {counts['midi_files']:,} MIDI; {counts['nested_archives']} nested archives", flush=True)
    return completed


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--nested-workers", type=int, default=2)
    p.add_argument("--max-gib", type=float, default=512.)
    args = p.parse_args(argv)
    if not 1 <= args.workers <= 4 or not 1 <= args.nested_workers <= 2 or args.max_gib <= 0:
        p.error("Use 1–4 collection workers, 1–2 nested workers and a positive extraction limit")
    archives = sorted(p for p in args.input_dir.iterdir() if p.is_file() and is_archive(p))
    if not archives:
        p.error("No supported archives in input directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    budget = Budget(int(args.max_gib * 1024**3))
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(extract_archive, path, args.output_dir / path.name, budget, 0, args.nested_workers): path for path in archives}
        for future in as_completed(jobs):
            results.append(future.result())
    summary = dict(collections=sorted(results, key=lambda x: x["source"]["path"]),
                   midi_files=sum(x["midi_files"] for x in results),
                   nested_archives=sum(x["nested_archives"] for x in results))
    with (args.output_dir / "extraction_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"All done: {summary['midi_files']:,} MIDI; summary={args.output_dir/'extraction_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
