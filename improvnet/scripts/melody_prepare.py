"""Prepare raw MIDI/MusicXML once; sample corruptions later in the DataLoader."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import astuple, replace
import hashlib
import json
from itertools import islice
from pathlib import Path
import re
import sqlite3

from improvnet.model import melody_config as cfg
from improvnet.utils.melody import MelodyTokenizer, Note, ordered, read_midi_notes, skyline


def title_key(text):
    text = re.sub(r'\([^)]*\)|\[[^]]*\]', '', text.lower())
    text = re.sub(r'[-_]\d+$', '', text)  # Alternate take suffix, e.g. Weimar -2.
    key = re.sub('[^a-z0-9]', '', text)
    return key if len(key) >= 7 and not key.isdigit() else ''


def discover(root):
    """Use only known labelled sources; DTL metadata and unlabelled scores are excluded."""
    root = Path(root)
    aria = root / 'aria_jazz_blues'
    if (aria / 'manifest.jsonl').exists():
        for line in (aria / 'manifest.jsonl').open():
            row = json.loads(line)
            yield aria / row['path'], row['genre'].lower(), 'aria', '', False
    titles = {}
    db = root / 'weimar/wjazzd.db'
    if db.exists():
        with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as con:
            for performer, title in con.execute('select performer, title from solo_info'):
                titles[title_key(performer + title)] = title
    for directory, source, mono in [('weimar/raw', 'weimar', True),
                                    ('pijama-retranscribed', 'pijama', False),
                                    ('doug_mcenzie_jazz', 'doug', False)]:
        for path in sorted((root / directory).rglob('*')):
            if path.is_file() and path.suffix.lower() in ('.mid', '.midi'):
                stem = re.sub(r'_FINAL$', '', path.stem)
                title = titles.get(title_key(stem), stem.split('_', 1)[-1]) if source == 'weimar' else stem
                yield path, 'jazz', source, title, mono
    subset = root / 'wikifonia/wikimt_subset'
    manifest = subset / 'jazz_manifest.jsonl'
    if manifest.exists():
        for line in manifest.open():
            row = json.loads(line)
            yield subset / row['path'], 'jazz', 'wikifonia', row['title'], True
    # More scores can be explicitly curated without changing this script.
    extra = root / 'curated_scores.jsonl'
    if extra.exists():
        for line in extra.open():
            row = json.loads(line)
            yield root / row['path'], row['genre'].lower(), 'curated', row.get('title', ''), row.get('monophonic', True)


def read_score_notes(path, tokenizer):
    from music21 import converter, note, chord, harmony
    score = converter.parse(str(path))
    # Written repeats are performed repeats. A malformed repeat map is a parse
    # failure recorded in the report, not silently a different arrangement.
    score = score.expandRepeats()
    notes = []
    for event in score.flatten().secondsMap:
        element = event['element']
        if isinstance(element, harmony.Harmony) or not isinstance(element, (note.Note, chord.Chord)):
            continue
        pitches = element.pitches if isinstance(element, chord.Chord) else [element.pitch]
        start = max(0, round(event['offsetSeconds'] * 100) * 10)
        duration = tokenizer._quantize_dur(max(10, round(event['durationSeconds'] * 1000)))
        velocity = tokenizer._quantize_velocity(element.volume.velocity or 90)
        notes.extend(Note('Acoustic Piano', int(p.midi), velocity, start, duration) for p in pitches)
    return ordered(notes)


def read_job(job):
    path = job[0]
    try:
        tok = MelodyTokenizer()
        digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
        notes = (read_score_notes(path, tok) if path.suffix.lower() in ('.mxl', '.xml', '.musicxml')
                 else read_midi_notes(path, tok))
        return job, digest, notes, None
    except Exception as exc:
        return job, None, None, f'{type(exc).__name__}: {exc}'


def prepare(root=cfg.DATA_ROOT, output=cfg.DATA_FILE, max_files=0, workers=0):
    root, output = Path(root), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix('.index.json').exists():
        raise FileExistsError(f'Choose a fresh output path: {output}')
    tok = MelodyTokenizer()
    rows, errors, seen_hashes, parents, keys = [], [], set(), [], {}
    counts = Counter()
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def connect(i, key):
        if not key:
            return
        if key in keys:
            parents[find(i)] = find(keys[key])
        else:
            keys[key] = i
    temporary = output.with_suffix('.jsonl.tmp')
    if max_files < 0 or workers < 0:
        raise ValueError('File limit and worker count must be nonnegative')
    jobs = list(islice(discover(root), max_files)) if max_files else list(discover(root))
    manager = ProcessPoolExecutor(max_workers=workers) if workers else nullcontext()
    with temporary.open('wb') as handle, manager as pool:
        results = pool.map(read_job, jobs, chunksize=8) if pool else map(read_job, jobs)
        for (path, genre, source, title, mono), digest, notes, error in results:
            try:
                if error:
                    raise ValueError(error)
                if genre not in ('jazz', 'blues', 'classical'):
                    raise ValueError(f'Unrecognized genre {genre}')
                if digest in seen_hashes:
                    counts['duplicate_files'] += 1
                    continue
                if not notes:
                    raise ValueError('No pitched notes')
                offset = min(n.onset for n in notes)
                notes = [replace(n, onset=n.onset - offset) for n in ordered(notes)]
                melody, harmony = (notes, []) if mono else skyline(notes, cfg.SKYLINE_TOLERANCE_MS)
                selected = {id(n) for n in melody}
                melody_indices = [i for i, n in enumerate(notes) if id(n) in selected]
                has_harmony = not mono and bool(harmony)
                # Group transposed copies too. Full tune fingerprint avoids
                # grouping unrelated songs solely by a shared short lick.
                fingerprint = hashlib.sha256(json.dumps([
                    (n.pitch - melody[0].pitch, n.onset - melody[0].onset, n.duration)
                    for n in melody]).encode()).hexdigest()
                idx = len(rows)
                parents.append(idx)
                connect(idx, 'melody:' + fingerprint)
                normalized_title = title_key(title)
                connect(idx, 'title:' + normalized_title if normalized_title else '')
                record = dict(path=str(path), source=source, genre=genre, title=title,
                              notes=[astuple(n) for n in notes], melody_indices=melody_indices,
                              has_harmony=has_harmony)
                rows.append(dict(offset=handle.tell(), source=source, genre=genre,
                                 has_harmony=has_harmony, sha256=digest))
                handle.write((json.dumps(record, separators=(',', ':')) + '\n').encode())
                seen_hashes.add(digest)
                counts[source] += 1
                if len(rows) % 250 == 0:
                    print(f'Prepared {len(rows)} performances; {len(errors)} failures', flush=True)
            except Exception as exc:
                errors.append(dict(path=str(path), error=f'{type(exc).__name__}: {exc}'))
    group_members = {}
    for i, row in enumerate(rows):
        group_members.setdefault(find(i), []).append(row['sha256'])
    for i, row in enumerate(rows):
        group = min(group_members[find(i)])
        number = int(hashlib.sha256(f'{cfg.DATA_SPLIT_SEED}:{group}'.encode()).hexdigest()[:12], 16) / 16**12
        row['group'] = group
        row['split'] = ('train' if number < cfg.DATA_SPLIT_RATIOS['train'] else
                        'validation' if number < cfg.DATA_SPLIT_RATIOS['train'] + cfg.DATA_SPLIT_RATIOS['validation'] else 'test')
    index = dict(version=2, tokenizer=tok.signature(),
                 skyline_tolerance_ms=cfg.SKYLINE_TOLERANCE_MS, rows=rows)
    report = dict(prepared=len(rows), sources=dict(counts),
                  splits=dict(Counter(r['split'] for r in rows)), errors=errors,
                  excluded='DTL metadata and unlabelled Wikifonia mirror scores',
                  pretraining_overlap_audited=False)
    if not rows:
        temporary.unlink(missing_ok=True)
        raise ValueError(f'No usable records: {errors[:3]}')
    temporary.replace(output)
    output.with_suffix('.index.json').write_text(json.dumps(index))
    output.with_suffix('.report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'errors'}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=cfg.DATA_ROOT)
    parser.add_argument('--output', type=Path, default=cfg.DATA_FILE)
    parser.add_argument('--max-files', type=int, default=0, help='Limit for a preparation smoke test')
    parser.add_argument('--workers', type=int, default=cfg.NUM_WORKERS, help='CPU conversion processes; 0 runs serially')
    args = parser.parse_args()
    prepare(args.root, args.output, args.max_files, args.workers)
