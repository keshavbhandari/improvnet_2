"""Append a seeded classical subset matching an existing jazz/blues corpus.

Existing records and splits are retained. Classical candidates come only from
the labelled MIDI index; shared source recordings and transposed melody copies
stay in one split. Exact/base-melody duplicates are excluded before publication.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import shutil
import sqlite3

from improvnet.model import melody_config as cfg
from improvnet.scripts.melody_prepare import make_record, melody_fingerprint, read_job
from improvnet.utils.melody import MelodyTokenizer


SOURCES = {'aria': 'aria_classical', 'gigamidi': 'gigamidi_classical',
           'classical_archives': 'classical_archives', 'maestro': 'maestro',
           'symphonynet': 'symphonynet_classical'}


def add_classical(base=cfg.BASE_DATA_FILE, index=cfg.CLASSICAL_INDEX,
                  output=cfg.DATA_FILE, workers=16, seed=42, limit=None):
    base, index, output = Path(base), Path(index), Path(output)
    if workers < 0 or limit is not None and limit <= 0:
        raise ValueError('Workers must be nonnegative and sample limit must be positive')
    if any(p.exists() for p in (output, output.with_suffix('.index.json'), output.with_suffix('.report.json'))):
        raise FileExistsError(f'Choose a fresh output path: {output}')
    original = json.loads(base.with_suffix('.index.json').read_text())
    if original['tokenizer'] != MelodyTokenizer().signature():
        raise ValueError('Base corpus tokenizer differs')
    if any(r['genre'] not in ('jazz', 'blues') for r in original['rows']):
        raise ValueError('Base corpus must contain only jazz/blues')
    target = len(original['rows']) if limit is None else limit
    seen = {r['sha256'] for r in original['rows']}
    base_melodies, base_paths = set(), set()
    print(f'Indexing {len(original["rows"])} existing jazz/blues performances', flush=True)
    with base.open('rb') as handle:
        for line in handle:
            record = json.loads(line)
            base_melodies.add(melody_fingerprint(record))
            base_paths.add(record['path'])
    with sqlite3.connect(index.resolve().as_uri() + '?mode=ro', uri=True) as db:
        candidates = [(Path(path), collection, group) for path, collection, group in
                      db.execute("SELECT path,collection,group_id FROM files WHERE genre='classical' ORDER BY path")
                      if collection in SOURCES and path not in base_paths]
    random.Random(seed).shuffle(candidates)
    print(f'Sampling {target} usable classical performances from {len(candidates)} labelled candidates', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = output.with_suffix('.classical.tmp')
    temporary = output.with_suffix('.jsonl.tmp')
    if cache.exists() or temporary.exists():
        raise FileExistsError('A previous preparation is incomplete; inspect its temporary files first')
    rows, parents, keys, errors = [], [], {}, []
    rejected = Counter()

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def connect(i, key):
        if key in keys:
            parents[find(i)] = find(keys[key])
        else:
            keys[key] = i

    manager = ProcessPoolExecutor(max_workers=workers) if workers else nullcontext()
    try:
        with cache.open('wb') as handle, manager as pool:
            # Bounded batches avoid submitting the whole candidate pool at once.
            for begin in range(0, len(candidates), 256):
                batch = candidates[begin:begin + 256]
                jobs = [(path, 'classical', SOURCES[source], '', False) for path, source, _ in batch]
                results = pool.map(read_job, jobs, chunksize=4) if pool else map(read_job, jobs)
                for (_, _, source_group), (job, digest, notes, error) in zip(batch, results):
                    if len(rows) >= target:
                        break
                    try:
                        if error:
                            raise ValueError(error)
                        if digest in seen:
                            rejected['duplicate_hash'] += 1
                            continue
                        record, fingerprint = make_record(job, notes)
                        if fingerprint in base_melodies:
                            rejected['base_melody_copy'] += 1
                            continue
                        i = len(rows)
                        parents.append(i)
                        connect(i, 'recording:' + source_group)
                        connect(i, 'melody:' + fingerprint)
                        rows.append(dict(offset=handle.tell(), source=record['source'], genre='classical',
                                         has_harmony=record['has_harmony'], has_drums=record['has_drums'],
                                         sha256=digest, source_group=source_group))
                        handle.write((json.dumps(record, separators=(',', ':')) + '\n').encode())
                        seen.add(digest)
                        if len(rows) % 500 == 0:
                            print(f'Prepared classical {len(rows)}/{target}; parse failures={len(errors)}', flush=True)
                    except Exception as exc:
                        errors.append(dict(path=str(job[0]), error=f'{type(exc).__name__}: {exc}'))
                if len(rows) >= target:
                    break
        if len(rows) != target:
            raise ValueError(f'Only {len(rows)} usable classical performances; need {target}')
        representatives = {}
        base_size = base.stat().st_size
        for i, row in enumerate(rows):
            root = find(i)
            representatives[root] = min(representatives.get(root, row['sha256']), row['sha256'])
        for i, row in enumerate(rows):
            group = representatives[find(i)]
            number = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:12], 16) / 16**12
            row['group'] = group
            row['split'] = ('train' if number < cfg.DATA_SPLIT_RATIOS['train'] else
                            'validation' if number < cfg.DATA_SPLIT_RATIOS['train'] + cfg.DATA_SPLIT_RATIOS['validation'] else 'test')
            row['offset'] += base_size
        combined = original['rows'] + rows
        report = dict(prepared=len(combined), jazz_blues=len(original['rows']), classical=len(rows),
                      classical_sample_seed=seed, classical_candidates=len(candidates),
                      base_data=str(base.resolve()), classical_index=str(index.resolve()),
                      base_index_sha256=hashlib.sha256(base.with_suffix('.index.json').read_bytes()).hexdigest(),
                      sources=dict(Counter(r['source'] for r in combined)),
                      genres=dict(Counter(r['genre'] for r in combined)),
                      splits=dict(Counter(r['split'] for r in combined)),
                      classical_splits=dict(Counter(r['split'] for r in rows)),
                      drum_splits=dict(Counter(r['split'] for r in combined if r.get('has_drums'))),
                      drum_genres=dict(Counter(r['genre'] for r in combined if r.get('has_drums'))),
                      rejected=dict(rejected), errors=errors, pretraining_overlap_audited=False,
                      base_splits_preserved=True,
                      grouping='Shared indexed recording IDs and complete transposed melody fingerprints')
        with temporary.open('wb') as handle, base.open('rb') as old, cache.open('rb') as new:
            shutil.copyfileobj(old, handle)
            shutil.copyfileobj(new, handle)
        temporary.replace(output)
        output.with_suffix('.index.json').write_text(json.dumps(dict(original, rows=combined)))
        output.with_suffix('.report.json').write_text(json.dumps(report, indent=2))
    finally:
        cache.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    print(json.dumps({k: v for k, v in report.items() if k != 'errors'}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=cfg.BASE_DATA_FILE)
    parser.add_argument('--index', type=Path, default=cfg.CLASSICAL_INDEX)
    parser.add_argument('--output', type=Path, default=cfg.DATA_FILE)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=cfg.DATA_SPLIT_SEED)
    args = parser.parse_args()
    add_classical(args.base, args.index, args.output, args.workers, args.seed)
