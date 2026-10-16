"""Build a larger, mixture-rich classical corpus for targeted stem restoration."""
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
from improvnet.scripts.melody_add_classical import SOURCES
from improvnet.scripts.melody_prepare import make_record, melody_fingerprint, read_job
from improvnet.utils.melody import MelodyTokenizer, record_instruments


def parsed_candidate(candidate):
    path, collection, group = candidate
    job, digest, notes, error = read_job((Path(path), 'classical', SOURCES[collection], '', False))
    if error:
        return None, None, None, group, error
    try:
        record, fingerprint = make_record(job, notes)
        return json.dumps(record, separators=(',', ':')).encode() + b'\n', digest, fingerprint, group, None
    except Exception as exc:
        return None, None, None, group, f'{type(exc).__name__}: {exc}'


def prepare_stems(base=cfg.BASE_DATA_FILE, cached=cfg.CLASSICAL_CACHE, index=cfg.CLASSICAL_INDEX,
                  output=cfg.DATA_FILE, ratio=cfg.CLASSICAL_RATIO,
                  multi_fraction=cfg.CLASSICAL_MULTI_FRACTION, workers=24, seed=42):
    base, cached, index, output = map(Path, (base, cached, index, output))
    if ratio <= 0 or workers < 0 or not 0 <= multi_fraction <= 1:
        raise ValueError('Invalid sampling ratio, workers or mixture fraction')
    if any(p.exists() for p in (output, output.with_suffix('.index.json'))):
        raise FileExistsError(output)
    original = json.loads(base.with_suffix('.index.json').read_text())
    signature = MelodyTokenizer().signature()
    if original['tokenizer'] != signature[:len(original['tokenizer'])]:
        raise ValueError('Base vocabulary is not a prefix of the current tokenizer')
    target = round(len(original['rows']) * ratio)
    desired_multi = round(target * multi_fraction)
    rows_base, base_fp, processed_paths = [], set(), set()
    seen = {r['sha256'] for r in original['rows']}
    print('Profiling the retained jazz/blues corpus', flush=True)
    with base.open('rb') as handle:
        for row, line in zip(original['rows'], handle):
            record = json.loads(line)
            processed_paths.add(record['path'])
            instruments = record_instruments(record)
            rows_base.append(dict(row, instruments=instruments, has_multi_instrument=len(instruments) >= 2))
            base_fp.add(melody_fingerprint(record))
    output.parent.mkdir(parents=True, exist_ok=True)
    spool = output.with_suffix('.records.tmp')
    temporary = output.with_suffix('.jsonl.tmp')
    if spool.exists() or temporary.exists():
        raise FileExistsError('An unfinished preparation already exists')
    pools, errors, rejected = {True: [], False: []}, [], Counter()
    rng = random.Random(seed)

    def store(payload, digest, fingerprint, group, handle):
        if digest in seen:
            rejected['duplicate_hash'] += 1
            return
        if fingerprint in base_fp:
            rejected['base_melody_copy'] += 1
            return
        record = json.loads(payload)
        instruments = record_instruments(record)
        # Every clean onset group must fit the shared 512-token cap.
        if max(Counter(n[3] for n in record['notes']).values()) * 4 + 2 > cfg.CHUNK_TOKENS:
            rejected['oversized_pitched_chord'] += 1
            return
        row = dict(offset=handle.tell(), sha256=digest, fingerprint=fingerprint,
                   source_group=group, genre='classical', source=record['source'],
                   has_harmony=record['has_harmony'], has_drums=bool(record['drums']),
                   instruments=instruments, has_multi_instrument=len(instruments) >= 2)
        handle.write(payload)
        pools[row['has_multi_instrument']].append(row)
        seen.add(digest)

    def enough():
        return len(pools[True]) >= desired_multi and len(pools[False]) >= target - desired_multi

    try:
        with spool.open('wb') as handle:
            if cached.exists():
                old = json.loads(cached.with_suffix('.index.json').read_text())
                with cached.open('rb') as source:
                    for row in old['rows']:
                        if row['genre'] != 'classical':
                            continue
                        source.seek(row['offset'])
                        payload = source.readline()
                        record = json.loads(payload)
                        processed_paths.add(record['path'])
                        store(payload, row['sha256'], melody_fingerprint(record), row['source_group'], handle)
                print(f'Reused classical cache: mixtures={len(pools[True])}, solo={len(pools[False])}', flush=True)
            with sqlite3.connect(index.resolve().as_uri() + '?mode=ro', uri=True) as db:
                candidates = [(p, c, g) for p, c, g in db.execute(
                    "SELECT path,collection,group_id FROM files WHERE genre='classical' ORDER BY path")
                    if c in SOURCES and p not in processed_paths and '/drums-only/' not in p]
            rng.shuffle(candidates)
            # Parse orchestral/band collections before the piano-only pools.
            preferred = [x for x in candidates if x[1] in ('gigamidi', 'classical_archives', 'symphonynet')]
            remaining = [x for x in candidates if x[1] not in ('gigamidi', 'classical_archives', 'symphonynet')]
            manager = ProcessPoolExecutor(max_workers=workers) if workers else nullcontext()
            processed = 0
            with manager as pool:
                for candidate_pool in (preferred, remaining):
                    for begin in range(0, len(candidate_pool), 128):
                        if enough():
                            break
                        batch = candidate_pool[begin:begin + 128]
                        results = pool.map(parsed_candidate, batch, chunksize=2) if pool else map(parsed_candidate, batch)
                        for candidate, (payload, digest, fingerprint, group, error) in zip(batch, results):
                            processed += 1
                            if error:
                                errors.append(dict(path=candidate[0], error=error))
                            else:
                                store(payload, digest, fingerprint, group, handle)
                        if processed % 1024 == 0:
                            print(f'Candidates {processed}: mixtures={len(pools[True])}/{desired_multi}, '
                                  f'solo={len(pools[False])}/{target - desired_multi}', flush=True)
                    if enough():
                        break
                    # After all potentially orchestral sources, use the verified
                    # attainable fraction and fill from genuine solo records.
                    if candidate_pool is preferred and len(pools[True]) < desired_multi:
                        desired_multi = len(pools[True])
                        print(f'Available verified mixtures: {desired_multi}; filling remaining slots with solo performances', flush=True)
            if len(pools[True]) + len(pools[False]) < target:
                raise ValueError('Not enough unique usable classical performances')
        multi_count = min(desired_multi, len(pools[True]))
        if len(pools[False]) < target - multi_count:
            multi_count = target - len(pools[False])
        selected = rng.sample(pools[True], multi_count) + rng.sample(pools[False], target - multi_count)
        rng.shuffle(selected)
        parents, keys = list(range(len(selected))), {}

        def find(i):
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        for i, row in enumerate(selected):
            for key in ('recording:' + row['source_group'], 'melody:' + row['fingerprint']):
                if key in keys:
                    parents[find(i)] = find(keys[key])
                else:
                    keys[key] = i
        representatives = {}
        for i, row in enumerate(selected):
            group = find(i)
            representatives[group] = min(representatives.get(group, row['sha256']), row['sha256'])
        with temporary.open('wb') as handle, base.open('rb') as old, spool.open('rb') as source:
            shutil.copyfileobj(old, handle)
            for i, row in enumerate(selected):
                group = representatives[find(i)]
                number = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:12], 16) / 16**12
                source.seek(row['offset'])
                payload = source.readline()
                row['offset'] = handle.tell()
                row['group'] = group
                row['split'] = ('train' if number < cfg.DATA_SPLIT_RATIOS['train'] else
                                'validation' if number < cfg.DATA_SPLIT_RATIOS['train'] + cfg.DATA_SPLIT_RATIOS['validation'] else 'test')
                row.pop('fingerprint')
                handle.write(payload)
        combined = rows_base + selected
        report = dict(prepared=len(combined), jazz_blues=len(rows_base), classical=len(selected),
                      classical_ratio=ratio, requested_multi_fraction=multi_fraction,
                      classical_multi_instrument=multi_count, classical_multi_fraction=multi_count / target,
                      genres=dict(Counter(r['genre'] for r in combined)),
                      splits=dict(Counter(r['split'] for r in combined)),
                      eligible_stem_splits=dict(Counter(r['split'] for r in combined if r['has_multi_instrument'])),
                      drum_genres=dict(Counter(r['genre'] for r in combined if r['has_drums'])),
                      sources=dict(Counter(r['source'] for r in combined)),
                      sample_seed=seed, base_splits_preserved=True, rejected=dict(rejected), errors=errors,
                      pretraining_overlap_audited=False)
        temporary.replace(output)
        output.with_suffix('.index.json').write_text(json.dumps(dict(
            version=4, tokenizer=signature, skyline_tolerance_ms=cfg.SKYLINE_TOLERANCE_MS, rows=combined)))
        output.with_suffix('.report.json').write_text(json.dumps(report, indent=2))
    finally:
        spool.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    print(json.dumps({k: v for k, v in report.items() if k != 'errors'}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--cache', type=Path, default=cfg.CLASSICAL_CACHE,
                        help='Reuse parsed classical records from this indexed corpus')
    args = parser.parse_args()
    prepare_stems(cached=args.cache, workers=args.workers)
