"""Export all PiJAMA summaries and a small selection from other performed sources."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
from tqdm import tqdm

from improvnet.model import melody_config as cfg
from improvnet.utils.melody import Note, melody_voice
from improvnet.utils.melody_groove import reference_windows
from improvnet.utils.melody_style import STYLE_FEATURES, STYLE_DEFINITIONS, performance_statistics


def summarize_reference(job):
    data, row = job
    with Path(data).open('rb') as handle:
        handle.seek(row['offset'])
        record = json.loads(handle.readline())
    notes = [Note('Acoustic Piano' if n[0] == 'piano' else n[0], *n[1:]) for n in record['notes']]
    melody = melody_voice([notes[i] for i in record['melody_indices']])
    groove=reference_windows(notes+[Note(*n) for n in record.get('drums',[])],melody)
    groove['feature_names']=list(STYLE_FEATURES)
    groove['windows']=[[w['start_ms'],w['end_ms'],*[float(f'{w["statistics"].get(k,0.):.5g}') for k in STYLE_FEATURES]]
                       for w in groove['windows'][:32]]
    return dict(id=row['sha256'][:16], genre=row['genre'], source=row['source'],
                split=row['split'], path=record['path'],
                statistics=performance_statistics(notes, melody),
                melody_statistics=performance_statistics(melody, melody),
                groove=groove)


def build_references(data=cfg.PREPARED_DATA_FILE, output=cfg.STYLE_REFERENCES, per_source=8, seed=42,
                     workers=0, progress=False):
    if workers < 0 or per_source < 1:
        raise ValueError('Workers must be nonnegative and per-source count positive')
    data, output = Path(data), Path(output)
    index = json.loads(data.with_suffix('.index.json').read_text())
    groups = defaultdict(list)
    for row in index['rows']:
        if (row['split'] == 'train' or row['source'] == 'pijama') and row['genre'] in ('jazz', 'blues') and row['source'] not in ('wikifonia', 'curated'):
            groups[(row['genre'], row['source'])].append(row)
    selected, rng = [], random.Random(seed)
    for (_, source), rows in sorted(groups.items()):
        selected.extend(rows if source == 'pijama' else rng.sample(rows, min(per_source, len(rows))))
    manager = ProcessPoolExecutor(max_workers=workers) if workers else nullcontext()
    with manager as pool:
        jobs = ((str(data), row) for row in selected)
        results = pool.map(summarize_reference, jobs, chunksize=8) if pool else map(summarize_reference, jobs)
        examples = list(tqdm(results, total=len(selected), desc='Reference statistics', unit='piece',
                             disable=not progress))
    payload = dict(version=2, feature_names=list(STYLE_FEATURES), definitions=STYLE_DEFINITIONS,
                   data_index_sha256=hashlib.sha256(data.with_suffix('.index.json').read_bytes()).hexdigest(),
                   note='Rhythm summaries only. All PiJAMA pieces retain their original split; automatic references use training entries only.',
                   coverage=dict(pijama_splits=dict(Counter(x['split'] for x in examples if x['source'] == 'pijama')),
                                 automatic_reference_split='train'),
                   examples=examples)
    output.parent.mkdir(parents=True, exist_ok=True)
    header={k:v for k,v in payload.items() if k!='examples'}
    output.write_text(json.dumps(header,indent=2)[:-2]+',\n  \"examples\": [\n'+',\n'.join(
        json.dumps(x,separators=(',',':')) for x in examples)+'\n  ]\n}\n')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=cfg.PREPARED_DATA_FILE)
    parser.add_argument('--output', type=Path, default=cfg.STYLE_REFERENCES)
    parser.add_argument('--per-source', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    result = build_references(args.data, args.output, args.per_source, workers=args.workers, progress=True)
    print(f'Wrote {len(result["examples"])} performance summaries to {args.output}; '
          f'PiJAMA splits: {result["coverage"]["pijama_splits"]}')
