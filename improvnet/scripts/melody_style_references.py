"""Export aggregate controls from training performances, without note sequences."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

from improvnet.model import melody_config as cfg
from improvnet.utils.melody import Note, melody_voice
from improvnet.utils.melody_style import STYLE_FEATURES, STYLE_DEFINITIONS, performance_statistics


def build_references(data=cfg.DATA_FILE, output=cfg.STYLE_REFERENCES, per_source=8, seed=42):
    data, output = Path(data), Path(output)
    index = json.loads(data.with_suffix('.index.json').read_text())
    groups = defaultdict(list)
    for row in index['rows']:
        if row['split'] == 'train' and row['source'] not in ('wikifonia', 'curated'):
            groups[(row['genre'], row['source'])].append(row)
    examples, rng = [], random.Random(seed)
    with data.open('rb') as handle:
        for key, rows in sorted(groups.items()):
            for row in rng.sample(rows, min(per_source, len(rows))):
                handle.seek(row['offset'])
                record = json.loads(handle.readline())
                notes = [Note('Acoustic Piano' if n[0] == 'piano' else n[0], *n[1:]) for n in record['notes']]
                melody = melody_voice([notes[i] for i in record['melody_indices']])
                examples.append(dict(id=row['sha256'][:16], genre=row['genre'], source=row['source'],
                                     split='train', path=record['path'],
                                     statistics=performance_statistics(notes, melody),
                                     melody_statistics=performance_statistics(melody, melody)))
    payload = dict(version=1, feature_names=list(STYLE_FEATURES), definitions=STYLE_DEFINITIONS,
                   data_index_sha256=hashlib.sha256(data.with_suffix('.index.json').read_bytes()).hexdigest(),
                   note='Aggregate statistics only. Use melody_statistics for mono rhythm and statistics for poly rhythm/harmony.',
                   examples=examples)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + '\n')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=cfg.DATA_FILE)
    parser.add_argument('--output', type=Path, default=cfg.STYLE_REFERENCES)
    parser.add_argument('--per-source', type=int, default=8)
    args = parser.parse_args()
    result = build_references(args.data, args.output, args.per_source)
    print(f'Wrote {len(result["examples"])} training references to {args.output}')
