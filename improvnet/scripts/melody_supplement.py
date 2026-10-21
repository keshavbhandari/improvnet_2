"""Index newly acquired performance MIDI for V5; task windows stay dynamic.

Only this new corpus is read once. Existing prepared data and its split index
are reused. Raw files, licenses, official song groups and source splits remain
in the acquisition manifest; corruption and skyline extraction in training are
still sampled by the DataLoader.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from improvnet.model import melody_config as cfg
from improvnet.scripts.melody_prepare import make_record, read_job
from improvnet.utils.melody import record_instruments


def prepare_supplement(manifest_path, output_index, seed=42):
    manifest_path, output_index = Path(manifest_path), Path(output_index)
    acquisition = json.loads(manifest_path.read_text())
    rows = acquisition['performances']
    signature = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    output_data = output_index.with_name(output_index.name.removesuffix('.index.json')+'.jsonl')
    if output_index.exists():
        previous = json.loads(output_index.read_text())
        if previous.get('acquisition_sha256') == signature and output_data.exists():
            return previous
        raise FileExistsError('Supplement differs; use a fresh index rather than replacing training data')
    output_index.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_data.with_suffix('.tmp')
    indexed, errors, duplicates, seen = [], [], [], set()
    group_splits = {}
    with temporary.open('wb') as handle:
        for item in rows:
            path = Path(item['path'])
            group = item.get('group') or f"{item['source']}:{path.stem}"
            split = item.get('split')
            if split is None:
                number = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:12],16)/16**12
                split = 'train' if number < .98 else 'validation' if number < .99 else 'test'
            if split not in ('train','validation','test'):
                raise ValueError(f'Invalid split for {path}: {split}')
            if group in group_splits and group_splits[group] != split:
                raise ValueError(f'Performance takes cross official splits: {group}')
            group_splits[group] = split
            job, digest, notes, error = read_job((path, item.get('genre','jazz'), item['source'],
                                                  item.get('title',path.stem), False))
            try:
                if error:
                    raise ValueError(error)
                if digest in seen:
                    duplicates.append(str(path))
                    continue
                record, fingerprint = make_record(job,notes)
                instruments = record_instruments(record)
                indexed.append(dict(path=str(path), jsonl=str(output_data), offset=handle.tell(),
                    source=item['source'], genre=item.get('genre','jazz'), group=group, split=split,
                    split_locked=True, sha256=digest, fingerprint=fingerprint, instruments=instruments,
                    has_drums=record['has_drums'], has_harmony=record['has_harmony'],
                    has_multi_instrument=len(instruments)>=2))
                handle.write((json.dumps(record,separators=(',',':'))+'\n').encode())
                seen.add(digest)
            except (ValueError,OSError) as exc:
                errors.append(dict(path=str(path),error=str(exc)))
    if not indexed:
        temporary.unlink(missing_ok=True)
        raise ValueError('No usable pitched performance MIDI in the acquisition manifest')
    temporary.replace(output_data)
    payload = dict(version=1, acquisition=str(manifest_path), acquisition_sha256=signature,
                   data=str(output_data), rows=indexed,
                   counts=dict(sources=dict(Counter(r['source'] for r in indexed)),
                               splits=dict(Counter(r['split'] for r in indexed)),
                               drum_records=sum(r['has_drums'] for r in indexed)),
                   errors=errors, duplicate_paths=duplicates,
                   policy='Official groups/splits locked; legacy-copy filtering occurs at manifest assembly')
    temporary_index=output_index.with_suffix('.tmp')
    temporary_index.write_text(json.dumps(payload,separators=(',',':')))
    temporary_index.replace(output_index)
    print(json.dumps({k:v for k,v in payload.items() if k!='rows'},indent=2),flush=True)
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output-index',type=Path,default=cfg.EXTRA_PERFORMANCE_INDEX)
    args=parser.parse_args()
    prepare_supplement(args.manifest,args.output_index,cfg.DATA_SPLIT_SEED)


if __name__=='__main__':
    main()
