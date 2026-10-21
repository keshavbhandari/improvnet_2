"""Index paths for lazy training; do not parse or tokenize the expanded corpus."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sqlite3

from improvnet.model import melody_config as cfg
from improvnet.scripts.melody_prepare import melody_fingerprint


INSTRUMENT_PATTERN=re.compile(rb'\["instrument",\s*"([^"\\]+)"\]')


def bounded_prefix(handle,offset,size=16384):
    return os.pread(handle.fileno(),size,offset).split(b'\n',1)[0]


def fingerprint_matches(job):
    path,rows,wanted=job
    found=[]
    with Path(path).open('rb') as handle:
        for row in rows:
            handle.seek(row['offset'])
            fingerprint=melody_fingerprint(json.loads(handle.readline()))
            if fingerprint in wanted:
                found.append((fingerprint,row.get('group') or row.get('source_group') or row['sha256'],row['split']))
    return found


def _link_legacy_groups(config,rows,prepared_digest):
    """Link supplemental transposed copies before assigning any held-out split."""
    path=Path(config.RECOVERED_DATA_FILE).with_suffix('.index.json')
    if not path.exists():
        return [],{},b''
    index=json.loads(path.read_text())
    if index.get('groups_linked_to_prepared_sha256')==prepared_digest:
        return index['rows'],index.get('prepared_group_aliases',{}),path.read_bytes()
    extra=index['rows'];wanted={r['fingerprint'] for r in extra}
    matches={fp:[] for fp in wanted}
    jobs=[(str(config.PREPARED_DATA_FILE),rows[i:i+512],wanted) for i in range(0,len(rows),512)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for number,found in enumerate(pool.map(fingerprint_matches,jobs)):
            for fp,group,split in found:
                matches[fp].append((group,split))
            if (number+1)%25==0:
                print(f'Checked duplicate groups in {min((number+1)*512,len(rows)):,}/{len(rows):,} prepared files.',flush=True)
    parents={}
    def find(group):
        parents.setdefault(group,group)
        while parents[group]!=group:
            group=parents[group]
        return group
    for fp in wanted:
        groups=[group for group,split in matches[fp]]+[r.get('group') or r['sha256'] for r in extra if r['fingerprint']==fp]
        root=min(find(group) for group in groups)
        for group in groups:
            parents[find(group)]=root
    aliases={group:find(group) for group in parents}
    linked=[]
    for row in extra:
        original=matches[row['fingerprint']]
        split=('train' if any(s=='train' for _,s in original) else
               'validation' if any(s=='validation' for _,s in original) else
               'test' if original else row['split'])
        linked.append(dict(row,group=find(row.get('group') or row['sha256']),split=split))
    index.update(rows=linked,prepared_group_aliases=aliases,groups_linked_to_prepared_sha256=prepared_digest,
                 matched_supplemental_records=sum(bool(matches[r['fingerprint']]) for r in extra))
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(index,separators=(',',':')))
    temporary.replace(path)
    print(f'Linked {index["matched_supplemental_records"]} supplemental records to existing composition groups.',flush=True)
    return linked,aliases,path.read_bytes()



def _extra_index_bytes(config):
    legacy = Path(config.RECOVERED_DATA_FILE).with_suffix('.index.json')
    parts = [legacy.read_bytes() if legacy.exists() else b'']
    supplemental = getattr(config, 'EXTRA_PERFORMANCE_INDEX', None)
    if supplemental and Path(supplemental).exists():
        parts.extend([b'\nV5_PERFORMANCE_SUPPLEMENT\n', Path(supplemental).read_bytes()])
    return b''.join(parts)


def link_extra_groups(config, rows, prepared_digest):
    """Keep legacy groups; add genuinely new performances with official splits.

    Supplemental copies of existing prepared/recovered performances are omitted
    instead of moving an official held-out performance into the training pool.
    The original V4 membership and supplemental source indexes are unchanged.
    """
    legacy, aliases, legacy_bytes = _link_legacy_groups(config, rows, prepared_digest)
    supplemental = getattr(config, 'EXTRA_PERFORMANCE_INDEX', None)
    if not supplemental or not Path(supplemental).exists():
        return legacy, aliases, legacy_bytes
    supplemental = Path(supplemental)
    inputs = _extra_index_bytes(config)
    signature = dict(prepared_index_sha256=prepared_digest,
                     inputs_sha256=hashlib.sha256(inputs).hexdigest())
    cache = Path(config.DATA_FILE).with_suffix('.extra_groups.json')
    if cache.exists():
        previous = json.loads(cache.read_text())
        if previous.get('signature') == signature:
            return previous['rows'], previous['aliases'], inputs
    incoming = json.loads(supplemental.read_text())['rows']
    known_hashes = {r['sha256'] for r in rows + legacy}
    known_fingerprints = {r['fingerprint'] for r in legacy}
    wanted = {r['fingerprint'] for r in incoming}
    jobs = [(str(config.PREPARED_DATA_FILE), rows[i:i+512], wanted)
            for i in range(0, len(rows), 512)]
    if wanted:
        with ProcessPoolExecutor(max_workers=4) as pool:
            for found in pool.map(fingerprint_matches, jobs):
                known_fingerprints.update(fp for fp, _, _ in found)
    added, duplicates = [], []
    for row in incoming:
        if row['sha256'] in known_hashes or row['fingerprint'] in known_fingerprints:
            duplicates.append(dict(path=row.get('path'), reason='existing_hash_or_full_melody'))
            continue
        added.append(dict(row, supplemental=True))
        known_hashes.add(row['sha256'])
        # Two new performances may share piano but have distinct recorded drum
        # takes. Their full files and official song group, rather than just the
        # skyline fingerprint, decide membership within this new corpus.
    payload = dict(signature=signature, rows=legacy+added, aliases=aliases,
                   supplemental_added=len(added), duplicates=duplicates)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, separators=(',', ':')))
    temporary.replace(cache)
    print(f'Additional performance MIDI: {len(added)} new records; '
          f'{len(duplicates)} existing copies omitted.', flush=True)
    return payload['rows'], aliases, inputs


def compact_splits(rows,ratios,seed):
    """Shrink existing held-out groups; never move previously trained music out."""
    if set(ratios)!={'train','validation','test'} or abs(sum(ratios.values())-1)>1e-8:
        raise ValueError('Split ratios must name train/validation/test and sum to one')
    groups={}
    for row in rows:
        identity=row.get('group') or row.get('source_group') or row['sha256']
        groups.setdefault(identity,[]).append(row)
    strata,totals={},Counter()
    locked_assignments={}
    for identity,members in groups.items():
        locked={r['split'] for r in members if r.get('split_locked')}
        if len(locked)>1:
            raise ValueError('One performance group has conflicting official splits')
        if locked:
            locked_assignments[identity]=next(iter(locked))
            continue
        first=members[0]
        stratum=(first['genre'],first['source'],bool(first.get('has_drums')))
        totals[stratum]+=len(members)
        splits={r['split'] for r in members}
        original='train' if 'train' in splits else 'validation' if 'validation' in splits else 'test'
        strata.setdefault((stratum,original),[]).append((identity,members))
    assignments={identity:locked_assignments.get(identity,'train') for identity in groups}
    for (stratum,split),candidates in strata.items():
        if split=='train':
            continue
        target=round(totals[stratum]*ratios[split])
        count=0
        candidates.sort(key=lambda pair:hashlib.sha256(f'{seed}:{split}:{pair[0]}'.encode()).digest())
        for identity,members in candidates:
            if abs(count+len(members)-target)<abs(count-target):
                assignments[identity]=split
                count+=len(members)
    return [dict(row,split=assignments[row.get('group') or row.get('source_group') or row['sha256']]) for row in rows]


def build_ar_mixtures(config, prepared_rows, prepared_digest):
    """Read bounded token headers, not MIDI, from the actual AR training offsets.

    Instrument eligibility is checked again on complete notes/windows in workers.
    Filename variants present in either held-out split are excluded together.
    """
    files = [Path(p) for p in config.AR_JSONL_FILES]
    if not files:
        return [], None
    from improvnet.model import ar_config
    from improvnet.utils.ar_utils import load_split_byte_index
    signature = dict(version=2, prepared_index_sha256=prepared_digest,
        files={str(p): [p.stat().st_size, p.stat().st_mtime_ns,
               p.with_name(p.stem+'_index.pkl').stat().st_mtime_ns] for p in files},
        seed=config.DATA_SPLIT_SEED, limit=config.AR_MIXTURE_LIMIT,
        scan_per_file=config.AR_MIXTURE_SCAN_PER_FILE,
        ar_split_seed=ar_config.DATA_SPLIT_SEED, ar_split_ratios=ar_config.DATA_SPLIT_RATIOS)
    index = Path(config.AR_MIXTURE_INDEX)
    if index.exists():
        cached = json.loads(index.read_text())
        if cached.get('signature') == signature:
            return cached['rows'], hashlib.sha256(index.read_bytes()).hexdigest()
        if cached.get('signature')==dict(signature,version=1):
            # Repair the selected header cache, without rescanning the entire
            # AR training corpus. A short record must not borrow its neighbour's
            # instrument names from the fixed-size read.
            repaired=[]
            for path in files:
                candidates=sorted((r for r in cached['rows'] if r['jsonl']==str(path)),key=lambda r:r['offset'])
                with path.open('rb') as handle,ThreadPoolExecutor(max_workers=16) as pool:
                    prefixes=pool.map(lambda r:bounded_prefix(handle,r['offset']),candidates)
                    for row,prefix in zip(candidates,prefixes):
                        instruments=sorted({i.decode() for i in INSTRUMENT_PATTERN.findall(prefix)})
                        if set(instruments)-{'drum'} and len(instruments)>=2:
                            repaired.append(dict(row,instruments=instruments))
            random.Random(config.DATA_SPLIT_SEED).shuffle(repaired)
            cached.update(signature=signature,rows=repaired,
                          repaired_header_count=len(cached['rows'])-len(repaired),
                          eligibility='Pitched class plus another class in this record only; full window verified on demand.',
                          counts=dict(Counter(r['source'] for r in repaired)))
            temporary=index.with_suffix('.tmp')
            temporary.write_text(json.dumps(cached,separators=(',',':')))
            temporary.replace(index)
            print(f'Repaired AR headers: retained {len(repaired):,}; removed {cached["repaired_header_count"]:,} false mixture candidates.',flush=True)
            return repaired,hashlib.sha256(index.read_bytes()).hexdigest()
    path_pattern = re.compile(rb'"midi_filepath"\s*:\s*("(?:[^"\\]|\\.)*")')
    note_path_pattern = re.compile(rb'"path"\s*:\s*("(?:[^"\\]|\\.)*")')
    instrument_pattern = INSTRUMENT_PATTERN
    genre_pattern = re.compile(rb'"genre"\s*:\s*(null|"(?:[^"\\]|\\.)*")')
    protected = set()
    prepared = Path(config.PREPARED_DATA_FILE)
    with prepared.open('rb') as handle, ThreadPoolExecutor(max_workers=16) as pool:
        offsets = [r['offset'] for r in prepared_rows if r['genre'] in ('jazz','blues') or r['split']!='train']
        for prefix in pool.map(lambda offset: os.pread(handle.fileno(),1024,offset).split(b'\n',1)[0],offsets):
            match = note_path_pattern.search(prefix)
            if match:
                protected.add(Path(json.loads(match[1])).stem.lower())
    splits = {}
    for path in files:
        splits[path] = {split: load_split_byte_index(str(path),split,ar_config)
                        for split in ('train','validation','test')}
        held_out = splits[path]['validation']+splits[path]['test']
        with path.open('rb') as handle, ThreadPoolExecutor(max_workers=16) as pool:
            for prefix in pool.map(lambda offset: os.pread(handle.fileno(),1024,offset).split(b'\n',1)[0],held_out):
                match = path_pattern.search(prefix)
                if match:
                    protected.add(Path(json.loads(match[1])).stem.lower())
    rows, seen, scanned = [], set(), Counter()
    for path in files:
        offsets = splits[path]['train']
        selected = sorted(random.Random(config.DATA_SPLIT_SEED).sample(offsets,min(len(offsets),config.AR_MIXTURE_SCAN_PER_FILE)))
        print(f'Inspecting {len(selected):,} AR train headers from {path.name}.',flush=True)
        with path.open('rb') as handle, ThreadPoolExecutor(max_workers=16) as pool:
            prefixes = pool.map(lambda offset: os.pread(handle.fileno(),16384,offset).split(b'\n',1)[0],selected)
            for offset,prefix in zip(selected,prefixes):
                scanned[path.name]+=1
                match = path_pattern.search(prefix)
                if not match:
                    continue
                midi = json.loads(match[1])
                identity = Path(midi).stem.lower()
                instruments = sorted({i.decode() for i in instrument_pattern.findall(prefix)})
                if (identity in protected or identity in seen or '/drums-only/' in midi
                        or not set(instruments)-{'drum'} or len(instruments)<2):
                    continue
                match = genre_pattern.search(prefix)
                genre = (json.loads(match[1]) if match else None) or 'unknown'
                if genre in ('jazz','blues'):
                    continue
                seen.add(identity)
                rows.append(dict(kind='ar',jsonl=str(path),offset=offset,path=midi,
                    source='ar_gigamidi' if 'gigamidi' in path.name else 'ar_misc',genre=genre,
                    instruments=instruments,has_multi_instrument=True,ar_train=True))
    rng = random.Random(config.DATA_SPLIT_SEED)
    rng.shuffle(rows)
    rows = rows[:config.AR_MIXTURE_LIMIT]
    payload = dict(signature=signature,rows=rows,scanned=dict(scanned),
                   eligibility='At least two pitched classes in token header; full window verified on demand.',
                   counts=dict(Counter(r['source'] for r in rows)))
    index.parent.mkdir(parents=True,exist_ok=True)
    temporary = index.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload,separators=(',',':')))
    temporary.replace(index)
    print(f'Indexed {len(rows):,} unique AR training mixture candidates: {payload["counts"]}',flush=True)
    return rows,hashlib.sha256(index.read_bytes()).hexdigest()


def resolve_ar_midi_paths(rows,config):
    """Recover original GigaMIDI/SymphonyNet sources and real drum attributes."""
    wanted={Path(r['path']).stem.lower() for r in rows if r['source']=='ar_gigamidi'}
    paths={}
    if wanted:
        with sqlite3.connect(Path(config.CLASSICAL_INDEX).resolve().as_uri()+'?mode=ro',uri=True) as db:
            for path,group in db.execute("SELECT path,group_id FROM files WHERE collection='gigamidi'"):
                identity=group.removeprefix('gigamidi:').lower()
                if identity not in wanted or '/drums-only/' in path:
                    continue
                if identity not in paths or '/all-instruments-with-drums/' in path:
                    paths[identity]=path
    symphony_wanted={r['path'].rsplit('/SymphonyNet_Dataset/',1)[1].lower()
                     for r in rows if '/SymphonyNet_Dataset/' in r['path']}
    symphony={}
    if symphony_wanted:
        with sqlite3.connect(Path(config.CLASSICAL_INDEX).resolve().as_uri()+'?mode=ro',uri=True) as db:
            for (path,) in db.execute("SELECT path FROM files WHERE collection='symphonynet'"):
                suffix=path.rsplit('/SymphonyNet_Dataset/',1)[-1].lower()
                if suffix in symphony_wanted:
                    symphony.setdefault(suffix,set()).add(path)
    result=[]
    for row in rows:
        raw=paths.get(Path(row['path']).stem.lower()) if row['source']=='ar_gigamidi' else None
        if raw is None and '/SymphonyNet_Dataset/' in row['path']:
            matches=symphony.get(row['path'].rsplit('/SymphonyNet_Dataset/',1)[1].lower(),set())
            if len(matches)==1:
                raw=next(iter(matches))
        if raw is None and Path(row['path']).is_file():
            raw=row['path']
        result.append(dict(row,raw_path=raw) if raw else row)
    print(f'Resolved original MIDI for {sum(bool(r.get("raw_path")) for r in result):,} AR mixtures.',flush=True)
    return result


def attach_ar_mixtures(payload,config,rows,digest,indexed=None):
    mixtures,ar_digest = indexed if indexed is not None else build_ar_mixtures(config,rows,digest)
    mixtures=resolve_ar_midi_paths(mixtures,config)
    extra,aliases,extra_bytes=link_extra_groups(config,rows,digest)
    rows=[dict(r,group=aliases.get(r.get('group') or r.get('source_group') or r['sha256'],
                                 r.get('group') or r.get('source_group') or r['sha256'])) for r in rows]
    effective=compact_splits(rows+extra,config.DATA_SPLIT_RATIOS,config.DATA_SPLIT_SEED)
    overrides={str(r.get('jsonl',''))+':'+str(r['offset']):r['split'] for r in effective}
    # Only failures rechecked under the current recovery reader are excluded.
    # File signatures let repaired files re-enter on a later manifest rebuild.
    unusable={}
    for failed in (Path(config.PARSED_CACHE)/'rejected_v2').glob('*/*.json'):
        try:
            item=json.loads(failed.read_text());source=Path(item['source']);stat=source.stat()
            if source.suffix.lower() in ('.mid','.midi') and item['signature']==[stat.st_size,stat.st_mtime_ns]:
                unusable[str(source)]=item
        except (OSError,ValueError,KeyError):
            continue
    payload['raw_train']=[r for r in payload['raw_train'] if r['path'] not in unusable]
    extra_paths={r.get('path') for r in extra}
    mixtures=[r for r in mixtures if r.get('raw_path') not in unusable and r.get('raw_path') not in extra_paths]
    protected=set(payload['protected_hashes'])|{r['sha256'] for r in extra}
    fingerprints=set(payload['protected_melody_fingerprints'])|{r['fingerprint'] for r in extra}
    counts=Counter(r['split'] for r in effective if r['genre'] in ('jazz','blues'))
    prepared_counts=Counter(r['split'] for r in effective)
    payload.update(prepared_extra_rows=extra,prepared_extra_index_sha256=hashlib.sha256(extra_bytes).hexdigest(),
                   prepared_group_aliases=aliases,
                   prepared_split_overrides=overrides,prepared_split_ratios=config.DATA_SPLIT_RATIOS,
                   prepared_record_splits=dict(prepared_counts),jazz_blues_splits=dict(counts),
                   protected_hashes=sorted(protected),protected_melody_fingerprints=sorted(fingerprints),
                   train_samples_per_epoch=(config.CLASSICAL_RATIO+1)*counts['train'],
                   candidate_non_jazz_train=sum(r['split']=='train' and r['genre'] not in ('jazz','blues') for r in effective)+len(payload['raw_train']),
                   excluded_unusable_sources=unusable,data_policy_revision=4,
                   split_policy='Compact existing held-out composition groups; never hold out previously trained groups; one evaluation example per file.')
    supplemental=[r for r in effective if r.get('supplemental')]
    payload['supplemental_performances']=dict(
        records=len(supplemental), sources=dict(Counter(r['source'] for r in supplemental)),
        splits=dict(Counter(r['split'] for r in supplemental)),
        drum_records=sum(bool(r.get('has_drums')) for r in supplemental))
    payload.update(version=6,ar_mixtures=mixtures,ar_mixture_index_sha256=ar_digest,
                   ar_source_resolution_version=2)
    output = Path(config.DATA_FILE)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload,separators=(',',':')))
    temporary.replace(output)
    print(f'Prepared file splits: {dict(prepared_counts)}; jazz/blues: {dict(counts)}; '
          f'AR mixtures={len(mixtures):,}; excluded verified unusable sources={len(unusable)}.',flush=True)
    return payload


def build_manifest(config=cfg):
    output = Path(config.DATA_FILE)
    prepared = Path(config.PREPARED_DATA_FILE)
    index = prepared.with_suffix('.index.json')
    digest = hashlib.sha256(index.read_bytes()).hexdigest()
    if output.exists():
        payload = json.loads(output.read_text())
        if (payload['prepared_index_sha256'] != digest or payload['ratio'] != config.CLASSICAL_RATIO
                or payload['seed'] != config.DATA_SPLIT_SEED):
            raise ValueError('Existing lazy manifest differs from this run')
        rows = json.loads(index.read_text())['rows']
        indexed = build_ar_mixtures(config,rows,digest)
        extra_digest=hashlib.sha256(_extra_index_bytes(config)).hexdigest()
        if (payload.get('version')==6 and payload.get('ar_source_resolution_version')==2
                and payload.get('ar_mixture_index_sha256')==indexed[1]
                and payload.get('data_policy_revision')==4
                and payload.get('prepared_split_ratios')==config.DATA_SPLIT_RATIOS
                and payload.get('prepared_extra_index_sha256')==extra_digest):
            return payload
        return attach_ar_mixtures(payload,config,rows,digest,indexed)
    rows = json.loads(index.read_text())['rows']
    # Preserve the existing curated/round-one splits. New candidates are added
    # to training only; existing validation/test pieces remain held out.
    known_groups = {r['source_group'] for r in rows if r.get('source_group')}
    protected_hashes = {r['sha256'] for r in rows if r['genre'] in ('jazz', 'blues') or r['split'] != 'train'}
    fingerprints = set()
    print('Building copy protection from existing jazz/blues and held-out records (no new MIDI parsing).', flush=True)
    with prepared.open('rb') as f:
        for row in rows:
            if row['genre'] in ('jazz', 'blues') or row['split'] != 'train':
                f.seek(row['offset'])
                fingerprints.add(melody_fingerprint(json.loads(f.readline())))
    raw, seen = [], set()
    with sqlite3.connect(Path(config.CLASSICAL_INDEX).resolve().as_uri() + '?mode=ro', uri=True) as db:
        query = """SELECT path,collection,genre,group_id FROM files
                   WHERE split='train' AND genre NOT IN ('jazz','blues','unknown')
                   AND genres NOT LIKE '%"jazz"%' AND genres NOT LIKE '%"blues"%'
                   ORDER BY group_id, path"""
        for path, collection, genre, group in db.execute(query):
            if group in known_groups or group in seen or '/drums-only/' in path:
                continue
            seen.add(group)
            raw.append(dict(path=path, source=collection, genre=genre, group=group,
                            mixture_hint=collection in ('gigamidi', 'classical_archives', 'symphonynet')))
    counts = Counter(r['split'] for r in rows if r['genre'] in ('jazz', 'blues'))
    non_jazz_train = sum(r['split'] == 'train' and r['genre'] not in ('jazz', 'blues') for r in rows) + len(raw)
    target = config.CLASSICAL_RATIO * counts['train']
    if non_jazz_train < target:
        raise ValueError(f'Only {non_jazz_train} candidate non-jazz records; need {target}')
    payload = dict(version=5, prepared=str(prepared), prepared_index_sha256=digest,
                   sql_index=str(config.CLASSICAL_INDEX), ratio=config.CLASSICAL_RATIO,
                   seed=config.DATA_SPLIT_SEED, raw_train=raw,
                   protected_hashes=sorted(protected_hashes), protected_melody_fingerprints=sorted(fingerprints),
                   jazz_blues_splits=dict(counts), candidate_non_jazz_train=non_jazz_train,
                   train_samples_per_epoch=(config.CLASSICAL_RATIO + 1) * counts['train'],
                   split_policy='Retain all prepared splits; add only indexed training groups; reject protected copies on first parse.')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, separators=(',', ':')))
    temporary.replace(output)
    print(json.dumps({k: v for k, v in payload.items() if k not in
                      ('raw_train', 'protected_hashes', 'protected_melody_fingerprints')}, indent=2), flush=True)
    print(f'Lazy raw candidates: {len(raw)}; manifest: {output}', flush=True)
    return attach_ar_mixtures(payload,config,rows,digest)


if __name__ == '__main__':
    build_manifest()

