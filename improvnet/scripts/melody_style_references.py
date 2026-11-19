"""Build or extend a verified jazz/blues trio catalogue with musical descriptors."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import torch
from tqdm import tqdm
from improvnet.model import melody_config as cfg
from improvnet.model.encoders.runtime import FrozenConditioners
from improvnet.utils.melody import MelodyTokenizer
from improvnet.utils.melody_data import MelodyDataset
from improvnet.utils.melody_reference import PIANOS, trio_instruments
from improvnet.utils.melody_stems import percussion_mode
from improvnet.utils.reference_catalog import (DEFAULT_CATALOG, CATALOG_FORMAT, DESCRIPTOR_POLICY,
    record_notes, rhythm_summary, rhythm_descriptors, load_catalog)


def annotate_entry(entry, notes, encoders, tok):
    """Verify actual trio notes and consecutive pairs; describe the complete trio view."""
    trio=trio_instruments(n.instrument for n in notes)
    if not trio:
        raise ValueError('Reference must contain exactly one piano, drums and acoustic/double bass, with no additional instruments')
    if not Path(entry['path']).is_file():
        raise FileNotFoundError(entry['path'])
    piano=next(instrument for instrument in trio if instrument in PIANOS)
    views=encoders.from_notes(notes,tok,mode='trio',rng=random.Random(42),
        stage_instruments={'refine':[piano],'harmonize':[piano],'drums':[piano,'drum'],'bass':sorted(trio)})['stage_conditions']
    if not all(views[stage]['drum_palette'] for stage in ('drums','bass')):
        raise ValueError('Reference has no standard drum-kit sounds')
    rhythm=rhythm_summary([n for n in notes if n.instrument in trio])
    # Old embedding-bucket and single-excerpt fields are deliberately discarded.
    result={key:entry[key] for key in ('id','title','genre','source','split','path','record')}
    result.update(instruments=sorted({n.instrument for n in notes}),
        reference_instruments=sorted(trio),percussion_mode=percussion_mode(notes),
        instrument_note_counts=dict(sorted(Counter(n.instrument for n in notes).items())),
        trio_validation=dict(reference_tokens=encoders.style.max_tokens,windows=2,instruments=sorted(trio),
            stage_candidate_pairs={stage:condition['reference_region']['candidate_pairs'] for stage,condition in views.items()}),
        rhythm=rhythm,rhythm_scope='whole_trio_view',descriptors=rhythm_descriptors(rhythm))
    return result


def entry_identity(row):
    identity=row.get('group') or row.get('sha256') or row.get('path')
    return f"{row['genre']}-{row['source']}-{hashlib.sha256(str(identity).encode()).hexdigest()[:10]}"


def make_entry(data,row,encoders,tok):
    record=data.load_record(row)
    entry=dict(id=entry_identity(row),title=record.get('title') or Path(record['path']).stem,
        genre=row['genre'],source=row['source'],split='train',path=record['path'],
        record=dict(jsonl=str(row.get('jsonl') or data.prepared),offset=row['offset']))
    return annotate_entry(entry,record_notes(record),encoders,tok)


def catalogue_payload(entries,encoders,seed=42,rejected=()):
    return dict(format=CATALOG_FORMAT,manifest=str(cfg.DATA_FILE),seed=seed,
        selection='Training-fold pieces containing exactly one piano, drums and double bass; selected by whole-piece descriptors.',
        instrument_policy='exact_three_instrument_trio',
        descriptor_policy=DESCRIPTOR_POLICY,
        encoders={name:dict(checkpoint=str(getattr(encoders,name).path),sha256=getattr(encoders,name).sha256,
                           trained_step=getattr(encoders,name).step) for name in ('style','beat')},
        coverage=dict(entries=len(entries),genre={genre:sum(e['genre']==genre for e in entries) for genre in ('jazz','blues')},
            source=dict(Counter(e['source'] for e in entries)),
            descriptors={genre:dict(Counter(d for e in entries if e['genre']==genre for d in e['descriptors']))
                         for genre in ('jazz','blues')}),
        rejected_candidates=list(rejected),examples=entries)


def write_catalogue(output,payload):
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    load_catalog(temporary)
    temporary.replace(output)


def reference_candidates(data,seed):
    groups=defaultdict(list)
    for row in data.jazz_blues:
        if trio_instruments(row.get('instruments',[])):
            groups[row['genre'],row['source']].append(row)
    rng=random.Random(seed)
    for group,rows in sorted(groups.items()):
        rng.shuffle(rows)
        yield group,rows


def build_references(output=DEFAULT_CATALOG,per_source=32,seed=42,device=None):
    if per_source<1:
        raise ValueError('Per-source count must be positive')
    data=MelodyDataset(cfg.DATA_FILE,'train',cfg)
    encoders=FrozenConditioners(cfg.STYLE_ENCODER,cfg.BEAT_ENCODER,
        device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    tok=MelodyTokenizer()
    entries,rejected,seen=[],[],set()
    for (genre,source),rows in tqdm(list(reference_candidates(data,seed)),desc='Trio reference sources',unit='source'):
        accepted=0
        for row in rows:
            if accepted>=per_source:break
            identity=entry_identity(row)
            if identity in seen:continue
            try:
                entry=make_entry(data,row,encoders,tok)
                if entry['path'] in seen:continue
                entries.append(entry);seen.update((identity,entry['path']));accepted+=1
            except (OSError,ValueError,KeyError) as error:
                rejected.append(dict(genre=genre,source=source,reason=str(error)[:200]))
    if not entries:
        raise ValueError('No verified exact-trio references available')
    payload=catalogue_payload(entries,encoders,seed,rejected)
    write_catalogue(output,payload)
    return payload


def extend_trio_references(output=DEFAULT_CATALOG,additional=32,seed=42,device=None,*,genre='jazz'):
    if genre not in ('jazz','blues') or additional<1:
        raise ValueError('Choose jazz or blues and a positive additional count')
    old=load_catalog(output)
    data=MelodyDataset(cfg.DATA_FILE,'train',cfg)
    encoders=FrozenConditioners(cfg.STYLE_ENCODER,cfg.BEAT_ENCODER,
        device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    for name in ('style','beat'):
        if getattr(encoders,name).sha256!=old['encoders'][name]['sha256']:
            raise ValueError('Catalogue encoder changed; rebuild before extending')
    groups=[rows for (g,_),rows in reference_candidates(data,seed) if g==genre]
    candidates=[rows[i] for i in range(max(map(len,groups),default=0)) for rows in groups if i<len(rows)]
    entries=list(old['examples']);initial=len(entries)
    seen={value for entry in entries for value in (entry['id'],entry['path'])}
    rejected=list(old.get('rejected_candidates',[]));tok=MelodyTokenizer()
    for row in tqdm(candidates,desc=f'Validate {genre} trios',unit='piece'):
        if len(entries)-initial>=additional:break
        if entry_identity(row) in seen:continue
        try:
            entry=make_entry(data,row,encoders,tok)
            if entry['path'] in seen:continue
            entries.append(entry);seen.update((entry['id'],entry['path']))
        except (OSError,ValueError,KeyError) as error:
            rejected.append(dict(genre=genre,source=row['source'],reason=str(error)[:200]))
    if len(entries)==initial:
        raise ValueError(f'No new usable training-fold {genre} trios; catalogue unchanged')
    payload=catalogue_payload(entries,encoders,seed,rejected)
    write_catalogue(output,payload)
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_CATALOG)
    parser.add_argument('--per-source',type=int,default=32,help='Verified trio pieces per genre/source when rebuilding')
    extension=parser.add_mutually_exclusive_group()
    extension.add_argument('--add-jazz-trios',type=int,default=0)
    extension.add_argument('--add-blues-trios',type=int,default=0)
    args=parser.parse_args()
    requested=args.add_jazz_trios or args.add_blues_trios
    result=(extend_trio_references(args.output,requested,genre='blues' if args.add_blues_trios else 'jazz')
            if requested else build_references(args.output,args.per_source))
    print(f'Wrote {len(result["examples"])} references to {args.output}: {result["coverage"]}')
