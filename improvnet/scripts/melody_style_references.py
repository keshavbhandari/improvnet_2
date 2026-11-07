"""Build a small jazz/blues reference catalogue with instruments and encoder buckets."""
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
from improvnet.utils.melody import MelodyTokenizer, notes_to_tokens
from improvnet.utils.melody_data import MelodyDataset
from improvnet.utils.melody_reference import PIANOS
from improvnet.utils.reference_catalog import (DEFAULT_CATALOG, CATALOG_FORMAT, record_notes,
    rhythm_summary, cosine_clusters, bucket_summaries)


def reference_candidates(data, per_source, seed):
    """Balance genre/source groups and include both solo and drum mixtures."""
    groups = defaultdict(list)
    for row in data.jazz_blues:
        if PIANOS.intersection(row.get('instruments', [])):
            groups[row['genre'], row['source']].append(row)
    rng = random.Random(seed)
    for group, rows in sorted(groups.items()):
        rng.shuffle(rows)
        drum = [r for r in rows if 'drum' in r.get('instruments', [])]
        solo = [r for r in rows if len(r.get('instruments', [])) == 1]
        # Alternation preserves diversity without requiring a drum track in solo corpora.
        pools = (drum, solo, rows)
        seen, ordered_rows = set(), []
        for index in range(len(rows)):
            for pool in pools:
                if index < len(pool):
                    row = pool[index]
                    identity = (row.get('jsonl'), row.get('offset'), row.get('path'))
                    if identity not in seen:
                        seen.add(identity)
                        ordered_rows.append(row)
        yield group, ordered_rows


def build_references(output=DEFAULT_CATALOG, per_source=4, seed=42, buckets=4, device=None):
    if per_source < 1:
        raise ValueError('Per-source count must be positive')
    data = MelodyDataset(cfg.DATA_FILE, 'train', cfg)
    encoders = FrozenConditioners(cfg.STYLE_ENCODER, cfg.BEAT_ENCODER,
                                 device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    tok = MelodyTokenizer()
    entries, style_vectors, beat_vectors, rejected, pieces = [], [], [], [], set()
    groups = list(reference_candidates(data, per_source, seed))
    for (genre, source), candidates in tqdm(groups, desc='Reference sources', unit='source'):
        accepted = 0
        for row in candidates:
            if accepted >= per_source:
                break
            identity = row.get('group') or row.get('sha256') or row.get('path')
            if identity in pieces:
                continue
            try:
                record = data.load_record(row)
                if not Path(record['path']).is_file():
                    raise FileNotFoundError(record['path'])
                notes = record_notes(record)
                if not PIANOS.intersection(n.instrument for n in notes):
                    continue
                entry_id = f'{genre}-{source}-{hashlib.sha256(str(identity).encode()).hexdigest()[:10]}'
                window_seed = int(hashlib.sha256(entry_id.encode()).hexdigest()[:8], 16)
                view = encoders.style.from_notes(notes, tok, random.Random(window_seed))
                excerpt = view.pop('_notes')
                if len(excerpt) < 12 or view['end_ms'] - view['start_ms'] < 2000:
                    raise ValueError('Reference excerpt is too short or sparse')
                ids = torch.tensor([[tok.tok_to_id[t] for t in notes_to_tokens(excerpt, tok)]], device=encoders.beat.device)
                beat = encoders.beat.encode(ids, torch.ones_like(ids, dtype=torch.bool),
                    torch.ones(1, dtype=torch.bool, device=ids.device))[0]
                locator = dict(jsonl=str(row.get('jsonl') or data.prepared), offset=row['offset'])
                entry = dict(id=entry_id, title=record.get('title') or Path(record['path']).stem,
                    genre=genre, source=source, split='train', path=record['path'], record=locator,
                    instruments=sorted({n.instrument for n in notes}),
                    instrument_note_counts=dict(sorted(Counter(n.instrument for n in notes).items())),
                    excerpt=dict(start_ms=view['start_ms'], end_ms=view['end_ms'], tokens=view['tokens'],
                        instruments=view['instruments'], seed=window_seed), rhythm=rhythm_summary(excerpt))
                entries.append(entry)
                style_vectors.append(view['solo_embedding'].cpu().tolist())
                beat_vectors.append(beat.cpu().tolist())
                pieces.add(identity)
                accepted += 1
            except (OSError, ValueError, KeyError) as error:
                rejected.append(dict(genre=genre, source=source, reason=str(error)[:200]))
    if not entries or {e['genre'] for e in entries} != {'jazz', 'blues'}:
        raise ValueError('Need usable training references for both jazz and blues')
    for kind, vectors in (('style', style_vectors), ('beat', beat_vectors)):
        for entry, label in zip(entries, cosine_clusters(vectors, buckets, seed)):
            entry[kind + '_bucket'] = f'{kind}_{label}'
    payload = dict(format=CATALOG_FORMAT, manifest=str(cfg.DATA_FILE), seed=seed,
        selection='Training-fold piano references, balanced per genre/source with solo and drum mixtures.',
        encoder_buckets=dict(style='Cosine clustering of frozen solo-piano style embeddings.',
                             beat='Cosine clustering of frozen piano/drum (or solo-piano) beat embeddings.',
                             caveat='Exploratory similarities; instrument content and tempo may affect groups. No annotated swing/shuffle labels.'),
        encoders={name:dict(checkpoint=getattr(cfg,name.upper()+'_ENCODER'),sha256=getattr(encoders,name).sha256,
                           trained_step=getattr(encoders,name).step) for name in ('style','beat')},
        coverage=dict(entries=len(entries),genre=dict(Counter(e['genre'] for e in entries)),
                      source=dict(Counter(e['source'] for e in entries))),
        buckets={kind:bucket_summaries(entries,kind+'_bucket') for kind in ('style','beat')},
        rejected_candidates=rejected, examples=entries)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_CATALOG)
    parser.add_argument('--per-source', type=int, default=4, help='References per genre/source combination')
    parser.add_argument('--buckets', type=int, default=4)
    args = parser.parse_args()
    result = build_references(args.output, args.per_source, buckets=args.buckets)
    print(f'Wrote {len(result["examples"])} references to {args.output}: {result["coverage"]}')
