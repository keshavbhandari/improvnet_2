"""Train-fold-only automatic performance reference selection."""
from pathlib import Path
import random
from improvnet.model import melody_config
from improvnet.utils.melody import MelodyTokenizer, Note
from improvnet.utils.melody_reference import PIANOS
from improvnet.utils.melody_data import MelodyDataset
from improvnet.utils.melody_stems import kit_only_notes
from improvnet.utils.reference_catalog import DEFAULT_CATALOG, load_catalog, load_entry_notes


def choose_reference(encoder, genre, rng, *, regions='same', reference_id=None,
                     bucket=None, catalog_path=DEFAULT_CATALOG):
    if reference_id or bucket or Path(catalog_path).is_file():
        catalogue = load_catalog(catalog_path)
        rows = [e for e in catalogue['examples'] if e['genre'] == genre
                and (reference_id is None or e['id'] == reference_id)
                and (bucket is None or bucket in (e['style_bucket'], e['beat_bucket']))]
        if not rows:
            raise ValueError(f'No {genre} catalogue reference matches ID={reference_id!r}, bucket={bucket!r}')
        # Balance datasets rather than favouring whichever has the most files.
        source = rng.choice(sorted({e['source'] for e in rows}))
        rows = [e for e in rows if e['source'] == source]
        mixtures = [e for e in rows if 'drum' in e['excerpt']['instruments']]
        if mixtures and rng.random() < .75:
            rows = mixtures
        rng.shuffle(rows)
        failures = []
        for entry in rows:
            try:
                for name in ('style', 'beat'):
                    if getattr(encoder, name).sha256 != catalogue['encoders'][name]['sha256']:
                        raise ValueError('Catalogue encoder changed; rebuild the reference catalogue')
                result = encoder.from_notes(load_entry_notes(entry), MelodyTokenizer(),
                    random.Random(entry['excerpt']['seed']), regions=regions)
                result.update(id=entry['id'], path=entry['path'], title=entry['title'],
                    genre=entry['genre'], source=entry['source'], instruments=entry['instruments'],
                    rhythm=entry['rhythm'], style_bucket=entry['style_bucket'], beat_bucket=entry['beat_bucket'],
                    catalogue_excerpt=entry['excerpt'], catalogue=str(catalog_path),
                    bucket_scope='Stored excerpt; --reference-regions different changes the beat window')
                return result
            except (OSError, ValueError, KeyError) as error:
                failures.append(str(error))
        raise ValueError(f'Catalogue references unavailable: {failures[-3:]}')
    data = MelodyDataset(melody_config.DATA_FILE, 'train', melody_config)
    rows = [r for r in data.jazz_blues if r['genre'] == genre]
    preferred = [r for r in rows if 'drum' in r.get('instruments', []) and PIANOS.intersection(r.get('instruments', []))]
    rng.shuffle(preferred)
    rng.shuffle(rows)
    for row in preferred+rows:
        try:
            record = data.load_record(row)
            notes = [Note('Acoustic Piano' if n[0]=='piano' else n[0], *n[1:]) for n in record['notes']]
            notes += kit_only_notes([Note(*n) for n in record.get('drums', [])])
            result = encoder.from_notes(notes, MelodyTokenizer(), rng, regions=regions)
            result['id'] = record.get('path') or row.get('path') or str(row.get('offset'))
            return result
        except (ValueError, OSError, KeyError):
            continue
    raise ValueError('No usable training-fold piano reference; pass --style-reference')
