"""Descriptor-based selection from the verified training-fold trio catalogue."""
from improvnet.utils.melody import MelodyTokenizer
from improvnet.utils.melody_reference import reference_instruments, trio_instruments
from improvnet.utils.reference_catalog import DEFAULT_CATALOG, load_catalog, load_entry_notes, parse_reference_description


def choose_reference(encoder,genre,rng,*,description='any',catalog_path=DEFAULT_CATALOG,
                     mode='trio',stage_instruments=None,aggregation='window'):
    description=parse_reference_description(description)
    requested=set() if description=='any' else set(description.split())
    catalogue=load_catalog(catalog_path)
    rows=[entry for entry in catalogue['examples'] if entry['genre']==genre
          and requested<=set(entry['descriptors'])]
    if not rows:
        available=sorted({' '.join(e['descriptors']) for e in catalogue['examples'] if e['genre']==genre})
        raise ValueError(f'No {genre} trio reference matches "{description}". Available combinations: {available}; use another description or --style-reference')
    if stage_instruments:
        rows=[entry for entry in rows if all(
            (wanted:=reference_instruments(target,'exact' if stage=='refine' else mode,entry['instruments']))
            and wanted<=set(entry['instruments']) for stage,target in stage_instruments.items())]
        if not rows:
            raise ValueError(f'No {genre} "{description}" trio reference supports the requested stage instruments: {stage_instruments}; use another description or a compatible --style-reference')
    for name in ('style','beat'):
        if getattr(encoder,name).sha256!=catalogue['encoders'][name]['sha256']:
            raise ValueError('Catalogue encoder changed; rebuild the reference catalogue')
    sources=sorted({entry['source'] for entry in rows})
    rng.shuffle(sources)
    failures=[]
    for source in sources:
        candidates=[entry for entry in rows if entry['source']==source]
        rng.shuffle(candidates)
        for entry in candidates:
            try:
                notes=load_entry_notes(entry)
                if not trio_instruments(n.instrument for n in notes):
                    raise ValueError('Reference notes must contain exactly one piano, drums and double bass, with no extra instruments')
                result=encoder.from_notes(notes,MelodyTokenizer(),
                    mode=mode,stage_instruments=stage_instruments,aggregation=aggregation,rng=rng)
                result.update(id=entry['id'],path=entry['path'],title=entry['title'],genre=entry['genre'],
                    source=entry['source'],instruments=entry['instruments'],rhythm=entry['rhythm'],
                    descriptors=entry['descriptors'],requested_description=description,selection_method='description',catalogue=str(catalog_path),
                    selection_scope='Whole-piece trio descriptors select the piece; encoders use consecutive reference pairs')
                return result
            except (OSError,ValueError,KeyError) as error:
                failures.append(f"{entry['id']}: {error}")
    raise ValueError(f'No usable {genre} "{description}" reference across eligible sources: {failures[-3:]}')


def reference_report(value):
    """Remove embeddings/internal fields recursively before serializing an inference report."""
    if isinstance(value,dict):
        return {k:reference_report(v) for k,v in value.items() if not hasattr(v,'detach') and not k.startswith('_')}
    if isinstance(value,(list,tuple)):
        return [reference_report(v) for v in value]
    return value
