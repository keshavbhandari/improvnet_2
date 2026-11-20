"""Download expressive jazz MIDI with author splits, licenses and provenance.

JazzSAMBA ground-truth piano/drum takes form paired training mixtures. Its
automatic bass/wind estimates are retained separately, never clean labels.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import zipfile

ZENODO = 'https://zenodo.org/api/records/22943963'
JTD_URL = 'https://github.com/HuwCheston/Jazz-Trio-Database/releases/download/v02-zenodo/jazz-trio-database-v02.zip'


def request(url, byte_range=None):
    headers = {'User-Agent': 'ImprovNet-research-MIDI-acquisition/1.0'}
    if byte_range is not None:
        headers['Range'] = f'bytes={byte_range[0]}-{byte_range[1]}'
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                if byte_range is not None:
                    expected = f'bytes {byte_range[0]}-{byte_range[1]}/'
                    if response.status != 206 or not response.headers.get('Content-Range', '').startswith(expected):
                        raise ValueError('Server ignored byte range; refusing audio archive download')
                payload = response.read()
                if byte_range is not None and len(payload) != byte_range[1] - byte_range[0] + 1:
                    raise ValueError('Truncated byte-range response')
                return payload
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == 4:
                raise
            wait = min(45, 3 * 2 ** attempt)
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
                wait = min(45, max(wait, int(exc.headers.get('Retry-After', 30))))
            time.sleep(wait)


class RangeReader(io.RawIOBase):
    """Cached seekable HTTP reader; ZIP CRC verifies each extracted member."""
    def __init__(self, url, size, block_size=65536):
        super().__init__()
        self.url, self.size, self.block_size = url, int(size), block_size
        self.position, self.blocks, self.bytes_downloaded = 0, {}, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0:
            raise ValueError('Negative seek')
        self.position = position
        return position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position + size)
        parts = []
        while self.position < end:
            block = self.position // self.block_size
            start = block * self.block_size
            stop = min(self.size, start + self.block_size)
            if block not in self.blocks:
                self.blocks[block] = request(self.url, (start, stop - 1))
                self.bytes_downloaded += stop - start
            data = self.blocks[block]
            take = min(end - self.position, stop - self.position)
            parts.append(data[self.position - start:self.position - start + take])
            self.position += take
        return b''.join(parts)


def write_file(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_bytes(data)
    temporary.replace(path)


def extract_members(archive, destination, predicate):
    entries = []
    root = Path(destination).resolve()
    for info in archive.infolist():
        if info.is_dir() or not predicate(info.filename):
            continue
        path = (root / info.filename).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Archive member escapes destination')
        data = archive.read(info)
        write_file(path, data)
        entries.append(dict(path=str(path), member=info.filename, bytes=len(data),
                            sha256=hashlib.sha256(data).hexdigest(), crc32=f'{info.CRC:08x}'))
    return entries


def small_archive(asset):
    data = request(asset['links']['self'])
    kind, expected = asset['checksum'].split(':', 1)
    if hashlib.new(kind, data).hexdigest() != expected:
        raise ValueError(f'Checksum mismatch: {asset["key"]}')
    return zipfile.ZipFile(io.BytesIO(data))


def merge_performance(paths, destination):
    """Merge in performed seconds; distinct input tempo maps are respected.

    A fixed 120 BPM / 1000 PPQ output clock retains expressive timing to 0.5 ms.
    Never concatenate delta ticks from separate MIDI clocks.
    """
    import mido
    output = mido.MidiFile(type=1, ticks_per_beat=1000)
    output.tracks.append(mido.MidiTrack([mido.MetaMessage('set_tempo', tempo=500000, time=0)]))
    note_count = 0
    for label, path in paths:
        original = mido.MidiFile(str(path))
        track = mido.MidiTrack()
        channel = 9 if label == 'drums' else 0
        track.append(mido.MetaMessage('track_name', name=label, time=0))
        track.append(mido.Message('program_change', program=0, channel=channel, time=0))
        seconds, last_tick, attacks = 0., 0, 0
        for message in original:
            seconds += message.time
            if message.is_meta or not hasattr(message, 'channel') or message.type == 'program_change':
                continue
            tick = round(seconds * 2000)
            track.append(message.copy(channel=channel, time=tick-last_tick))
            last_tick = tick
            attacks += message.type == 'note_on' and message.velocity > 0
        if not attacks:
            raise ValueError(f'No notes in ground-truth {label}: {path}')
        track.append(mido.MetaMessage('end_of_track', time=0))
        output.tracks.append(track)
        note_count += attacks
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.save(str(destination))
    return note_count


def acquire_jazzsamba(root, reuse_existing=False):
    destination = root / 'raw' / 'jazzsamba'
    record = json.loads(request(ZENODO))
    assets = {entry['key']: entry for entry in record['files']}
    with small_archive(assets['JazzSAMBA-catalog.zip']) as archive:
        members = extract_members(archive, destination, lambda name: True)
    license_text = (destination / 'LICENSE').read_text()
    if 'Permission is hereby granted' not in license_text or 'ground-truth MIDI' not in license_text:
        raise ValueError('Dataset license changed; review before reuse')
    downloads = []
    for mode in ('async', 'sync'):
        asset = assets[f'JazzSAMBA-{mode}.zip']
        expected_piano = 100 if mode == 'async' else 52
        existing = list((destination / mode / 'songs').rglob('piano.mid'))
        complete = len(existing) == expected_piano and all((destination / mode / 'splits' / f'{name}.txt').exists() for name in ('train', 'val', 'test'))
        if mode == 'async':
            complete = complete and len(list((destination / mode / 'songs').rglob('drums.mid'))) == 100
        if reuse_existing and complete:
            for path in sorted((destination / mode).rglob('*')):
                if path.is_file() and '.derived.' not in path.name:
                    data = path.read_bytes()
                    members.append(dict(path=str(path), member=str(path.relative_to(destination)),
                                        bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            downloads.append(dict(archive=asset['key'], url=asset['links']['self'], archive_bytes=asset['size'],
                                  downloaded_bytes=0, validation='Reused prior ZIP-CRC-verified extracted members; SHA256 inventoried again'))
            continue
        reader = RangeReader(asset['links']['self'], asset['size'])
        with zipfile.ZipFile(reader) as archive:
            members += extract_members(archive, destination, lambda name:
                (name.endswith('.mid') and '/extras/' not in name) or '/splits/' in name or name.endswith('/bars.csv'))
        downloads.append(dict(archive=asset['key'], url=asset['links']['self'],
                              archive_bytes=asset['size'], downloaded_bytes=reader.bytes_downloaded,
                              validation='HTTP Content-Range, member CRC32, member SHA256; whole audio ZIP not fetched'))
        print(f'JazzSAMBA {mode}: {reader.bytes_downloaded:,} bytes fetched; audio skipped.', flush=True)
    with small_archive(assets['JazzSAMBA-async-derived.zip']) as archive:
        members += extract_members(archive, destination, lambda name: name.endswith('.mid') and '/extras/' not in name)
    with (destination / 'songs.csv').open(newline='') as handle:
        titles = {int(row['song_id']): row['title'] for row in csv.DictReader(handle)}
    performances = []
    for mode in ('async', 'sync'):
        splits = {}
        for name, split in [('train', 'train'), ('val', 'validation'), ('test', 'test')]:
            for line in (destination / mode / 'splits' / f'{name}.txt').read_text().splitlines():
                if line.strip() and not line.lstrip().startswith('#'):
                    splits[int(line.strip())] = split
        for take in sorted((destination / mode / 'songs').glob('*/*')):
            if take.name not in ('preferred', 'alternate'):
                continue
            piano = take / 'piano.mid'
            if not piano.exists():
                continue
            song_id = int(take.parent.name.split('-', 1)[0])
            if song_id not in splits:
                raise ValueError(f'Missing official split for {song_id}')
            stems = [('piano', piano)]
            if mode == 'async':
                drums = take / 'drums.mid'
                if not drums.exists():
                    raise ValueError(f'Missing paired ground-truth drums: {take}')
                stems.append(('drums', drums))
            mixture = destination / 'training_midi' / mode / take.parent.name / f'{take.name}.mid'
            count = merge_performance(stems, mixture)
            performances.append(dict(path=str(mixture), source='jazzsamba', genre='jazz',
                title=titles[song_id], group=f'jazzsamba:{song_id}', split=splits[song_id],
                protocol=mode, take=take.name, midi_quality='recorded_ground_truth', notes=count,
                stems=[dict(instrument=label, path=str(path), ground_truth=True) for label, path in stems]))
    return performances, dict(name='JazzSAMBA 1.0.0', source_url='https://zenodo.org/records/22943963',
        documentation_url='https://pnlong.github.io/jazzsamba-demo/about.html',
        license='JazzSAMBA original recordings and annotations: free use/copy/modify/distribute with copyright notice',
        license_path=str(destination / 'LICENSE'), raw_root=str(destination), raw_members=members, downloads=downloads,
        label_policy='Only directly recorded piano and asynchronous drum MIDI are clean labels. Derived bass/wind MIDI retained raw, excluded from mixtures.',
        performances=len(performances), split_counts=dict(Counter(p['split'] for p in performances)),
        protocol_counts=dict(Counter(p['protocol'] for p in performances)))


def acquire_jtd(root):
    destination = root / 'raw' / 'jazz_trio_database'
    data = request(JTD_URL)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = extract_members(archive, destination, lambda name:
            name.endswith('piano_midi.mid') or name.endswith('metadata.json') or Path(name).name.upper().startswith('LICENSE'))
    license_path = destination / 'LICENSE'
    if not license_path.exists():
        write_file(license_path, request('https://raw.githubusercontent.com/HuwCheston/Jazz-Trio-Database/main/LICENSE'))
    performances = []
    for path in sorted(destination.rglob('piano_midi.mid')):
        metadata_path = path.with_name('metadata.json')
        metadata = json.loads(metadata_path.read_text(), parse_constant=lambda value: None) if metadata_path.exists() else {}
        identity = metadata.get('mbz_id') or path.parent.name
        recording_urls = metadata.get('links', {}).get('external', [])
        performances.append(dict(path=str(path), source='jtd', genre='jazz',
            title=metadata.get('track_name', path.parent.name), group=f'jtd:{identity}',
            split=None, artist=metadata.get('pianist', metadata.get('musicians', {}).get('pianist')),
            album=metadata.get('album_name'), recording_id=identity,
            recording_url=recording_urls[0] if recording_urls else None,
            source_recording_key=metadata.get('fname', path.parent.name),
            midi_quality='automatically_transcribed_performance', metadata=metadata))
    return performances, dict(name='Jazz Trio Database v0.2',
        source_url='https://github.com/HuwCheston/Jazz-Trio-Database/releases/tag/v02-zenodo',
        documentation_url='https://huwcheston.github.io/Jazz-Trio-Database/installation/database-structure.html',
        license='MIT (annotations/metadata, not commercial audio)', license_path=str(license_path),
        raw_root=str(destination), raw_members=members, archive_url=JTD_URL, archive_bytes=len(data),
        archive_sha256=hashlib.sha256(data).hexdigest(),
        label_policy='Automatically transcribed performed piano only; no bass/drum MIDI or audio.',
        performances=len(performances), split_policy='No author split located; group before deterministic splitting.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--include-jtd', action='store_true')
    parser.add_argument('--reuse-existing', action='store_true', help='Reuse complete previously extracted JazzSAMBA MIDI/catalog subsets')
    args = parser.parse_args()
    root = args.data_root.resolve()
    performances, jazzsamba = acquire_jazzsamba(root, args.reuse_existing)
    datasets = [jazzsamba]
    if args.include_jtd:
        extra, jtd = acquire_jtd(root)
        performances += extra
        datasets.append(jtd)
    result = dict(version=1, performances=performances, datasets=datasets,
                  note='Author splits retained; unsplit sources require grouped splitting. No training submitted.')
    manifest = root / 'raw' / 'acquisition_manifest.json'
    write_file(manifest, json.dumps(result, indent=2).encode())
    print(f'Wrote {len(performances):,} performance records to {manifest}', flush=True)


if __name__ == '__main__':
    main()

