"""Current V5 data contracts."""
import json
import mido
import pytest
import torch
import io
import zipfile
from collections import Counter
from improvnet.scripts.melody_manifest import bounded_prefix, compact_splits, INSTRUMENT_PATTERN, link_extra_groups, build_manifest
from improvnet.scripts.melody_prepare import melody_fingerprint
from improvnet.utils.melody import read_midi_notes, Note, write_midi_notes
from improvnet.utils.melody_midi import read_midi_dict
from improvnet.model.melody_train import report_data_stats
from improvnet.utils.melody_data import DATA_STAT_NAMES, MelodyDataset
from melody_test_helpers import corpus, mix, response, attrs
from unittest.mock import MagicMock
from improvnet.scripts import melody_download_jazz as download
from pathlib import Path
from improvnet.scripts.melody_supplement import prepare_supplement


def midi_fixture(path,broken=False,invalid_note=False):
    midi=mido.MidiFile(ticks_per_beat=480)
    track=mido.MidiTrack([
        mido.MetaMessage('set_tempo',tempo=500000),
        mido.Message('program_change',channel=0,program=0),
        (mido.UnknownMetaMessage(0x59,data=[0,255],time=120) if broken else
         mido.MetaMessage('key_signature',key='C',time=120)),
        mido.Message('sysex',data=[1,128 if broken else 2],time=60,skip_checks=broken),
        mido.Message('note_on',channel=0,note=60,velocity=90,time=60),
        mido.Message('control_change',control=64,value=127,time=60),
        mido.Message('note_off',channel=0,note=60,velocity=0,time=180),
        mido.Message('control_change',control=64,value=0,time=120),
        mido.MetaMessage('set_tempo',tempo=400000,time=120),
        mido.Message('note_on',channel=0,note=200 if invalid_note else 64,velocity=80,
                     time=120,skip_checks=invalid_note),
        mido.Message('note_off',channel=0,note=64,velocity=0,time=240),
    ])
    midi.tracks.append(track)
    midi.save(path)


def test_metadata_recovery_preserves_notes_tempo_pedal_and_delta_times(tmp_path):
    clean,broken=tmp_path/'clean.mid',tmp_path/'broken.mid'
    midi_fixture(clean);midi_fixture(broken,broken=True)
    recovery={}
    assert read_midi_notes(broken,recovery=recovery)==read_midi_notes(clean)
    assert recovery=={'invalid_key_signature':1,'invalid_sysex':1}
    a,b=read_midi_dict(clean),read_midi_dict(broken)
    for attr in ('note_msgs','tempo_msgs','pedal_msgs','instrument_msgs','ticks_per_beat'):
        assert getattr(a,attr)==getattr(b,attr)


def test_recovery_never_clips_a_corrupt_note(tmp_path):
    path=tmp_path/'corrupt_note.mid'
    midi_fixture(path,broken=True,invalid_note=True)
    with pytest.raises(Exception):
        read_midi_notes(path)


def test_fixed_size_header_read_cannot_borrow_neighbour_instruments(tmp_path):
    path=tmp_path/'records.jsonl'
    first=json.dumps(dict(instrument=['instrument','drum'])).encode()+b'\n'
    second=json.dumps(dict(instrument=['instrument','Acoustic Piano'],other=['instrument','Electric Bass'])).encode()+b'\n'
    path.write_bytes(first+second)
    with path.open('rb') as handle:
        assert set(INSTRUMENT_PATTERN.findall(bounded_prefix(handle,0)))=={b'drum'}
        assert set(INSTRUMENT_PATTERN.findall(bounded_prefix(handle,len(first))))=={b'Acoustic Piano',b'Electric Bass'}


def test_compact_splits_reclaim_held_out_groups_without_holding_out_seen_music():
    rows=[]
    for i in range(1000):
        split='train' if i<900 else 'validation' if i<950 else 'test'
        for variant in range(2):
            rows.append(dict(offset=2*i+variant,group=str(i),sha256=str(i),source='fixture',genre='jazz',split=split))
    compact=compact_splits(rows,dict(train=.98,validation=.01,test=.01),42)
    assert Counter(r['split'] for r in compact)==dict(train=1960,validation=20,test=20)
    assert all(r['split']=='train' for r in compact if int(r['group'])<900)
    assert compact==compact_splits(rows,dict(train=.98,validation=.01,test=.01),42)
    for split in ('train','validation','test'):
        groups={r['group'] for r in compact if r['split']==split}
        assert all((r['group'] in groups)==(r['split']==split) for r in compact)


def test_evaluation_visits_each_prepared_file_once_without_training_ratio(tmp_path,monkeypatch):
    config,_=corpus(tmp_path,ratio=8)
    dataset=MelodyDataset(config.DATA_FILE,'validation',config)
    assert len(dataset)==len(dataset.eval_rows)==2
    loaded=[]
    monkeypatch.setattr(dataset,'load_record',lambda row:loaded.append(row) or mix(20,row['genre']))
    assert len([dataset[i] for i in range(len(dataset))])==2
    assert loaded==dataset.eval_rows
    assert len({r['offset'] for r in loaded})==2


def test_retry_percentage_uses_consumed_attempts_and_reports_routing_separately():
    messages=[]
    counts=dict(attempts=110,retries=10,routed=20,malformed=6,protected=4)
    stats=torch.tensor([counts.get(k,0) for k in DATA_STAT_NAMES])
    metrics=report_data_stats(stats,'train',messages.append)
    assert metrics['data_retry_percent']==pytest.approx(1000/110)
    assert metrics['data_routed_percent']==20
    assert 'retries=10/110 (9.09% of attempts)' in messages[0]
    assert 'routed=20/100 (20.00% of accepted)' in messages[0]
    assert 'malformed=6' in messages[0] and 'protected=4' in messages[0]


def test_new_transposed_copies_inherit_existing_held_out_groups(tmp_path,monkeypatch):
    from types import SimpleNamespace
    base=tmp_path/'base.jsonl';extra_path=tmp_path/'extra.jsonl'
    rows=[];records=[]
    with base.open('wb') as handle:
        for i,split in enumerate(('train','validation','test')):
            record=mix(10)
            record['notes']=[list(n[:4])+[100+100*i] for n in record['notes']]
            records.append(record)
            rows.append(dict(offset=handle.tell(),group='old_'+split,sha256=str(i),
                             split=split,genre='jazz',source='fixture'))
            handle.write((json.dumps(record)+'\n').encode())
    supplemental=[]
    for i in (1,2):
        record=dict(records[i],notes=[[n[0],n[1]+12,*n[2:]] for n in records[i]['notes']])
        supplemental.append(dict(fingerprint=melody_fingerprint(record),group='new_'+str(i),
                                 sha256='new_'+str(i),split='train',source='fixture',genre='jazz',offset=i,jsonl=str(extra_path)))
    extra_path.with_suffix('.index.json').write_text(json.dumps(dict(rows=supplemental)))
    config=SimpleNamespace(PREPARED_DATA_FILE=base,RECOVERED_DATA_FILE=extra_path)
    linked,aliases,_=link_extra_groups(config,rows,'base_digest')
    assert [r['split'] for r in linked]==['validation','test']
    combined=[dict(r,group=aliases.get(r['group'],r['group'])) for r in rows]+linked
    compact=compact_splits(combined,dict(train=1/3,validation=1/3,test=1/3),42)
    assert compact[1]['split']==compact[3]['split']=='validation'
    assert compact[2]['split']==compact[4]['split']=='test'
    monkeypatch.setattr('improvnet.scripts.melody_manifest.fingerprint_matches',lambda job:pytest.fail('Must reuse group links'))
    assert link_extra_groups(config,rows,'base_digest')[:2]==(linked,aliases)


def timed_messages(path):
    seconds, result = 0., []
    for message in mido.MidiFile(path):
        seconds += message.time
        if message.type in ('note_on', 'note_off', 'control_change'):
            attrs = message.dict()
            attrs.pop('time')
            attrs.pop('channel')
            result.append((attrs, seconds, message.channel))
    return result


def performance(path, channel, tempo, pitch):
    midi = mido.MidiFile(type=1, ticks_per_beat=960)
    clock = mido.MidiTrack([
        mido.MetaMessage('set_tempo', tempo=tempo),
        mido.MetaMessage('set_tempo', tempo=tempo+187321, time=1321),
    ])
    notes = mido.MidiTrack([
        mido.Message('program_change', channel=channel, program=42),
        mido.Message('control_change', channel=channel, control=64, value=127, time=101),
        mido.Message('note_on', channel=channel, note=pitch, velocity=87, time=1136),
        mido.Message('note_off', channel=channel, note=pitch, velocity=12, time=419),
        mido.Message('control_change', channel=channel, control=64, value=0, time=187),
        mido.Message('note_on', channel=channel, note=pitch+2, velocity=102, time=242),
        mido.Message('note_off', channel=channel, note=pitch+2, time=211),
    ])
    midi.tracks = [clock, notes]
    midi.save(path)


def test_merge_respects_distinct_and_changing_tempo_maps_and_pedal(tmp_path):
    piano, drums, mixture = (tmp_path / name for name in ('piano.mid', 'drums.mid', 'mix.mid'))
    performance(piano, channel=2, tempo=722891, pitch=60)
    performance(drums, channel=4, tempo=344828, pitch=36)
    assert download.merge_performance([('piano', piano), ('drums', drums)], mixture) == 4
    actual = timed_messages(mixture)
    for source, channel in ((piano, 0), (drums, 9)):
        before = timed_messages(source)
        after = [event for event in actual if event[2] == channel]
        assert [event[0] for event in before] == [event[0] for event in after]
        assert [event[1] for event in before] == pytest.approx([event[1] for event in after], abs=.000251)
        assert sum(event[0]['type'] == 'control_change' for event in after) == 2
    # Program labels cannot make a recorded piano/drum pair turn into violin.
    programs = [m.program for t in mido.MidiFile(mixture).tracks for m in t if m.type == 'program_change']
    assert programs == [0, 0]


def test_empty_ground_truth_track_is_rejected(tmp_path):
    empty = tmp_path / 'empty.mid'
    mido.MidiFile().save(empty)
    with pytest.raises(ValueError, match='No notes in ground-truth'):
        download.merge_performance([('piano', empty)], tmp_path / 'mix.mid')


def test_range_reader_seeks_across_blocks_and_reuses_downloads(monkeypatch):
    content = b'abcdefghijklmnopqrstuvwxyz' * 500
    requests = []
    def fetch(url, bounds):
        requests.append(bounds)
        return content[bounds[0]:bounds[1]+1]
    monkeypatch.setattr(download, 'request', fetch)
    reader = download.RangeReader('mock', len(content), block_size=1024)
    reader.seek(-32, 2)
    assert reader.read() == content[-32:]
    reader.seek(1010)
    assert reader.read(2048) == content[1010:3058]
    count = len(requests)
    reader.seek(1050)
    assert reader.read(100) == content[1050:1150]
    assert len(requests) == count
    assert reader.bytes_downloaded <= len(content)
    with pytest.raises(ValueError, match='Negative seek'):
        reader.seek(-1)


@pytest.mark.parametrize('status,content_range', [(200, ''), (206, 'bytes 0-9/100')])
def test_range_download_refuses_full_audio_or_wrong_slice(monkeypatch, status, content_range):
    response = MagicMock()
    response.status = status
    response.headers = {'Content-Range': content_range}
    response.__enter__.return_value = response
    monkeypatch.setattr(download.urllib.request, 'urlopen', lambda *a, **kw: response)
    with pytest.raises(ValueError, match='refusing audio archive download'):
        download.request('https://example.invalid/audio.zip', (10, 19))
    response.read.assert_not_called()


def test_zip_extraction_cannot_write_outside_dataset(tmp_path):
    content = io.BytesIO()
    with zipfile.ZipFile(content, 'w') as archive:
        archive.writestr('../escape.mid', b'invalid')
    content.seek(0)
    with zipfile.ZipFile(content) as archive, pytest.raises(ValueError, match='escapes destination'):
        download.extract_members(archive, tmp_path / 'raw', lambda _: True)
    assert not (tmp_path / 'escape.mid').exists()


def test_supplement_reads_recorded_mix_and_preserves_official_take_group(tmp_path):
    performances=[]
    for i, split in enumerate(('train','train','validation','test')):
        path=tmp_path/f'take{i}.mid'
        write_midi_notes([Note('Acoustic Piano',72,90,0,200+i*10),
                          Note('Acoustic Piano',60,80,0,300+i*10),
                          Note('drum',42,100,30,20)],path)
        performances.append(dict(path=str(path),source='recorded_jazz',genre='jazz',
            group='song:train' if split=='train' else 'song:'+split,split=split))
    acquisition=tmp_path/'acquisition.json'
    acquisition.write_text(json.dumps(dict(performances=performances)))
    index=tmp_path/'new.index.json'
    prepared=prepare_supplement(acquisition,index)
    assert len(prepared['rows'])==4 and prepared['counts']['drum_records']==4
    assert Counter(r['split'] for r in prepared['rows'])==dict(train=2,validation=1,test=1)
    assert {r['group'] for r in prepared['rows'] if r['split']=='train'}=={'song:train'}
    assert all(r['split_locked'] for r in prepared['rows'])
    assert all(r['instruments']==['Acoustic Piano','drum'] for r in prepared['rows'])
    assert prepare_supplement(acquisition,index)==prepared


def test_official_held_out_groups_are_never_compacted_into_training():
    rows=[dict(group=f'song{i}',source='new',genre='jazz',sha256=str(i),split_locked=True,
               split='train' if i<80 else 'validation' if i<90 else 'test') for i in range(100)]
    result=compact_splits(rows,dict(train=.98,validation=.01,test=.01),42)
    assert [r['split'] for r in result]==[r['split'] for r in rows]


def test_supplement_group_cannot_cross_official_splits(tmp_path):
    acquisition=tmp_path/'acquisition.json'
    path=tmp_path/'take.mid'
    write_midi_notes([Note('Acoustic Piano',60,90,0,200)],path)
    acquisition.write_text(json.dumps(dict(performances=[
        dict(path=str(path),source='new',group='song1',split=split) for split in ('train','test')])))
    with pytest.raises(ValueError,match='cross official splits'):
        prepare_supplement(acquisition,tmp_path/'out.index.json')


def test_new_performances_extend_manifest_without_duplicate_or_split_leakage(tmp_path,monkeypatch):
    config, before=corpus(tmp_path,ratio=1)
    base_index_bytes=Path(config.PREPARED_DATA_FILE).with_suffix('.index.json').read_bytes()
    supplement_data=tmp_path/'supplement.jsonl'
    extra=[]
    with supplement_data.open('wb') as handle:
        for i, split in enumerate(('train','validation','test')):
            record=mix(12)
            # The third record is a transposed copy of existing prepared music.
            record['notes']=[list(n[:4])+[n[4]+(i+1)*10 if i<2 else n[4]] for n in record['notes']]
            if i==2:
                record['notes']=[[n[0],n[1]+12,*n[2:]] for n in record['notes']]
            extra.append(dict(path=f'performance{i}.mid',offset=handle.tell(),jsonl=str(supplement_data),
                source='new_jazz',genre='jazz',split=split,split_locked=True,group=f'new:{i}',
                sha256=f'new{i}',fingerprint=melody_fingerprint(record),
                instruments=['Acoustic Piano','Acoustic Bass','drum'],has_drums=True))
            handle.write((json.dumps(record)+'\n').encode())
    config.EXTRA_PERFORMANCE_INDEX=tmp_path/'supplement.index.json'
    config.EXTRA_PERFORMANCE_INDEX.write_text(json.dumps(dict(rows=extra)))
    source_index_bytes=config.EXTRA_PERFORMANCE_INDEX.read_bytes()
    manifest=build_manifest(config)
    added=[r for r in manifest['prepared_extra_rows'] if r.get('supplemental')]
    assert [r['split'] for r in added]==['train','validation']
    assert manifest['supplemental_performances']==dict(records=2,sources={'new_jazz':2},
        splits={'train':1,'validation':1},drum_records=2)
    assert Path(config.PREPARED_DATA_FILE).with_suffix('.index.json').read_bytes()==base_index_bytes
    assert config.EXTRA_PERFORMANCE_INDEX.read_bytes()==source_index_bytes
    validation=MelodyDataset(config.DATA_FILE,'validation',config)
    assert any(r['source']=='new_jazz' for r in validation.eval_rows)
    train=MelodyDataset(config.DATA_FILE,'train',config)
    assert sum(r['source']=='new_jazz' for r in train.jazz_blues)==1
    monkeypatch.setattr('improvnet.scripts.melody_manifest.fingerprint_matches',
                        lambda *args: pytest.fail('Manifest and duplicate links must be reused'))
    assert build_manifest(config)==manifest


def test_distinct_recorded_drum_takes_are_not_dropped_for_sharing_piano(tmp_path):
    config,_=corpus(tmp_path,ratio=1)
    data=tmp_path/'paired.jsonl'
    rows=[]
    with data.open('wb') as handle:
        for i in range(2):
            record=mix(12)
            record['notes']=[list(n[:4])+[n[4]+50] for n in record['notes']]
            record['drums']=[[n[0],42+i,*n[2:]] for n in record['drums']]
            rows.append(dict(offset=handle.tell(),jsonl=str(data),source='new_gt',genre='jazz',
                split='train',split_locked=True,group='recorded_song',sha256=f'take{i}',
                fingerprint=melody_fingerprint(record),instruments=['Acoustic Piano','drum'],has_drums=True))
            handle.write((json.dumps(record)+'\n').encode())
    config.EXTRA_PERFORMANCE_INDEX=tmp_path/'paired.index.json'
    config.EXTRA_PERFORMANCE_INDEX.write_text(json.dumps(dict(rows=rows)))
    manifest=build_manifest(config)
    assert manifest['supplemental_performances']['records']==2
    assert len([r for r in manifest['prepared_extra_rows'] if r.get('supplemental')])==2

