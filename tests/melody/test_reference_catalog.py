"""Descriptor matching, verified trio catalogue boundaries and source fallback."""
import json
import random
from pathlib import Path
from types import SimpleNamespace
import pytest
from improvnet.model.encoders.reference import choose_reference
from improvnet.utils.melody import Note
from improvnet.utils.melody_reference import trio_instruments
from improvnet.utils.reference_catalog import (CATALOG_FORMAT,load_catalog,rhythm_summary,
    parse_reference_description,rhythm_descriptors)


def catalog_fixture(tmp_path):
    instruments=['Acoustic Piano','Acoustic Bass','drum']
    path=multi_source_catalog(tmp_path,[('example',instruments,80.,1.)])
    return path,json.loads(path.read_text())['examples'][0]


def multi_source_catalog(tmp_path,rows):
    entries=[]
    for source,instruments,bpm,activity in rows:
        midi=tmp_path/(source+'.mid')
        piano=next(i for i in instruments if 'Piano' in i)
        record=dict(path=str(midi),notes=[[piano,61 if source=='broken' else 60,80,i*300,200] for i in range(15)],
                    drums=[['drum',42,80,i*300,20] for i in range(15)])
        record['notes'] += [['Acoustic Bass',48,80,i*300,200] for i in range(15)]
        data=tmp_path/(source+'.jsonl');data.write_text(json.dumps(record)+'\n')
        rhythm=dict(estimated_bpm=bpm,piano_attacks_per_second=activity)
        entries.append(dict(id=source,genre='blues',source=source,split='train',title=source,
            path=str(midi),record=dict(jsonl=str(data),offset=0),instruments=instruments,
            trio_validation=dict(windows=2,stage_candidate_pairs=dict(harmonize=1,drums=1,bass=1)),
            rhythm=rhythm,rhythm_scope='whole_trio_view',descriptors=rhythm_descriptors(rhythm)))
    path=tmp_path/'catalog.json'
    path.write_text(json.dumps(dict(format=CATALOG_FORMAT,examples=entries,
        encoders={k:dict(sha256=k) for k in ('style','beat')})))
    return path


def fake_encoder(embed):
    return SimpleNamespace(style=SimpleNamespace(sha256='style'),beat=SimpleNamespace(sha256='beat'),from_notes=embed)


@pytest.mark.parametrize('description,canonical',[
    ('slow','slow'),(' FAST sparse ','fast sparse'),('medium','moderate'),('busy','dense'),
    ('sparse, slow','slow sparse'),('slow slow','slow'),('any','any')])
def test_supported_descriptions_are_explicit_and_composable(description,canonical):
    assert parse_reference_description(description)==canonical


@pytest.mark.parametrize('description',['','slow fast','sparse dense','any slow','swing','slow mysterious'])
def test_unsupported_or_conflicting_descriptions_fail_clearly(description):
    with pytest.raises(ValueError):parse_reference_description(description)


@pytest.mark.parametrize('bpm,activity,expected',[
    (89.99,1.99,['slow','sparse']),(90.,2.,['moderate','balanced']),
    (139.99,3.99,['moderate','balanced']),(140.,4.,['fast','dense'])])
def test_descriptor_thresholds_are_unambiguous(bpm,activity,expected):
    assert rhythm_descriptors(dict(estimated_bpm=bpm,piano_attacks_per_second=activity))==expected


def test_rhythm_descriptions_do_not_claim_a_groove_without_evidence():
    stats=rhythm_summary([Note('Acoustic Piano',60,90,0,100)])
    assert stats['feel_estimate']=='uncertain' and stats['pulse_confidence']==0
    assert stats['piano_ioi_median_ms'] is None


def test_catalog_selection_loads_entire_piece_and_reports_description(tmp_path):
    path,entry=catalog_fixture(tmp_path);seen=[]
    encoder=fake_encoder(lambda notes,*a,**kw:seen.append(notes) or {})
    result=choose_reference(encoder,'blues',random.Random(99),description='slow sparse',catalog_path=path)
    assert result['id']==entry['id'] and len(seen)==1 and len(seen[0])==45
    assert seen[0][0].onset==0 and seen[0][-1].onset==4200
    assert result['requested_description']=='slow sparse'
    assert result['descriptors']==['slow','sparse']
    with pytest.raises(ValueError,match='No jazz'):
        choose_reference(encoder,'jazz',random.Random(0),catalog_path=path)
    encoder.beat.sha256='changed'
    with pytest.raises(ValueError,match='encoder changed'):
        choose_reference(encoder,'blues',random.Random(0),catalog_path=path)


def test_descriptor_filters_never_silently_relax_to_another_tempo(tmp_path):
    trio=['Acoustic Piano','Acoustic Bass','drum']
    path=multi_source_catalog(tmp_path,[('slow',trio,80,1),('fast',trio,160,5)])
    encoder=fake_encoder(lambda *a,**kw:{})
    for seed in range(8):
        selected=choose_reference(encoder,'blues',random.Random(seed),description='fast dense',catalog_path=path)
        assert selected['id']=='fast'
    with pytest.raises(ValueError,match='Available combinations'):
        choose_reference(encoder,'blues',random.Random(0),description='slow dense',catalog_path=path)


@pytest.mark.parametrize('change,error',[
    (lambda e:e.update(split='test'),'training-fold'),
    (lambda e:e.update(instruments=['Acoustic Piano','Electric Bass','drum']),'double-bass'),
    (lambda e:e.update(instruments=['Acoustic Piano','Acoustic Bass','drum','Tenor Sax']),'extra instruments'),
    (lambda e:e['trio_validation']['stage_candidate_pairs'].update(bass=0),'validated'),
    (lambda e:e.update(descriptors=['fast','dense']),'descriptors')])
def test_catalog_rejects_held_out_nontrio_or_stale_entries(tmp_path,change,error):
    path,_=catalog_fixture(tmp_path)
    payload=json.loads(path.read_text());change(payload['examples'][0]);path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match=error):load_catalog(path)


def test_exact_instrument_matching_filters_before_balancing_sources(tmp_path):
    path=multi_source_catalog(tmp_path,[('electric',['Electric Piano','Acoustic Bass','drum'],80,1),
        ('acoustic',['Acoustic Piano','Acoustic Bass','drum'],80,1)])
    encoder=fake_encoder(lambda *a,**kw:{})
    for seed in range(10):
        selected=choose_reference(encoder,'blues',random.Random(seed),catalog_path=path,mode='exact',
            stage_instruments={'bass':['Acoustic Piano','Acoustic Bass','drum']})
        assert selected['id']=='acoustic'
    with pytest.raises(ValueError,match='requested stage instruments'):
        choose_reference(encoder,'blues',random.Random(0),catalog_path=path,
            stage_instruments={'reference':['Violin']})


def test_catalog_tries_other_sources_when_a_piece_has_no_local_reference_pair(tmp_path):
    trio=['Acoustic Piano','Acoustic Bass','drum']
    path=multi_source_catalog(tmp_path,[(s,trio,80,1) for s in ('broken','working')]);seen=[]
    def embed(notes,*args,**kwargs):
        seen.append(notes[0].pitch)
        if notes[0].pitch==61:raise ValueError('No complete local reference pair')
        return {}
    class OrderedRandom(random.Random):
        def shuffle(self,values):pass
    result=choose_reference(fake_encoder(embed),'blues',OrderedRandom(0),catalog_path=path)
    assert result['id']=='working' and seen==[61,60]


def test_reference_instrument_filter_preserves_trio_piano_substitution_only():
    from improvnet.utils.melody_reference import reference_instruments
    available={'Electric Piano','Acoustic Bass','drum'};target={'Acoustic Piano','Acoustic Bass','drum'}
    assert reference_instruments(target,'trio',available)==available
    assert reference_instruments(target,'exact',available)==target


@pytest.mark.parametrize('piano',['Acoustic Piano','Electric Piano','Bright Piano','Honky-tonk Piano'])
def test_trio_catalogue_accepts_piano_family_with_actual_double_bass(piano):
    assert trio_instruments({piano,'drum','Acoustic Bass'})=={piano,'drum','Acoustic Bass'}
    assert not trio_instruments({piano,'drum','Acoustic Bass','Trumpet'})
    assert not trio_instruments({piano,'drum','Electric Bass'})
    assert not trio_instruments({piano,'Acoustic Bass'})


def test_trio_catalogue_rejects_multiple_piano_instruments():
    assert not trio_instruments({'Acoustic Piano','Electric Piano','drum','Acoustic Bass'})


def test_reference_selection_rechecks_actual_notes_if_metadata_is_stale(tmp_path):
    path,entry=catalog_fixture(tmp_path)
    records=Path(entry['record']['jsonl'])
    record=json.loads(records.read_text())
    record['notes'].append(['Tenor Sax',66,80,0,100])
    records.write_text(json.dumps(record)+'\n')
    def embed(*args,**kwargs):raise AssertionError('An ensemble reference must not reach the encoders')
    with pytest.raises(ValueError,match='extra instruments'):
        choose_reference(fake_encoder(embed),'blues',random.Random(42),catalog_path=path)


def test_refinement_reference_matching_uses_exact_solo_instrument(tmp_path):
    path=multi_source_catalog(tmp_path,[('electric',['Electric Piano','Acoustic Bass','drum'],80,1),
        ('acoustic',['Acoustic Piano','Acoustic Bass','drum'],80,1)])
    selected=choose_reference(fake_encoder(lambda *a,**kw:{}),'blues',random.Random(0),catalog_path=path,
        mode='trio',stage_instruments={'refine':['Acoustic Piano']})
    assert selected['id']=='acoustic'
