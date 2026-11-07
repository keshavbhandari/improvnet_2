"""Reference selection must be train-only, reproducible and linked to encoder identities."""
import json
import random
from types import SimpleNamespace
import pytest
from improvnet.model.encoders.reference import choose_reference
from improvnet.utils.melody import Note
from improvnet.utils.reference_catalog import CATALOG_FORMAT, cosine_clusters, load_catalog, rhythm_summary


def test_cosine_groups_ignore_vector_scale_and_are_reproducible():
    x = [[1,0],[3,0],[-1,0],[-4,0]]
    labels = cosine_clusters(x,2)
    assert labels == cosine_clusters(x,2)
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[0] != labels[2]
    with pytest.raises(ValueError):
        cosine_clusters([[0,0]],2)


def test_rhythm_descriptions_do_not_claim_a_groove_without_evidence():
    stats = rhythm_summary([Note('Acoustic Piano',60,90,0,100)])
    assert stats['feel_estimate'] == 'uncertain'
    assert stats['pulse_confidence'] == 0
    assert stats['piano_ioi_median_ms'] is None


def catalog_fixture(tmp_path):
    midi = tmp_path/'blues.mid'
    row = dict(path=str(midi),notes=[['Acoustic Piano',60,80,i*300,200] for i in range(15)],drums=[])
    source=tmp_path/'records.jsonl'
    source.write_text(json.dumps(row)+'\n')
    entry=dict(id='blues-example',genre='blues',source='fixture',split='train',title='Example',
        path=str(midi),record=dict(jsonl=str(source),offset=0),instruments=['Acoustic Piano'],
        excerpt=dict(seed=123,instruments=['Acoustic Piano']),rhythm={},style_bucket='style_0',beat_bucket='beat_0')
    path=tmp_path/'catalog.json'
    path.write_text(json.dumps(dict(format=CATALOG_FORMAT,examples=[entry],
        encoders={k:dict(sha256=k) for k in ('style','beat')})))
    return path,entry


def test_catalog_selection_reuses_saved_excerpt_seed_and_rejects_wrong_genre(tmp_path):
    path,entry=catalog_fixture(tmp_path)
    seen=[]
    encoder=SimpleNamespace(style=SimpleNamespace(sha256='style'),beat=SimpleNamespace(sha256='beat'))
    def embed(notes,tok,rng,regions):
        seen.append(rng.random())
        return dict(style_region={},beat_region={})
    encoder.from_notes=embed
    result=choose_reference(encoder,'blues',random.Random(99),reference_id=entry['id'],catalog_path=path)
    assert result['id']==entry['id'] and seen==[random.Random(123).random()]
    with pytest.raises(ValueError,match='No jazz'):
        choose_reference(encoder,'jazz',random.Random(0),catalog_path=path)
    with pytest.raises(ValueError,match='matches'):
        choose_reference(encoder,'blues',random.Random(0),bucket='beat_99',catalog_path=path)
    encoder.beat.sha256='changed'
    with pytest.raises(ValueError,match='encoder changed'):
        choose_reference(encoder,'blues',random.Random(0),catalog_path=path)


def test_catalog_rejects_held_out_references(tmp_path):
    path,_=catalog_fixture(tmp_path)
    payload=json.loads(path.read_text());payload['examples'][0]['split']='test'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='training-fold'):
        load_catalog(path)
