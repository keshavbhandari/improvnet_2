"""Shared-window pairing, frozen conditioning and strict neighboring-reference exclusion."""
from dataclasses import replace
import json
import random
from types import SimpleNamespace
import pytest
import torch
from improvnet.utils.melody import MelodyTokenizer, Note, notes_to_tokens, tokens_to_notes
from improvnet.utils.melody_reference import neighboring_reference, reference_ids as style_ids
from improvnet.model.encoders.data import positive_pair, transpose_view, mask_view
from improvnet.model.encoders.style.data import StylePairDataset
from improvnet.model.encoders.runtime import FrozenReferenceEncoder
from improvnet.model.melody_model import MelodyRefinementModel
from improvnet.utils.melody_data import prepare_example, collate_examples
from melody_test_helpers import mix, settings, corpus, reference_checkpoint


def music(count=600):
    return [Note(inst, pitch+i%5 if inst!='drum' else pitch, 90, i*100, 80)
            for i in range(count) for inst, pitch in [('Acoustic Piano',72),('Acoustic Piano',60),('drum',42)]]


def decode(ids):
    tok=MelodyTokenizer()
    return tokens_to_notes([tok.id_to_tok[int(i)] for i in ids],tok)


@pytest.mark.parametrize('aligned',(False,True))
def test_positive_views_keep_mixture_clock_and_token_budgets(aligned):
    tok=MelodyTokenizer()
    for seed in range(15):
        pair=positive_pair(music(), random.Random(seed), tok, aligned=aligned, augment=False)
        a,b=decode(pair['a']),decode(pair['b'])
        assert len(pair['a'])<=512 and len(pair['b'])<=512
        assert pair['a_instruments']==['Acoustic Piano']
        assert pair['b_instruments']==['Acoustic Piano','drum']
        for n in b:
            if n.instrument=='drum':
                assert any(p.instrument=='Acoustic Piano' and p.onset==n.onset for p in b)
        if aligned:
            assert pair['a_bounds']==pair['b_bounds']
        else:
            assert pair['a_bounds'][0]!=pair['b_bounds'][0] or seed!=0


def test_positive_transpositions_are_independent_and_never_transpose_drums():
    pairs=[positive_pair(music(),random.Random(seed),MelodyTokenizer()) for seed in range(40)]
    assert sum(p['shifts'][0]!=p['shifts'][1] for p in pairs)>=30
    assert all(-7<=s<=7 for p in pairs for s in p['shifts'])
    shifted,delta=transpose_view(music(5),random.Random(9))
    assert [n for n in shifted if n.instrument=='drum']==[n for n in music(5) if n.instrument=='drum']
    assert {(a.onset,a.duration) for a in shifted}=={(a.onset,a.duration) for a in music(5)}
    assert all(n.pitch==o.pitch+delta for n,o in zip(shifted,music(5)) if n.instrument!='drum')


@pytest.mark.parametrize('pitch',(0,127))
def test_transposition_bounds_keep_every_pitched_note_valid(pitch):
    notes=[Note('Acoustic Piano',pitch,90,0,100),Note('drum',42,90,0,20)]
    for seed in range(30):
        changed,shift=transpose_view(notes,random.Random(seed))
        assert all(0<=n.pitch<=127 for n in changed)
        assert abs(shift)<=7 and changed[-1]==notes[-1]


def test_solo_piano_files_supply_both_positive_views():
    result=positive_pair([n for n in music() if n.instrument=='Acoustic Piano'],random.Random(2),MelodyTokenizer(),aligned=False)
    assert result['a_instruments']==result['b_instruments']==['Acoustic Piano']
    assert result['a_bounds']!=result['b_bounds']


def test_masking_covers_attribute_and_structural_tokens_but_preserves_endpoints():
    tok=MelodyTokenizer()
    ids=[tok.tok_to_id[t] for t in notes_to_tokens(music(100),tok)]
    structural={i for i,t in enumerate(notes_to_tokens(music(100),tok)) if t=='<T>'}
    masked_positions=set()
    for seed in range(30):
        masked=mask_view(ids,tok,random.Random(seed),rate=.3)
        assert masked[0]==ids[0] and masked[-1]==ids[-1]
        positions={i for i,t in enumerate(masked) if t==tok.tok_to_id['<MASK>']}
        assert len(positions)==round((len(ids)-2)*.3)
        masked_positions|=positions
    assert structural<=masked_positions


@pytest.mark.parametrize('solo',(False,True))
def test_neighbor_excludes_target_window_and_overlapping_held_notes(solo):
    notes=music()+[Note('Acoustic Piano',100,90,100,100000)]
    for seed in range(15):
        ref=neighboring_reference(notes,20000,30000,random.Random(seed),solo=solo)
        assert ref and len(style_ids(ref,MelodyTokenizer()))<=512
        assert ref['end_ms']<=19750 or ref['start_ms']>=30250
        assert not any(n.pitch==100 for n in ref['notes'])
        assert (len({n.instrument for n in ref['notes']})==1)==solo


def test_short_piece_drops_conditioning_instead_of_reusing_target():
    assert neighboring_reference(music(10),0,2000,random.Random(2)) is None
    ex=prepare_example(mix(1),random.Random(3),settings(REFERENCE_PROB=.99),task='harmonize')
    assert not ex['reference_ids'].ne(2).any()




def test_frozen_encoder_identity_no_grad_and_absent_rows(tmp_path):
    path=reference_checkpoint(tmp_path)
    runtime=FrozenReferenceEncoder(path,'cpu')
    tok=MelodyTokenizer()
    ids=torch.tensor([[tok.tok_to_id[t] for t in notes_to_tokens(music(4),tok)]]*2)
    embedding=runtime.encode(ids,torch.ones_like(ids,dtype=torch.bool),torch.tensor([True,False]))
    assert embedding.shape==(2,256) and not embedding.requires_grad
    assert embedding[0].norm().item()==pytest.approx(1.) and not embedding[1].count_nonzero()
    assert not any(p.requires_grad for p in runtime.model.parameters())
    with pytest.raises(ValueError,match='changed'):
        FrozenReferenceEncoder(path,'cpu',expected_sha256='wrong')


def test_pair_dataset_respects_folds_and_deterministic_evaluation(tmp_path,monkeypatch):
    config,_=corpus(tmp_path,ratio=1,raw_count=0)
    import improvnet.model.encoders.data as module
    monkeypatch.setattr(module,'melody_config',config)
    train=StylePairDataset(config.DATA_FILE,'train')
    val=StylePairDataset(config.DATA_FILE,'validation')
    assert {r['group'] for r in train.rows}.isdisjoint({r['group'] for r in val.rows})
    a,b=val[0],val[0]
    assert a['piece_id']==b['piece_id'] and torch.equal(a['a'],b['a']) and torch.equal(a['b'],b['b'])
    assert train[(0,1)]['piece_id']!=val[0]['piece_id']


def test_inference_embeds_piano_drums_once_without_other_tracks(tmp_path):
    runtime=FrozenReferenceEncoder(reference_checkpoint(tmp_path),'cpu')
    reference=runtime.from_notes(music(100)+[Note('Electric Bass',40,90,0,200)],MelodyTokenizer(),random.Random(2),instruments={'Acoustic Piano','drum'})
    assert reference['instruments']==['Acoustic Piano','drum']
    assert reference['embedding'].shape==(256,) and reference['tokens']<=512




def test_target_matching_drops_condition_when_neighbor_has_wrong_instruments():
    only_piano=[n for n in music() if n.instrument=='Acoustic Piano']
    ref=neighboring_reference(only_piano,20000,30000,random.Random(1),instruments={'Acoustic Piano','drum'})
    assert ref is None


def test_pinned_encoder_survives_replacing_the_training_checkpoint(tmp_path):
    from improvnet.model.encoders.runtime import pin_checkpoint
    path=reference_checkpoint(tmp_path)
    pinned=pin_checkpoint(path,tmp_path/'snapshots')
    runtime=FrozenReferenceEncoder(pinned,'cpu')
    from pathlib import Path
    Path(path).write_bytes(b'replaced original checkpoint')
    assert FrozenReferenceEncoder(pinned,'cpu',expected_sha256=runtime.sha256).sha256==runtime.sha256
