"""Complete responses, safe references and history-first context budgets."""
from dataclasses import replace
import random
import torch
import pytest

from melody_test_helpers import settings,mix,response,prompt_source,scripted_inference,attrs
from improvnet.utils.melody import (Note,MelodyTokenizer,notes_to_tokens,
    tokens_to_notes,make_prompt,ordered)
from improvnet.utils.melody_chunks import history_tokens
from improvnet.utils.melody_data import prepare_example,NoReferenceWindow
from improvnet.utils.melody_corruptions import corrupt_history


def dense_piece(voices):
    return dict(source='fixture',genre='jazz',path='dense.mid',drums=[],
        notes=[('Acoustic Piano',48+p,90,i*100,80) for i in range(450) for p in range(voices)])


def test_complete_dense_target_uses_over_2048_tokens_and_reclaims_history():
    rec=dense_piece(12)
    ex=prepare_example(rec,random.Random(3),settings(POLY_HISTORY_PROB=1.,HISTORY_CORRUPTION_PROB=0.),'harmonize')
    clean=response(ex)
    assert ex['response_length']>2048 and ex['history_tokens_dropped']>0
    assert len(ex['input'])+66<=8192 and ex['source_length']<=512
    expected=ordered([replace(Note(*n),onset=n[3]-ex['start_ms']) for n in rec['notes'] if ex['start_ms']<=n[3]<ex['end_ms']])
    assert clean==expected
    source_notes=sum(isinstance(t,tuple) and t[0]=='note' for t in prompt_source(ex))
    assert len(clean)==12*source_notes and source_notes>=120


def test_unavoidably_oversized_mix_shortens_whole_window_never_partial_targets():
    rec=dense_piece(20)
    ex=prepare_example(rec,random.Random(3),settings(POLY_HISTORY_PROB=1.),'harmonize')
    clean=response(ex)
    assert len(ex['input'])+66<=8192 and ex['response_length']>2048
    assert len(clean)%20==0
    assert clean==ordered([replace(Note(*n),onset=n[3]-ex['start_ms']) for n in rec['notes'] if ex['start_ms']<=n[3]<ex['end_ms']])


def test_stem_source_can_be_clean_or_explicitly_reenable_corruption():
    for probability in (0.,1.):
        ex=prepare_example(mix(),random.Random(17),settings(STEM_SOURCE_CORRUPTION_PROB=probability),'stem',target_instrument='drum')
        retained=[n for n in response(ex) if n.instrument!='drum']
        assert (tokens_to_notes(prompt_source(ex))==retained)==(probability==0.)
        assert ex['stem_source_corrupted']==bool(probability)




def test_required_references_never_use_target_when_no_neighbor_exists():
    record=dict(source='curated',genre='jazz',path='short.mid',notes=[('Acoustic Piano',60,90,0,100)],drums=[])
    with pytest.raises(NoReferenceWindow):
        prepare_example(record,random.Random(0),settings(REFERENCE_PROB=1.),'harmonize')
    ex=prepare_example(mix(300),random.Random(1),settings(REFERENCE_PROB=1.),'harmonize')
    assert ex['reference_ids'].ne(2).any()


def test_decoder_rebuilds_cache_without_discarding_response():
    from improvnet.inference.melody_note_decode import generate_notes
    infer,_=scripted_inference([])
    infer.meta.update(version=19,seq_len=256)
    infer.progress_disabled=True
    tok=infer.tokenizer
    script=[]
    for i in range(20):
        script+=attrs(60+i%4,i*50,40)
    script+=['<E>']
    context=[];prefills=[]
    def model(ids,**kwargs):
        nonlocal context
        tokens=[tok.id_to_tok[i] for i in ids[0].tolist()]
        if kwargs.get('past_key_values') is None:
            prefills.append(tokens)
            last=len(tokens)-1-tokens[::-1].index('<SEP>')
            context=tokens[last+2:]
        else:
            context+=tokens
        logits=torch.full((1,ids.shape[1],tok.vocab_size),-100.)
        logits[:,:,-1]=-100.
        logits[...,tok.tok_to_id[script[len(context)]]]=100.
        return logits,()
    infer.model=model
    history=history_tokens([Note('Acoustic Piano',50,90,i*10,10) for i in range(24)],100,tok)
    source=notes_to_tokens([Note('Acoustic Piano',60,90,0,80)],tok)
    prompt=make_prompt('refine',source,history)
    out=generate_notes(infer,prompt,'jazz',.4,['Acoustic Piano'],2000,temperature=0.,monophonic=True)
    assert len(prefills)==2 and len(prefills[1])>len(source)+4
    assert len(out)==20
    assert infer.last_decode_stats['history_tokens_dropped']==len(history)
    assert all(len(p)+66<=256 for p in prefills)
