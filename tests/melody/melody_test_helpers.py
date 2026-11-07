"""Shared fixtures for current V5 behavior tests."""
import json
import sqlite3
import torch
from dataclasses import astuple
from types import SimpleNamespace
from improvnet.model import melody_config as cfg
from improvnet.scripts.melody_manifest import build_manifest
from improvnet.inference.melody_inference import MelodyInference
from improvnet.utils.ar_utils import ProcessData
from improvnet.utils.melody import Note, MelodyTokenizer, ordered, tokens_to_notes


def settings(**updates):
    return SimpleNamespace(**dict({k: v for k, v in vars(cfg).items() if k.isupper()}, **updates))


def mix(count=200, genre='jazz'):
    notes = ordered([Note(inst, p + i % 4, 90, i * 200, 400)
                     for i in range(count) for inst, p in [('Acoustic Piano', 76), ('Acoustic Piano', 60), ('Acoustic Bass', 48)]])
    return dict(notes=[astuple(n) for n in notes], source='fixture', genre=genre, path='fixture.mid',
                drums=[astuple(Note('drum', 42, 80, i * 100, 20)) for i in range(count * 2)],
                melody_indices=[i for i,n in enumerate(notes) if n.pitch >= 76], has_harmony=True)


def response(example):
    tok = MelodyTokenizer()
    return tokens_to_notes([tok.id_to_tok[i] for i in example['target'].tolist() if i >= 0], tok)


def reference_checkpoint(tmp_path,kind='style'):
    from dataclasses import asdict
    import hashlib
    from improvnet.metrics.continuity.continuity_model import ContinuityConfig, ContinuityEncoder
    tok = MelodyTokenizer()
    enc_config = ContinuityConfig(vocab_size=tok.vocab_size, pad_id=tok.pad_id, embed_dim=32,
        n_heads=4, n_kv_heads=2, n_layers=1, projection_dim=256, max_length=512, dropout=0.)
    path = tmp_path/(kind+'.pt')
    torch.save(dict(format='style_v1' if kind=='style' else 'beat_v1', config=asdict(enc_config),
        model_state_dict=ContinuityEncoder(enc_config).state_dict(), trained_steps=2,
        vocabulary_digest=hashlib.sha256(json.dumps(tok.signature()).encode()).hexdigest()), path)
    return str(path)


def corpus(tmp_path, ratio=8, raw_count=12):
    data = tmp_path / 'prepared.jsonl'
    rows = []
    with data.open('wb') as f:
        for split in ('train','validation','test'):
            for genre in ('jazz','classical'):
                rec = mix(12, genre)
                rows.append(dict(offset=f.tell(), genre=genre, source='fixture', split=split,
                                 source_group=f'known:{genre}:{split}', instruments=['Acoustic Piano','Acoustic Bass','drum'], has_multi_instrument=True, sha256=f'{len(rows):064x}'))
                f.write((json.dumps(rec)+'\n').encode())
    data.with_suffix('.index.json').write_text(json.dumps(dict(rows=rows)))
    index = tmp_path / 'midi.sqlite'
    with sqlite3.connect(index) as db:
        db.execute('CREATE TABLE files(path TEXT,collection TEXT,genre TEXT,genres TEXT,group_id TEXT,split TEXT)')
        for i in range(raw_count):
            db.execute('INSERT INTO files VALUES(?,?,?,?,?,?)',
                       (str(tmp_path/f'raw{i}.mid'),'gigamidi','pop','["pop"]',f'raw:{i}','train'))
        for g, split, group in [('classical','test','raw:test'),('jazz','train','raw:jazz'),
                                 ('classical','train','known:classical:test')]:
            db.execute('INSERT INTO files VALUES(?,?,?,?,?,?)',('excluded.mid','gigamidi',g,json.dumps([g]),group,split))
    config = settings(DATA_FILE=tmp_path/'manifest.json', PREPARED_DATA_FILE=data,
                      CLASSICAL_INDEX=index, PARSED_CACHE=tmp_path/'cache', CLASSICAL_RATIO=ratio, AR_JSONL_FILES=[],
                      RECOVERED_DATA_FILE=tmp_path/'recovered.jsonl', EXTRA_PERFORMANCE_INDEX=None,
                      DATA_SPLIT_RATIOS={'train':1/3,'validation':1/3,'test':1/3}, STYLE_ENCODER=reference_checkpoint(tmp_path), STYLE_ENCODER_SHA256="fixture", BEAT_ENCODER=reference_checkpoint(tmp_path,kind="beat"), BEAT_ENCODER_SHA256="beat-fixture")
    manifest = build_manifest(config)
    return config, manifest


def scripted_inference(script):
    """A model requesting specific attributes, with a last-tick fallback."""
    inference = MelodyInference.__new__(MelodyInference)
    inference.tokenizer = tok = MelodyTokenizer()
    inference.device = torch.device('cpu')
    inference.meta = dict(version=9,seq_len=512, genres=['jazz'])
    inference.processor = ProcessData()
    calls = []

    def model(ids, **kwargs):
        index = len(calls)
        calls.append(ids)
        logits = torch.full((1, ids.shape[1], tok.vocab_size), -100.)
        # The old in-window mask would replace an outside-window request with
        # this fallback, allowing another attack on the final legal tick.
        logits[..., tok.tok_to_id[('onset', 190)]] = 10.
        logits[..., tok.tok_to_id[script[index] if index < len(script) else '<E>']] = 100.
        return logits, ()

    inference.model = model
    return inference, calls


def attrs(pitch, onset, duration=500, instrument='Acoustic Piano', velocity=90):
    return [('instrument', instrument), ('note', pitch, velocity),
            ('onset', onset), ('dur', duration)]


def prompt_source(ex):
 tok=MelodyTokenizer();stream=[tok.id_to_tok[i] for i in ex['input'].tolist()];a=stream.index('<SEP>');b=stream.index('<SEP>',a+1)
 return stream[a+1:b]

