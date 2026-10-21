"""Current V5 decode contracts."""
import random
import sys
import pytest
import torch
from collections import Counter
from dataclasses import replace, astuple
from improvnet.inference.melody_inference import MelodyInference, deduplicate_attacks
from improvnet.utils.ar_utils import ProcessData
from improvnet.utils.melody import *
from improvnet.model import melody_config as cfg
from types import SimpleNamespace
from improvnet.utils.melody_chunks import *
from improvnet.utils.melody_data import prepare_example
from melody_test_helpers import response, attrs, scripted_inference


def decode(script, window_ms=200, **kwargs):
    inference, calls = scripted_inference(script)
    result = inference.generate(make_prompt('harmonize', ['<S>', '<E>']),
                                'jazz', 1., ['Acoustic Piano', 'Violin'], window_ms,
                                temperature=0., stop_at_window_end=True,
                                max_new_tokens=128, **kwargs)
    return inference, result, calls


@pytest.mark.parametrize('future_onset', (200, 210, 2000, 4990))
def test_future_onset_ends_chunk_without_folding_note_onto_last_tick(future_onset):
    infer, result, _ = decode(attrs(72, 190) + attrs(60, future_onset) + ['<E>'])
    assert result == [Note('Acoustic Piano', 72, 90, 190, 500)]
    assert infer.last_decode_stats['end_reason'] == 'window_boundary'
    # The accepted note retains its release across the end of this window.
    assert result[0].onset + result[0].duration > 200


def test_genuine_chord_can_finish_on_final_legal_tick():
    script = sum((attrs(p, 190) for p in (72, 67, 60)), []) + attrs(48, 200)
    infer, result, _ = decode(script)
    assert result == [Note('Acoustic Piano', p, 90, 190, 500) for p in (72, 67, 60)]
    assert infer.last_decode_stats['end_reason'] == 'window_boundary'


@pytest.mark.parametrize('window_ms', (200, 5000, 10000, 15000))
def test_clock_advance_past_window_ends_at_partial_and_exact_time_bins(window_ms):
    ticks, onset = divmod(window_ms - 10, 5000)
    infer, result, _ = decode(['<T>'] * ticks + attrs(72, onset) + ['<T>'], window_ms)
    assert result == [Note('Acoustic Piano', 72, 90, window_ms - 10, 500)]
    assert infer.last_decode_stats['end_reason'] == 'window_boundary'


def test_future_first_attack_returns_no_partial_note():
    infer, result, _ = decode(attrs(60, 200))
    assert result == []
    assert infer.last_decode_stats['end_reason'] == 'window_boundary'


def test_duplicate_attributes_do_not_create_another_strike_but_retriggers_survive():
    script = (attrs(60, 0, 100) + attrs(60, 0, 500, velocity=60)
              + attrs(60, 10, 100) + attrs(60, 10, 500, instrument='Violin') + ['<E>'])
    infer, result, _ = decode(script)
    assert result == ordered([Note('Acoustic Piano', 60, 90, 0, 100),
                              Note('Acoustic Piano', 60, 90, 10, 100),
                              Note('Violin', 60, 90, 10, 500)])
    assert infer.last_decode_stats == dict(duplicate_attacks_removed=1, end_reason='end_token')


def test_deduplication_uses_serialized_onset_and_roundtrip_preserves_repeated_notes(tmp_path):
    first = Note('Acoustic Piano', 60, 90, 100, 300)
    later = replace(first, onset=110)
    unison = replace(first, instrument='Violin')
    generated = [first, replace(first, onset=101), replace(first, onset=104), later, unison]
    unique = deduplicate_attacks(generated)
    assert unique == ordered([first, later, unison])
    path = tmp_path / 'unique.mid'
    write_midi_notes(unique, path)
    actual = read_midi_notes(path)
    assert Counter((n.instrument, n.pitch, n.onset) for n in actual) == Counter(
        (n.instrument, n.pitch, n.onset) for n in unique)


@pytest.mark.parametrize('expand',(False,True))
def test_refinement_source_count_changes_do_not_skip_input(expand,monkeypatch):
    infer,_=scripted_inference([])
    infer.meta.update(version=9,tasks=list(TASKS),seq_len=512,chunk_tokens=64,history_tokens=100,
                      mix_response_tokens=256,harmonize_expansion=4.,skyline_tolerance_ms=50)
    infer.options=cfg
    notes=[Note('Acoustic Piano',72,90,i*1000,2000) for i in range(100)]
    def corrupt(core,*a):
        return notes_to_tokens(core+[replace(n,pitch=60) for n in core] if expand else [])
    monkeypatch.setattr('improvnet.inference.melody_inference.refinement_source',corrupt)
    def generate(prompt,*a,**kw):
        infer.last_decode_stats={}
        assert len(prompt)+kw['max_new_tokens']+2<=512
        return [Note('Acoustic Piano',65,90,a[3]-10,100)]
    infer.generate=generate
    _,logs=infer.transform(notes,'refine','jazz',1.,random.Random(1),monophonic=True)
    assert len(logs)>1
    assert sum(x['consumed_input_notes'] for x in logs)==len(notes)
    assert all(x['source_tokens']<=64 and x['total_token_budget']<=512 for x in logs)
    if not expand:assert all(x['source_tokens']==2 for x in logs)


def test_output_cannot_overwrite_input_or_intermediate_source(tmp_path,monkeypatch):
    from improvnet.inference import melody_inference as module
    source=tmp_path/'out'/'harmonize.mid'
    source.parent.mkdir()
    write_midi_notes([Note('Acoustic Piano',60,90,0,100)],source)
    class Fake:
        def __init__(self,*a):
            self.tokenizer=MelodyTokenizer();self.meta=dict(version=9,skyline_tolerance_ms=50)
    monkeypatch.setattr(module,'MelodyInference',Fake)
    monkeypatch.setattr(sys,'argv',['melody_inference','--model','unused','--input',str(source),'--output',str(tmp_path/'out.mid'),'--no-rhythm-conditioning'])
    with pytest.raises(SystemExit):module.main()


@pytest.mark.parametrize('ending',(['<E>'],['<T>'],attrs(42,200,20,'drum'),attrs(42,210,20,'drum'),attrs(42,2000,20,'drum'),attrs(42,4990,20,'drum')))
def test_stem_chunk_can_rest_without_losing_anchors_or_inventing_a_last_tick_hit(ending):
    anchor=Note('Acoustic Piano',72,90,0,1000)
    infer,_=scripted_inference(attrs(72,0,1000)+ending)
    result=infer.generate(make_prompt('stem',['<S>','<E>'],target_instrument='drum'),
        'jazz',1.,['Acoustic Piano','drum'],200,temperature=0.,fixed_notes=[anchor],
        stem_add=True,target_instrument='drum',stop_at_window_end=True,max_new_tokens=64)
    assert result==[anchor]
    assert infer.last_decode_stats['stem_rest'] and infer.last_decode_stats['stem_note_count']==0


def test_stem_rest_is_allowed_when_the_budget_only_fits_the_retained_notes():
    anchor=Note('Acoustic Piano',72,90,0,1000)
    infer,_=scripted_inference(attrs(72,0,1000)+['<E>'])
    result=infer.generate(make_prompt('stem',['<S>','<E>'],target_instrument='drum'),
        'jazz',1.,['Acoustic Piano','drum'],200,temperature=0.,fixed_notes=[anchor],
        stem_add=True,target_instrument='drum',stop_at_window_end=True,max_new_tokens=5)
    assert result==[anchor] and infer.last_decode_stats['stem_rest']


@pytest.mark.parametrize('all_silent',(False,True))
def test_stem_stage_continues_over_silent_chunks_and_a_wholly_missing_stem(all_silent, capsys):
    infer,_=scripted_inference([])
    infer.meta.update(version=9,tasks=list(TASKS),seq_len=512,chunk_tokens=18,
        history_tokens=64,mix_response_tokens=128,harmonize_expansion=4.,skyline_tolerance_ms=50)
    infer.options=cfg
    original=[Note('Acoustic Piano',60+i%5,90,i*100,400) for i in range(12)]
    calls=[]
    def generate(prompt,*args,**kwargs):
        calls.append(kwargs)
        core=kwargs['fixed_notes'];add=not all_silent and len(calls)!=2
        infer.last_decode_stats=dict(stem_rest=not add,stem_note_count=int(add))
        return ordered(core+([Note('drum',42,80,core[0].onset,20)] if add else []))
    infer.generate=generate
    result,logs=infer.transform(original,'stem','jazz',1.,random.Random(1),instruments=['Acoustic Piano','drum'],target_instrument='drum')
    assert len(calls)==3
    assert [n for n in result if n.instrument!='drum']==original
    assert logs[1]['decoding']['stem_rest']
    if all_silent:
        assert result==original
        assert [x['added_stem_notes'] for x in logs]==[0,0,0]
        assert 'No drum notes generated; continuing' in capsys.readouterr().err
    else:
        assert [x['added_stem_notes'] for x in logs]==[1,0,1]
        assert sum(n.instrument=='drum' for n in result)==2


def test_v5_stem_repairs_timing_but_preserves_chords_melody_pitch_and_pitch_change_cap():
    anchors = ordered([Note('Acoustic Piano', pitch, 90, i*100, 200)
                       for i in range(21) for pitch in (72, 60)])
    script = []
    for note in anchors:
        # The model tries to move both melody and accompaniment pitches.
        # Only one of the 21 eligible accompaniment notes may change.
        pitch = 73 if note.pitch == 72 else 62
        requested_onset = note.onset + (20 if note.pitch == 72 else 40)
        script.extend(attrs(pitch, requested_onset, 240))
    infer, _ = scripted_inference(script + ['<E>'])
    infer.meta.update(version=9, stem_timing_jitter_ms=50, stem_pitch_shift_prob=.05,
                      stem_pitch_shift_semitones=2, skyline_tolerance_ms=50)
    result = infer.generate(make_prompt('stem', ['<S>', '<E>'], target_instrument='drum'),
        'jazz', 1., ['Acoustic Piano', 'drum'], 2200, temperature=0., fixed_notes=anchors,
        stem_add=True, target_instrument='drum', stop_at_window_end=True, max_new_tokens=256)
    assert len(result) == len(anchors)
    assert [n.pitch for n in result if n.pitch >= 70] == [72]*21
    assert sum(n.pitch == 62 for n in result) == 1
    for group in range(21):
        chord = [n for n in result if n.onset == group*100+20]
        assert len(chord) == 2 and all(n.duration == 240 for n in chord)
    assert infer.last_decode_stats['retained_pitch_correction_budget'] == 1
    assert infer.last_decode_stats['retained_pitch_corrections'] == 1
    assert infer.last_decode_stats['stem_rest']


def test_v5_stem_correction_can_cross_five_second_bin_without_losing_anchors():
    anchors = [Note('Acoustic Piano', 72, 90, 0, 200),
               Note('Acoustic Piano', 74, 90, 4990, 200)]
    infer, _ = scripted_inference(attrs(72, 0, 200)+['<T>']+attrs(74, 0, 200)+['<E>'])
    infer.meta.update(version=9)
    result = infer.generate(make_prompt('stem', ['<S>', '<E>'], target_instrument='drum'),
        'jazz', 1., ['Acoustic Piano', 'drum'], 5100, temperature=0., fixed_notes=anchors,
        stem_add=True, target_instrument='drum', stop_at_window_end=True, max_new_tokens=10)
    assert result == [anchors[0], replace(anchors[1], onset=5000)]
    assert infer.last_decode_stats['retained_onset_corrections'] == 1


def test_v5_anchor_policy_protects_percussion_pitch_and_owned_window_and_group_order():
    from improvnet.inference.melody_stem_anchors import StemAnchors
    anchors = [Note('Acoustic Piano', 72, 90, 0, 200),
               Note('drum', 42, 90, 20, 20),
               Note('Acoustic Piano', 60, 90, 30, 200)]
    policy = StemAnchors(anchors, MelodyTokenizer(), dict(stem_pitch_shift_prob=1.), 40,
                         protected=[True, False, False])
    assert policy.pitch_tokens(1) == [('note', 42, 90)]
    assert policy.onset_bounds(0, -10) == (0, 10)
    policy.commit(0, replace(anchors[0], onset=10))
    assert policy.onset_bounds(1, 10) == (20, 20)
    policy.commit(1, replace(anchors[1], onset=20))
    assert policy.onset_bounds(2, 20) == (30, 30)


def test_v5_stem_transform_protects_rolled_chord_skyline_across_chunk_edges(monkeypatch):
    infer, _ = scripted_inference([])
    infer.meta.update(version=9,tasks=['refine','harmonize','stem'],seq_len=512,chunk_tokens=10,
        history_tokens=64,mix_response_tokens=128,harmonize_expansion=4.,skyline_tolerance_ms=50)
    infer.options=cfg
    # The lower note is in the next exact-onset chunk but belongs to the same
    # 50ms performed chord. Computing skyline from each chunk would protect it.
    original=[Note('Acoustic Piano',72,90,0,200),Note('Acoustic Piano',60,90,30,200),
              Note('Acoustic Piano',74,90,100,200),Note('Acoustic Piano',62,90,130,200)]
    protected=[]
    def generate(prompt,*args,**kwargs):
        protected.extend(kwargs['protected_pitches'])
        infer.last_decode_stats={}
        return kwargs['fixed_notes']
    monkeypatch.setattr(infer,'generate',generate)
    infer.transform(original,'stem','jazz',1.,random.Random(42),instruments=['Acoustic Piano','drum'],target_instrument='drum')
    assert protected == [True,False,True,False]


def test_v5_stream_waits_for_complete_rolled_chord_without_changing_chunk_boundaries(monkeypatch):
    infer, _ = scripted_inference([])
    infer.meta.update(version=9,tasks=['refine','harmonize','stem'],seq_len=512,chunk_tokens=10,
        history_tokens=64,mix_response_tokens=128,harmonize_expansion=4.,skyline_tolerance_ms=50)
    infer.options=cfg
    # The first buffered prefix has a complete next onset group (30ms), yet
    # its apparent skyline at 0ms is replaced by a higher 40ms attack later.
    original=[Note('Acoustic Piano',72,90,0,200),Note('Acoustic Piano',60,90,20,200),
              Note('Acoustic Piano',55,90,30,200),Note('Acoustic Piano',76,90,40,200),
              Note('Acoustic Piano',74,90,200,200)]
    class Stream:
        done=False
        reads=0
        batches=iter([original[3:],None])
        def read(self):
            self.reads += 1
            batch=next(self.batches)
            self.done=batch is None
            return batch
    stream=Stream()
    decoded=[]
    def generate(prompt,*args,**kwargs):
        decoded.append((stream.reads,list(kwargs['protected_pitches']),list(kwargs['fixed_notes'])))
        infer.last_decode_stats={}
        return kwargs['fixed_notes']
    monkeypatch.setattr(infer,'generate',generate)
    common=dict(instruments=['Acoustic Piano','drum'],target_instrument='drum')
    streamed, stream_logs=infer.transform(original[:3],'stem','jazz',1.,random.Random(42),
                                          source_stream=stream,**common)
    assert decoded[0][0] == 1
    assert decoded[0][1] == [False,False]
    assert [n.onset for n in decoded[0][2]] == [0,20]
    decoded.clear()
    offline, offline_logs=infer.transform(original,'stem','jazz',1.,random.Random(42),**common)
    assert streamed == offline == original
    assert stream_logs == offline_logs


@pytest.mark.parametrize('window_ms', (5000,5100))
def test_v5_clock_correction_respects_tight_response_budget_and_owned_window(window_ms):
    anchor=Note('Acoustic Piano',72,90,4990,200)
    # No spare token for a clock marker: requesting one must not spend the
    # fixed-note reserve. Correction can stay on 4990ms without losing it.
    infer,_=scripted_inference(['<T>']+attrs(72,0,200)+['<E>'])
    infer.meta.update(version=9)
    result=infer.generate(make_prompt('stem',['<S>','<E>'],target_instrument='drum'),
        'jazz',1.,['Acoustic Piano','drum'],window_ms,temperature=0.,fixed_notes=[anchor],
        stem_add=True,target_instrument='drum',stop_at_window_end=True,max_new_tokens=5)
    assert len(result) == 1
    assert 4940 <= result[0].onset <= 4990
    assert infer.last_decode_stats['end_reason'] == 'end_token'


def performance():
    return [Note('Acoustic Piano', p, 90, start, dur) for p, start, dur in
            [(48, 0, 400), (60, 10, 400), (67, 30, 400), (64, 700, 800),
             (52, 700, 800), (65, 1600, 700), (53, 1600, 700),
             (67, 3000, 900), (55, 3000, 900), (69, 4900, 500),
             (57, 4900, 500), (71, 5500, 400), (59, 5500, 400)]]


def record():
    notes = sorted(performance(), key=lambda n: (n.onset, -n.pitch, n.instrument, n.duration))
    melody, _ = skyline(notes)
    selected = {id(n) for n in melody}
    return dict(notes=[astuple(n) for n in notes],
                melody_indices=[i for i, n in enumerate(notes) if id(n) in selected],
                genre='jazz', source='fixture', windows=[0, 1], harmony_windows=[0, 1],
                drums=[astuple(Note('drum', 51, 80, t, 100)) for t in range(0, 6000, 500)])


def long_record(attacks=2400):
    notes = [Note('Acoustic Piano', pitch, 90, i * 120, 600)
             for i in range(attacks) for pitch in (72 + i % 5, 60, 55, 48)]
    return dict(notes=[astuple(n) for n in notes], melody_indices=list(range(0, len(notes), 4)),
                genre='jazz', source='fixture', has_harmony=True)


def test_chunking_keeps_whole_chords_and_bounded_history():
    tok = MelodyTokenizer()
    notes = [Note(*n) for n in long_record(300)['notes']]
    starts = chunk_starts(notes, 600)
    chunks = []
    for start in starts:
        stop = chunk_end(notes, start, 600)
        assert start % 4 == stop % 4 == 0
        local = [replace(n, onset=n.onset - notes[start].onset) for n in notes[start:stop]]
        assert token_count(notes, start, stop) == len(notes_to_tokens(local, tok)) <= 600
        chunks.extend(notes[start:stop])
    assert chunks == notes
    # A long preceding rest must not consume the history budget as leading T's.
    late = [replace(n, onset=n.onset + 1_000_000) for n in notes]
    history = history_tokens(late, 2048, tok)
    assert len(history) <= 2048 and history[1] != '<T>'
    decoded = tokens_to_notes(history, tok)
    assert len(decoded) % 4 == 0
    assert decoded[0].onset == 0 and decoded[-1].pitch == notes[-1].pitch
    # Never spin forever if the last remaining note/group cannot fit.
    with pytest.raises(ValueError, match='complete onset group'):
        shorten_chunk(notes[:1], 0, 1, .5)
    with pytest.raises(ValueError, match='complete onset group'):
        shorten_chunk(notes[:4], 0, 4, .5)


def test_joint_window_owns_rests_before_the_next_source_chunk():
    notes=[Note('Acoustic Piano',72,90,0,100),Note('Electric Bass',48,90,1000,100),
           Note('Acoustic Piano',72,90,5000,100)]
    assert chunk_bounds(notes,0,2)==(0,1100)
    assert stem_bounds(notes,0,2)==(0,5000)
    assert stem_bounds(notes,2,3)==(5000,5100)


@pytest.mark.parametrize('prefer_clock', (False, True))
@pytest.mark.parametrize('budget', (5, 6, 9, 11, 13, 14, 64))
def test_note_adding_generation_finishes_inside_token_budget(budget, prefer_clock):
    from improvnet.inference.melody_inference import MelodyInference
    from improvnet.utils.ar_utils import ProcessData
    from improvnet.utils.melody import make_prompt
    inference = MelodyInference.__new__(MelodyInference)
    inference.tokenizer = tok = MelodyTokenizer()
    inference.meta = dict(seq_len=256, genres=list(cfg.GENRES))
    inference.device = torch.device('cpu')
    inference.processor = ProcessData()
    trace = []
    def eager_notes(ids, **kwargs):
        trace.append(ids[0].tolist())
        logits = torch.zeros(1, ids.shape[1], tok.vocab_size)
        logits[..., tok.tok_to_id[('instrument', 'Acoustic Piano')]] = 100
        if prefer_clock:
            logits[..., tok.tok_to_id['<T>']] = 200
        return logits, ()
    inference.model = eager_notes
    # Complete fragmentation: no notes remain in the source, but the response
    # may add notes. The test model always requests more notes/time markers.
    prompt = make_prompt('refine', ['<S>', '<E>'])
    result = inference.generate(prompt, 'jazz', 1, ['Acoustic Piano'], 1_000_000,
                                temperature=0, monophonic=True, max_new_tokens=budget)
    response = ['<S>'] + [tok.id_to_tok[ids[0]] for ids in trace[1:]] + ['<E>']
    assert len(response) - 1 <= budget
    assert len(prompt) + len(response) - 1 + 2 <= 256
    assert tokens_to_notes(response, tok) == result and result
    assert response[-2][0] == 'dur'  # Complete notes, no truncation of attributes.


def tiny_kwargs():
    return dict(vocab_size=MelodyTokenizer().vocab_size, embed_dim=32,
                n_heads=4, n_kv_heads=2, n_layers=1, dropout=.1,
                num_genres=4, num_instruments=41)


def test_skyline_partitions_originals_and_keeps_attacks():
    notes = performance()
    melody, harmony = skyline(notes)
    assert Counter(melody) + Counter(harmony) == Counter(notes)
    assert len(melody) + len(harmony) == len(notes)
    assert [n.pitch for n in melody[:2]] == [67, 64]
    repeated = [Note('Acoustic Piano', 60, 90, t, 10) for t in (0, 20, 40)]
    assert skyline(repeated)[0] == repeated
    fast_run = [Note('Acoustic Piano', 60 + i, 90, i * 20, 10) for i in range(3)]
    assert skyline(fast_run)[0] == fast_run


def test_window_ownership_and_midi_roundtrip(tmp_path):
    held = Note('Acoustic Piano', 60, 90, 4900, 500)
    assert window([held], 0, 5000) == [held]
    assert window([held], 5000, 10000) == []
    path = tmp_path / 'held.mid'
    write_midi_notes([held], path)
    assert read_midi_notes(path) == [held]
    repeated = [Note('Acoustic Piano', 60, 90, 0, 3000),
                Note('Acoustic Piano', 60, 90, 1000, 2000)]
    write_midi_notes(repeated, path)
    decoded = read_midi_notes(path)
    assert len(decoded) == 2 and decoded[0].duration == 1000
    assert decoded[1] == repeated[1]


def test_missing_program_change_uses_canonical_piano(tmp_path):
    import mido
    path = tmp_path / 'missing_program.mid'
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.extend([
        mido.Message('program_change', channel=1, program=40, time=0),
        mido.Message('note_on', channel=3, note=60, velocity=90, time=0),
        mido.Message('note_on', channel=1, note=64, velocity=90, time=0),
        mido.Message('note_off', channel=3, note=60, velocity=0, time=480),
        mido.Message('note_off', channel=1, note=64, velocity=0, time=0),
    ])
    midi.save(path)
    tok = MelodyTokenizer()
    notes = read_midi_notes(path, tok)
    assert {n.instrument for n in notes} == {'Acoustic Piano', 'Violin'}
    assert all(t in tok.tok_to_id for t in notes_to_tokens(notes, tok))


def test_quantized_solo_ties_become_one_melody_attack():
    notes = [Note('Acoustic Piano', 60, 90, 0, 100),
             Note('Acoustic Piano', 64, 90, 0, 100),
             Note('Acoustic Piano', 62, 90, 10, 100)]
    assert [n.pitch for n in melody_voice(notes)] == [64, 62]
    rec = dict(notes=[astuple(n) for n in notes], melody_indices=[0, 1, 2],
               genre='jazz', source='weimar', windows=[0], harmony_windows=[])
    config = SimpleNamespace(**dict({k:v for k,v in vars(cfg).items() if k.isupper()}, HISTORY_PROB=0))
    x = prepare_example(rec, random.Random(1), config, task='refine', strength=0)
    tok = MelodyTokenizer()
    response = [tok.id_to_tok[i] for i in x['target'].tolist() if i != -100]
    assert [n.pitch for n in tokens_to_notes(response, tok)] == [64, 62]



def test_all_kit_sounds_are_decodable_and_auxiliary_percussion_is_masked():
    from improvnet.utils.melody_stems import DRUM_KIT_PITCHES
    anchor=Note('Acoustic Piano',72,90,0,200)
    script=attrs(72,0,200)+sum((attrs(p,0,20,'drum') for p in sorted(DRUM_KIT_PITCHES)),[])+['<E>']
    infer,_=scripted_inference(script)
    output=infer.generate(make_prompt('stem',['<S>','<E>'],target_instrument='drum'),
        'jazz',1.,['Acoustic Piano','drum'],500,temperature=0.,fixed_notes=[anchor],
        stem_add=True,target_instrument='drum',max_new_tokens=128)
    assert {n.pitch for n in output if n.instrument=='drum'}==DRUM_KIT_PITCHES
    assert [n for n in output if n.instrument!='drum']==[anchor]
    # The model explicitly asks for a conga. Legal-token masking must reject it.
    infer,_=scripted_inference(attrs(72,0,200)+attrs(60,0,20,'drum')+['<E>'])
    output=infer.generate(make_prompt('stem',['<S>','<E>'],target_instrument='drum'),
        'jazz',1.,['Acoustic Piano','drum'],500,temperature=0.,fixed_notes=[anchor],
        stem_add=True,target_instrument='drum',max_new_tokens=64)
    assert any(n.instrument=='drum' for n in output)
    assert all(n.pitch in DRUM_KIT_PITCHES for n in output if n.instrument=='drum')
    assert [n for n in output if n.instrument!='drum']==[anchor]


def test_standard_kit_midi_program_channel_and_pitched_notes_are_preserved(tmp_path):
    import mido
    notes=[Note('Acoustic Piano',60,90,0,500)]+[Note('drum',p,60,100+i*100,20) for i,p in enumerate((36,38,42,46,51))]
    path=tmp_path/'kit.mid'
    write_midi_notes(notes,path)
    messages=[m for track in mido.MidiFile(path).tracks for m in track]
    assert [m.program for m in messages if m.type=='program_change' and m.channel==9]==[0]
    assert {m.note for m in messages if m.type=='note_on' and m.velocity>0 and m.channel==9}=={36,38,42,46,51}
    assert {m.note for m in messages if m.type=='note_on' and m.velocity>0 and m.channel!=9}=={60}


def test_stem_transform_filters_auxiliary_drums_before_anchoring(monkeypatch):
    infer,_=scripted_inference([])
    infer.meta.update(version=9,tasks=list(TASKS),seq_len=512,chunk_tokens=64,history_tokens=64,
                      mix_response_tokens=128,skyline_tolerance_ms=50)
    infer.options=cfg
    original=ordered([Note('Acoustic Piano',60,90,0,500),Note('drum',60,80,100,20)])
    seen=[]
    def generate(prompt,*args,**kwargs):
        seen.extend(kwargs['fixed_notes']);infer.last_decode_stats={}
        return kwargs['fixed_notes']+[Note('drum',42,80,0,20)]
    monkeypatch.setattr(infer,'generate',generate)
    result,_=infer.transform(original,'stem','jazz',1.,random.Random(1),instruments=['Acoustic Piano','drum'],target_instrument='drum')
    assert seen==[original[0]] and {n.pitch for n in result if n.instrument=='drum'}=={42}
