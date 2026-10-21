"""Current V5 corruptions contracts."""
import random
import pytest
import torch
from dataclasses import replace
from melody_test_helpers import settings, mix, response, prompt_source, scripted_inference, attrs
from improvnet.model import melody_config as cfg
from improvnet.utils.melody import Note, MelodyTokenizer, TASKS, ordered, skyline, tokens_to_notes, harmonize_strength, make_prompt, mask_timing
from improvnet.utils.melody_corruptions import corrupt_stem_source, refinement_source, sample_stem_pitch_probability, corrupt_history, refinement_strength
from improvnet.utils.melody_data import prepare_example
from improvnet.utils.melody_ornaments import merge_nonornaments, ornament_spans, merge_ornaments, invert_contour
from collections import Counter


def test_v5_has_three_tasks_fresh_path_and_original_ar_initialization():
    assert TASKS == ('refine', 'harmonize', 'stem')
    assert cfg.TASK_WEIGHTS == (.20, .40, .40) and cfg.PROTOCOL_VERSION == 9
    assert cfg.SAVE_DIR.endswith('/melody_refinement_v5')
    assert cfg.DATA_FILE.name == 'melody_manifest_v5.json'
    assert '/ar_context_split_instrument/' in cfg.INIT_CHECKPOINT
    assert cfg.BATCH_SIZE == 24 and cfg.CHUNK_TOKENS == 512 and cfg.MIX_RESPONSE_TOKENS == 2048
    assert cfg.HISTORY_TOKENS == 2048 and cfg.SEQ_LEN == 8192


def test_stem_jitter_preserves_chords_group_order_bounds_and_duration_continuation():
    notes = ordered([Note(inst, pitch, 90, t, 1000)
                     for t in (0,10,20,4970,4990,5000,5010,5990)
                     for inst,pitch in [('Acoustic Piano',76),('Acoustic Bass',48)]])
    for seed in range(40):
        changed = corrupt_stem_source(notes, random.Random(seed), MelodyTokenizer(),
                                      settings(STEM_PITCH_SHIFT_PROB=0.), 6000)
        assert len(changed) == len(notes)
        for clean, noisy in zip(notes, changed):
            assert clean.instrument == noisy.instrument and clean.pitch == noisy.pitch
            assert clean.velocity == noisy.velocity
            assert abs(clean.onset-noisy.onset) <= 50 and abs(clean.duration-noisy.duration) <= 50
            assert 0 <= noisy.onset < 6000 and noisy.duration > 0
            assert noisy.onset % 10 == noisy.duration % 10 == 0
        groups = [changed[i:i+2] for i in range(0,len(changed),2)]
        assert all(g[0].onset == g[1].onset for g in groups)
        assert all(a[0].onset < b[0].onset for a,b in zip(groups,groups[1:]))
        assert changed[-1].onset + changed[-1].duration > 6000


def test_stem_pitch_corruption_targets_only_original_50ms_skyline_accompaniment():
    # The delayed highest attack is the rolled chord's melodic note.
    notes = ordered([Note('Acoustic Piano',p,90,t,300) for p,t in [(60,0),(70,10),(80,30)]])
    assert skyline(notes,50)[0] == [notes[-1]]
    result = corrupt_stem_source(notes,random.Random(7),MelodyTokenizer(),
                                 settings(STEM_PITCH_SHIFT_PROB=1.,STEM_TIMING_JITTER_MS=0),500,pitch_probability=1.)
    assert result[-1] == notes[-1]
    assert all(0 < abs(a.pitch-b.pitch) <= 7 for a,b in zip(notes[:-1],result[:-1]))
    assert [n.onset for n in result] == [n.onset for n in notes]


def test_stem_pitch_corruption_never_changes_monophonic_melody_or_drum_ids():
    melody = [Note('Acoustic Piano',60+i%8,90,i*100,100) for i in range(20)]
    drums = [Note('drum',42,80,i*100,20) for i in range(20)]
    source = ordered(melody+drums)
    result = corrupt_stem_source(source,random.Random(1),MelodyTokenizer(),
                                 settings(STEM_PITCH_SHIFT_PROB=1.),2000,pitch_probability=1.)
    assert [n.pitch for n in result] == [n.pitch for n in source]


def test_stem_pitch_error_rate_is_small_and_never_introduces_duplicate_attacks():
    notes = ordered([Note('Acoustic Piano',p,90,i*200,200)
                     for i in range(1000) for p in (60,61,80)])
    result = corrupt_stem_source(notes,random.Random(5),MelodyTokenizer(),
                                 settings(STEM_TIMING_JITTER_MS=0),200000)
    clean_by_time = {t: {n.pitch for n in notes if n.onset==t} for t in range(0,200000,200)}
    changed = sum(n.pitch not in clean_by_time[n.onset] for n in result)
    assert 50 <= changed <= 150  # 5% of 2,000 eligible accompaniment notes.
    assert all(n.pitch == 80 for n in result[::3])
    assert len({(n.instrument,n.pitch,n.onset) for n in result}) == len(result)


def test_stem_pitch_shifts_cannot_reorder_retained_chord_anchors_across_instruments():
    notes = ordered([Note(inst,p,90,0,300) for inst,p in
                     [('Tenor Sax',80),('Acoustic Piano',63),('Acoustic Bass',61),('Trumpet',59)]])
    for seed in range(80):
        result = corrupt_stem_source(notes,random.Random(seed),MelodyTokenizer(),
                                     settings(STEM_PITCH_SHIFT_PROB=1.,STEM_TIMING_JITTER_MS=0),500,pitch_probability=1.)
        assert [n.instrument for n in result] == [n.instrument for n in notes]
        assert result[0].pitch == 80
        assert all(abs(a.pitch-b.pitch)<=7 for a,b in zip(notes,result))


def test_stem_source_is_corrupted_but_full_mix_and_history_labels_are_clean():
    rec = mix(200)
    ex = prepare_example(rec,random.Random(17),settings(POLY_HISTORY_PROB=1.,RHYTHM_STYLE_PROB=1.),
                         task='stem',target_instrument='drum')
    clean = response(ex); noisy = tokens_to_notes(prompt_source(ex))
    retained = [n for n in clean if n.instrument!='drum']
    assert len(noisy) == len(retained) and noisy != retained
    assert ex['stem_name']=='drum' and ex['stem_mask'].any()
    assert ex['history_length'] and not ex['history_mono'] and not ex['history_corrupted']
    assert ex['style_mask'].sum() > 0 and ex['exposure'].shape == (7,)
    expected = ordered([replace(Note(*n),onset=n[3]-ex['start_ms']) for n in rec['notes']+rec['drums']
                        if ex['start_ms']<=n[3]<ex['end_ms']])
    assert clean == expected
    assert ex['source_length']<=512 and ex['response_length']+1<=2048
    # Validation denoising remains task-realistic but exactly reproducible.
    same = prepare_example(rec,random.Random(17),settings(POLY_HISTORY_PROB=1.,RHYTHM_STYLE_PROB=1.),
                           task='stem',target_instrument='drum')
    assert torch.equal(ex['input'],same['input']) and torch.equal(ex['target'],same['target'])


def test_ordinary_run_merge_covers_nonornaments_without_bridging_rest_or_motif():
    plain = [Note('Acoustic Piano',p,90,i*300,300) for i,p in enumerate((60,64,67,65,69,72))]
    assert not ornament_spans(plain)
    merged = merge_nonornaments(plain,1.,random.Random(4))
    assert 2 <= len(plain)-len(merged)+1 <= 4 and merged!=plain
    assert merged[0].onset==plain[0].onset
    assert max(n.onset+n.duration for n in merged)==max(n.onset+n.duration for n in plain)
    separated = [replace(n,onset=i*2000,duration=100) for i,n in enumerate(plain)]
    assert merge_nonornaments(separated,1.,random.Random(4)) == separated
    trill = [Note('Acoustic Piano',60+i%2,90,i*60,60) for i in range(6)]
    assert merge_nonornaments(trill,1.,random.Random(4)) == trill


def test_generic_merge_has_ten_percent_extra_weight():
    class PickGeneric(random.Random):
        def choices(self,names,weights):
            self.seen = dict(zip(names,weights))
            return ['generic_merge' if 'generic_merge' in names else 'ornaments']
    notes = [Note('Acoustic Piano',p,90,i*300,300) for i,p in enumerate((60,64,67,65,69,72))]
    current = PickGeneric(4)
    source = refinement_source(notes,1.,current,MelodyTokenizer(),settings(MAX_CORRUPTIONS=1))
    assert current.seen == pytest.approx(dict(contour=.40,pitch=.25,timing=.25,generic_merge=.10))
    assert len(tokens_to_notes(source)) < len(notes)


def test_retired_orchestration_is_not_a_v5_training_task():
    with pytest.raises(ValueError,match='Unknown v5 task'):
        prepare_example(mix(20),random.Random(1),task='orchestrate')


@pytest.mark.parametrize('count',(1,2,3,5,7,127))
@pytest.mark.parametrize('fraction',(.6,.73,1.))
def test_timing_masking_strictly_exceeds_half_even_in_small_groups(count,fraction):
    notes=[Note('Acoustic Piano',72,90,i*100,80) for i in range(count)]
    source=mask_timing(notes,fraction,random.Random(42),round_up=True)
    actual=source.count('<MASK>')/(2*count)
    assert actual>=fraction and actual>.5
    assert sum(isinstance(t,tuple) and t[0]=='note' for t in source)==count
    if fraction==1:assert actual==1


def test_severity_sampler_includes_full_masking_and_never_weak_examples():
    rng=random.Random(4)
    fractions=[harmonize_strength(rng) for _ in range(2000)]
    assert all(.6<=f<=1 for f in fractions)
    assert 430<=sum(f==1 for f in fractions)<=570
    assert harmonize_strength(rng,0)==.6 and harmonize_strength(rng,.9)==.9


@pytest.mark.parametrize('requested',(0.,.2,.6,.9,1.))
def test_real_harmonize_examples_mask_heavily_and_keep_clean_targets(requested):
    record=mix(60)
    example=prepare_example(record,random.Random(14),settings(HISTORY_PROB=0.,POLY_HISTORY_PROB=0.,RHYTHM_STYLE_PROB=0.),
                            task='harmonize',strength=requested)
    source=prompt_source(example)
    note_count=sum(isinstance(t,tuple) and t[0]=='note' for t in source)
    assert source.count('<MASK>')/(2*note_count)>=.6
    assert example['timing_mask_fraction']>.5
    assert example['strength'].item()==pytest.approx(max(.6,requested))
    expected=ordered([replace(Note(*n),onset=n[3]-example['start_ms']) for n in record['notes']
                      if example['start_ms']<=n[3]<example['end_ms']])
    assert response(example)==expected and example['source_length']<=512
    assert len(example['input'])+3<=8192


def test_all_three_tasks_show_history_about_three_quarters_of_eligible_examples():
    record=mix(200)
    for task in TASKS:
        shown=0
        for seed in range(100):
            ex=prepare_example(record,random.Random(seed),settings(RHYTHM_STYLE_PROB=0.),task=task)
            shown+=bool(ex['history_length'])
            if ex['history_length']:assert ex['history_mono']==(task=='refine')
        assert 60<=shown<=88


def test_dynamic_pitch_rate_includes_explicit_pitch_clean_and_full_range():
    rng=random.Random(9)
    rates=[sample_stem_pitch_probability(rng,cfg) for _ in range(2000)]
    assert all(0<=rate<=.05 for rate in rates)
    assert 330<=rates.count(0.)<=470
    assert any(0<r<.01 for r in rates) and any(.04<r<=.05 for r in rates)


def test_pitch_clean_input_and_seven_semitone_errors_protect_melody_and_drums():
    source=ordered([Note('Acoustic Piano',p,90,0,200) for p in (90,70,40)]+[Note('drum',36,90,0,20)])
    clean=corrupt_stem_source(source,random.Random(2),MelodyTokenizer(),
        settings(STEM_PITCH_CLEAN_PROB=1.,STEM_TIMING_JITTER_MS=0),500)
    assert clean==source
    class RaiseBySeven(random.Random):
        def choice(self,values):return 7 if 7 in values else super().choice(values)
    noisy=corrupt_stem_source(source,RaiseBySeven(2),MelodyTokenizer(),
                             settings(STEM_TIMING_JITTER_MS=0),500,pitch_probability=1.)
    assert [n.pitch for n in noisy]==[90,77,47,36]
    assert len(noisy)==len(source)
    assert all(a.instrument==b.instrument for a,b in zip(source,noisy))


def test_model_can_repair_seven_semitones_but_not_change_skyline_or_exceed_cap():
    anchors=ordered([Note('Acoustic Piano',p,90,i*100,200) for i in range(21) for p in (72,60)])
    script=[]
    for note in anchors:script+=attrs(79 if note.pitch==72 else 67,note.onset,200)
    infer,_=scripted_inference(script+['<E>'])
    infer.meta.update(version=9,stem_pitch_shift_semitones=7,stem_pitch_shift_prob=.05)
    result=infer.generate(make_prompt('stem',['<S>','<E>'],target_instrument='drum'),
        'jazz',1.,['Acoustic Piano','drum'],2200,temperature=0.,fixed_notes=anchors,
        stem_add=True,target_instrument='drum',stop_at_window_end=True,max_new_tokens=256)
    assert len(result)==len(anchors)
    assert sum(n.pitch==72 for n in result)==21
    assert sum(n.pitch==67 for n in result)==1
    assert infer.last_decode_stats['retained_pitch_corrections']==1


def test_inference_harmonize_enforces_mask_floor_and_reports_actual_fraction(monkeypatch):
    infer,_=scripted_inference([])
    infer.meta.update(version=9,tasks=list(TASKS),seq_len=512,chunk_tokens=18,history_tokens=64,
                      mix_response_tokens=128,harmonize_expansion=4.,skyline_tolerance_ms=50)
    infer.options=cfg
    notes=[Note('Acoustic Piano',72,90,t,80) for t in (0,200)]
    calls=[]
    def generate(prompt,genre,strength,*args,**kwargs):
        calls.append((prompt,strength))
        infer.last_decode_stats={}
        return notes
    monkeypatch.setattr(infer,'generate',generate)
    result,logs=infer.transform(notes,'harmonize','jazz',0.,random.Random(5),instruments=['Acoustic Piano'])
    assert result==notes and calls[0][1]==.6
    assert calls[0][0][0]=='<HARMONIZE>' and calls[0][0].count('<MASK>')==4
    assert logs[0]['timing_mask_fraction']==1.


class PickEdit(random.Random):
    def __init__(self, edit, sign=1):
        super().__init__(0)
        self.edit, self.sign = edit, sign

    def choices(self, names, weights):
        return [self.edit] if self.edit in names else super().choices(names, weights)

    def random(self):
        return 0.

    def choice(self, values):
        return max(values) if self.sign > 0 else min(values)

    def randint(self, a, b):
        return b if self.sign > 0 else a

    def sample(self, values, count):
        return random.Random(0).sample(list(values), count)

    def randrange(self, *args):
        return 0


def trill():
    return [Note('Acoustic Piano', p, 90, 100+i*60, 60) for i,p in enumerate((60,62,60,62))]


def plain():
    return [Note('Acoustic Piano', p, 90, 100+i*300, 300) for i,p in enumerate((60,65,70,65,60))]


@pytest.mark.parametrize('strength', (0., .1, 1.))
def test_ornaments_always_merge_and_an_independent_pitch_error_can_coexist(strength):
    original=trill()
    source=tokens_to_notes(refinement_source(original,strength,PickEdit('pitch'),MelodyTokenizer(),cfg))
    assert source==[replace(original[0],pitch=67 if strength else 60,duration=240)]
    assert original==trill()
    assert refinement_strength(original,strength,cfg)==max(strength,.75)


def test_no_ornaments_still_get_an_extra_edit_but_zero_strength_can_remain_clean():
    notes=plain()
    source=tokens_to_notes(refinement_source(notes,1.,PickEdit('pitch'),MelodyTokenizer(),cfg))
    assert source==[replace(n,pitch=n.pitch+7) for n in notes]
    assert tokens_to_notes(refinement_source(notes,0.,PickEdit('pitch'),MelodyTokenizer(),cfg))==notes


@pytest.mark.parametrize('sign', (-1,1))
def test_refinement_timing_reaches_fifty_ms_in_both_directions_without_proportional_duration_errors(sign):
    notes=[replace(n,duration=4000) for n in plain()]
    source=tokens_to_notes(refinement_source(notes,1.,PickEdit('timing',sign),MelodyTokenizer(),cfg))
    assert source==[replace(n,onset=n.onset+sign*50,duration=n.duration+sign*50) for n in notes]


def test_dense_refinement_timing_keeps_chords_and_attack_order():
    notes=ordered([Note('Acoustic Piano',p,90,t,100) for t in (0,10,20,30) for p in (72,60)])
    source=tokens_to_notes(refinement_source(notes,1.,PickEdit('timing'),MelodyTokenizer(),cfg))
    assert len(source)==len(notes)
    for a,b in zip(notes,source):
        assert a.pitch==b.pitch and abs(a.onset-b.onset)<=50
        assert abs(a.duration-b.duration)<=50 and b.onset%10==b.duration%10==0
    assert len({n.onset for n in source})==4
    assert all(source[i].onset==source[i+1].onset for i in range(0,len(source),2))


@pytest.mark.parametrize('pitch', (0,1,60,126,127))
def test_pitch_corruption_stays_in_midi_range_and_uses_a_nonzero_offset(pitch):
    notes=[Note('Acoustic Piano',pitch,90,100,100)]
    for sign in (-1,1):
        source=tokens_to_notes(refinement_source(notes,1.,PickEdit('pitch',sign),MelodyTokenizer(),cfg))
        assert 0<=source[0].pitch<=127 and 1<=abs(source[0].pitch-pitch)<=7


@pytest.mark.parametrize('edit', ('pitch','timing'))
@pytest.mark.parametrize('sign', (-1,1))
def test_repeated_history_edits_cannot_accumulate_outside_original_bounds(edit,sign):
    notes=[Note('Acoustic Piano',60,90,100+i*200,4000) for i in range(8)]
    source=corrupt_history(notes,PickEdit(edit,sign),fraction=1.,end_ms=5000)
    assert len(source)==len(notes)
    for a,b in zip(notes,source):
        assert abs(a.pitch-b.pitch)<=7 and abs(a.onset-b.onset)<=50 and abs(a.duration-b.duration)<=50
        assert b.onset%10==b.duration%10==0
    if edit=='pitch':assert source[0].pitch==60+sign*7
    else:assert source[0].onset==100+sign*50 and source[0].duration==4000+sign*50


def test_history_additions_stay_bounded_when_copying_an_already_added_note():
    class CopyLast(PickEdit):
        def randrange(self, stop):return stop-1
    notes=[Note('Acoustic Piano',60,90,100+i*200,400) for i in range(5)]
    source=corrupt_history(notes,CopyLast('add'),fraction=1.,end_ms=2000)
    added=[n for n in source if n not in notes]
    assert added and all(n.pitch==67 and n.onset==950 for n in added)


def test_training_condition_reflects_mandatory_merge_and_target_stays_clean():
    notes=trill()
    record=dict(notes=[[n.instrument,n.pitch,n.velocity,n.onset,n.duration] for n in notes],
                drums=[],source='fixture',genre='jazz',path='trill.mid')
    example=prepare_example(record,PickEdit('pitch'),settings(HISTORY_PROB=0.,RHYTHM_STYLE_PROB=0.),
                            task='refine',strength=0.)
    assert example['requested_strength']==0. and example['strength'].item()==.75
    clean=[replace(n,onset=n.onset-notes[0].onset) for n in notes]
    assert response(example)==clean
    assert tokens_to_notes(prompt_source(example))==[replace(clean[0],duration=240)]
    assert example['source_length']<=512 and example['response_length']+1<=512
    assert len(example['input'])+3<=8192


def test_inference_uses_same_merge_strength_instead_of_bypassing_zero_request(monkeypatch):
    infer,_=scripted_inference([])
    infer.meta.update(version=9,tasks=list(TASKS),seq_len=512,chunk_tokens=64,history_tokens=64,
                      mix_response_tokens=128,skyline_tolerance_ms=50)
    infer.options=cfg
    notes=trill();seen=[]
    def generate(prompt,genre,strength,*args,**kwargs):
        seen.append(strength)
        infer.last_decode_stats={}
        return [replace(n,onset=n.onset-notes[0].onset) for n in notes]
    monkeypatch.setattr(infer,'generate',generate)
    result,logs=infer.transform(notes,'refine','jazz',0.,PickEdit('pitch'),monophonic=True)
    assert result==notes and seen==[.75]
    assert logs[0]['requested_strength']==0. and logs[0]['strength']==.75


@pytest.mark.parametrize('pitches',([60,61,60,61,60],[60,62,60,59,60],[60,61,62,63,64]))
def test_ornament_candidates_merge_to_the_first_pitch(pitches):
 notes=[Note('Acoustic Piano',p,90,i*60,60) for i,p in enumerate(pitches)]
 assert ornament_spans(notes)==[(0,len(notes))]
 result=merge_ornaments(notes,1.,random.Random(2))
 assert result==[replace(notes[0],duration=len(notes)*60)]
 # Slow scales / long rests are not ornaments.
 assert not ornament_spans([replace(n,onset=i*700,duration=300) for i,n in enumerate(notes)])


def test_contour_reflection_preserves_first_pitch_and_timing_without_clipping():
 notes=[Note('Acoustic Piano',p,90,i*200,200) for i,p in enumerate([60,62,65,64,67,69,68,70,71,72])]
 result=invert_contour(notes,1.,random.Random(5))
 assert result!=notes and [n.onset for n in result]==[n.onset for n in notes]
 assert all(n.duration==200 and 0<=n.pitch<=127 for n in result)
 original=[b.pitch-a.pitch for a,b in zip(notes,notes[1:])]
 changed=[b.pitch-a.pitch for a,b in zip(result,result[1:])]
 assert any(a==-b and a!=0 for a,b in zip(original,changed))


def test_jazz_instrument_weights_use_canonical_supported_classes():
 from improvnet.utils.ar_utils import ProcessData
 from improvnet.utils.melody_stems import JAZZ_INSTRUMENTS,instrument_weight
 assert JAZZ_INSTRUMENTS<=set(ProcessData().INSTRUMENT_CLASSES)
 assert instrument_weight('Soprano/Alto Sax')==instrument_weight('Tenor Sax')>instrument_weight('Acoustic Piano')


def test_refinement_additional_strength_levels_are_balanced_and_include_ten_percent():
    from improvnet.utils.melody import sample_strength
    counts=Counter(sample_strength(random.Random(seed)) for seed in range(20000))
    assert set(counts)=={0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.}
    assert 850<counts[0.]<1150
    assert all(1700<counts[level/10]<2100 for level in range(1,11))


@pytest.mark.parametrize('edit', ('pitch','timing'))
def test_requested_ten_to_hundred_percent_controls_actual_additional_note_coverage(edit):
    # Slow, wide-interval phrases avoid the independent mandatory-ornament pass.
    original=[Note('Acoustic Piano',60+i%2*7,90,100+i*300,200) for i in range(100)]
    for level in (.1,.5,1.):
        source=tokens_to_notes(refinement_source(original,level,PickEdit(edit),MelodyTokenizer(),cfg))
        unchanged=sum((Counter(original)&Counter(source)).values())
        assert 1-unchanged/len(original)==pytest.approx(level)
        assert len(source)==len(original)
        for before,after in zip(original,source):
            assert abs(after.pitch-before.pitch)<=7
            assert abs(after.onset-before.onset)<=50 and abs(after.duration-before.duration)<=50
            assert after.onset%10==after.duration%10==0


def test_training_removes_auxiliary_percussion_from_current_mix_and_history_without_changing_record():
    from copy import deepcopy
    from improvnet.utils.melody_stems import DRUM_KIT_PITCHES
    record=mix(80)
    record['drums'] += [['drum',p,80,t,20] for t in range(0,16000,100) for p in (39,54,56,60,82)]
    record['_augment']=True
    original=deepcopy(record)
    for task in TASKS:
        example=prepare_example(record,random.Random(14),settings(HISTORY_PROB=1.,POLY_HISTORY_PROB=1.,HISTORY_CORRUPTION_PROB=1.),
                                task=task,target_instrument='Acoustic Bass' if task=='stem' else None)
        assert example['history_length']>0
        assert example['nonkit_drum_notes_filtered']==800
        inst=None;drums=[]
        for token in [MelodyTokenizer().id_to_tok[i] for i in example['input'].tolist()]:
            if isinstance(token,tuple) and token[0]=='instrument':inst=token[1]
            if isinstance(token,tuple) and token[0]=='note' and inst=='drum':drums.append(token[1])
        assert all(p in DRUM_KIT_PITCHES for p in drums)
        if task=='stem':assert drums and all(n.pitch in DRUM_KIT_PITCHES for n in response(example) if n.instrument=='drum')
    assert record==original


def test_auxiliary_only_percussion_routes_to_another_task_without_parse_retries(tmp_path,monkeypatch):
    from melody_test_helpers import corpus
    from improvnet.utils.melody_data import MelodyDataset,DATA_STAT_NAMES
    config,_=corpus(tmp_path)
    config.TASK_WEIGHTS=(0.,0.,1.)
    dataset=MelodyDataset(config.DATA_FILE,'train',config)
    record=dict(mix(30),drums=[['drum',60,80,100,20]])
    record['notes']=[n for n in record['notes'] if n[0]=='Acoustic Piano']
    monkeypatch.setattr(dataset,'load_record',lambda row:record)
    example=dataset[0]
    assert TASKS[example['task']] in ('refine','harmonize')
    stats=dict(zip(DATA_STAT_NAMES,example['data_stats'].tolist()))
    assert stats['routed']==1 and stats['retries']==0 and stats['attempts']==1


def test_electronic_hihat_edges_normalize_without_losing_rhythm_or_remapping_other_instruments():
    from improvnet.utils.melody_stems import kit_only_notes
    notes=[Note('Acoustic Piano',22,90,0,400),Note('drum',22,60,100,20),
           Note('drum',42,90,100,20),Note('drum',26,80,300,30),Note('drum',60,80,400,20)]
    normalized=kit_only_notes(notes)
    assert normalized==[notes[0],notes[2],replace(notes[3],pitch=46)]
    assert [n.onset for n in normalized if n.instrument=='drum']==[100,300]
    assert kit_only_notes(normalized)==normalized
    record=mix(30)
    record['drums']=[['drum',22,60,i*200,20] for i in range(30)]+[['drum',26,80,i*200+100,20] for i in range(30)]
    example=prepare_example(record,random.Random(4),settings(),task='stem',target_instrument='drum')
    assert {n.pitch for n in response(example) if n.instrument=='drum'}<={42,46}
    assert example['drum_articulations_normalized']==60 and example['nonkit_drum_notes_filtered']==0
