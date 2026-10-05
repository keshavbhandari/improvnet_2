"""MIDI serialization, skyline, audible demo and deterministic modifications."""

from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import unittest

import mido
import numpy as np

from improvnet.metrics.pitch_similarity import PitchSimilarityMetric
from improvnet.metrics.pitch_similarity.demo import run_demo
from improvnet.metrics.pitch_similarity.midi import NoteEvent, clip_segment, modify_segment, pitched_sequence, read_midi, save_midi, skyline


def tune():
    pitches = [60,62,64,67,65,64,62,60,67,69,71,72,71,69,67,64]
    return [NoteEvent(p,i*.5,i*.5+.35,80,0,0) for i,p in enumerate(pitches)]


class MidiSimilarityTests(unittest.TestCase):
    def test_midi_tempo_and_program_changes_preserve_original_event_interpretation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'tempo.mid'
            midi = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            midi.tracks.append(track)
            track.extend([mido.MetaMessage('set_tempo',tempo=500000,time=0),
                          mido.Message('program_change',program=0,time=0),
                          mido.Message('note_on',note=60,velocity=80,time=240),
                          mido.MetaMessage('set_tempo',tempo=1000000,time=240),
                          mido.Message('note_off',note=60,velocity=0,time=240),
                          mido.Message('program_change',program=40,time=0),
                          mido.Message('note_on',note=64,velocity=80,time=0),
                          mido.Message('note_off',note=64,velocity=0,time=240)])
            midi.save(path)
            notes = read_midi(path)
            self.assertAlmostEqual(notes[0].onset,.25)
            self.assertAlmostEqual(notes[0].end,1.)
            self.assertEqual([note.program for note in notes],[0,40])
            self.assertAlmostEqual(notes[1].end,1.5)
            # Fail explicitly rather than silently replay earlier notes with
            # the last program on a changing channel.
            with self.assertRaises(ValueError):
                save_midi(notes,Path(directory)/'unsupported.mid')

    def setUp(self):
        self.notes = tune()
        self.metric = PitchSimilarityMetric()

    def test_midi_roundtrip_pitch_order_velocity_and_timing(self):
        notes = self.notes+[NoteEvent(36,.25,.4,90,9,0),NoteEvent(55,1.,1.3,60,1,40)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'tune.mid'
            save_midi(notes,path)
            recovered = read_midi(path)
            self.assertEqual(pitched_sequence(notes),pitched_sequence(recovered))
            self.assertAlmostEqual(self.metric.evaluate_pitches(pitched_sequence(notes),pitched_sequence(recovered)).score,1)
            self.assertEqual(len(recovered),len(notes))
            parsed = mido.MidiFile(path)
            self.assertTrue(all(message.time>=0 for track in parsed.tracks for message in track))
            with self.assertRaises(FileExistsError):
                save_midi(notes,path)

    def test_segment_attack_selection_and_tail_clipping(self):
        notes = [NoteEvent(60,0,4),NoteEvent(64,1,3),NoteEvent(67,2,5)]
        segment = clip_segment(notes,1,1)
        self.assertEqual(segment,[NoteEvent(64,0,1)])
        self.assertEqual(notes[1].onset,1)
        with self.assertRaises(ValueError):
            clip_segment(notes,-1,1)

    def test_monotonic_timing_and_velocity_changes_leave_pitch_score_one(self):
        modified = modify_segment(self.notes,'timing',random.Random(8))
        self.assertEqual(pitched_sequence(modified),pitched_sequence(self.notes))
        self.assertNotEqual(modified[0].velocity,self.notes[0].velocity)
        self.assertAlmostEqual(self.metric.evaluate_pitches(pitched_sequence(self.notes),pitched_sequence(modified)).score,1)

    def test_chord_order_is_velocity_and_instrument_independent(self):
        a = [NoteEvent(60,0,1,80,0),NoteEvent(64,0,1,80,1),NoteEvent(67,0,1,80,2)]
        b = [replace(note,velocity=40,channel=3-i) for i,note in enumerate(a)]
        self.assertEqual(pitched_sequence(a),pitched_sequence(b))

    def test_skyline_overlap_endpoints_and_separate_same_pitch_attacks(self):
        notes = [NoteEvent(60,0,3),NoteEvent(72,1,2),NoteEvent(60,3,4),NoteEvent(90,0,5,80,9)]
        result = skyline(notes)
        self.assertEqual([(note.pitch,note.onset,note.end) for note in result],[(60,0,1),(72,1,2),(60,2,3),(60,3,4)])
        self.assertEqual(pitched_sequence(notes,'skyline'),[60,72,60,60])

    def test_skyline_is_explicitly_timing_dependent(self):
        a = [NoteEvent(60,0,2),NoteEvent(72,0,1)]
        b = [replace(a[0],end=.5),a[1]]
        self.assertNotEqual(pitched_sequence(a,'skyline'),pitched_sequence(b,'skyline'))
        self.assertEqual(pitched_sequence(a),pitched_sequence(b))

    def test_all_modification_modes_deterministic_valid_and_nonmutating(self):
        original = list(self.notes)
        for mode in ('copy','timing','ornament','substitute','shuffle','transpose','mixed','unrelated'):
            with self.subTest(mode=mode):
                a = modify_segment(original,mode,random.Random(3),probability=.5)
                b = modify_segment(original,mode,random.Random(3),probability=.5)
                self.assertEqual(a,b)
                self.assertTrue(all(0<=note.pitch<=127 and note.end>note.onset for note in a))
        self.assertEqual(original,self.notes)
        with self.assertRaises(ValueError):
            modify_segment([NoteEvent(127,0,1)],'transpose',random.Random(0),transpose=1)

    def test_demo_modified_segment_writes_scored_midi_and_breakdown(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.mid'
            output = Path(directory)/'demo'
            save_midi(self.notes,source)
            report = run_demo(source,output,start=0,seconds=4,modification='ornament',probability=.5,plot=True)
            from xml.etree import ElementTree
            self.assertTrue(ElementTree.parse(output/'matrices.svg').getroot().tag.endswith('svg'))
            with (output/'scores.json').open() as handle:
                saved = json.load(handle)
            self.assertIn('pitch_identity',saved['scores'])
            self.assertIn('pattern_preservation',saved['scores'])
            self.assertIn('structural_preservation',saved['scores'])
            matrices = np.load(output/'matrices.npz')
            self.assertEqual(matrices['cross'].shape,(len(report['reference_pitches']),len(report['generated_pitches'])))
            self.assertEqual(pitched_sequence(read_midi(output/'original.mid')),report['reference_pitches'])
            self.assertEqual(pitched_sequence(read_midi(output/'comparison.mid')),report['generated_pitches'])
            again = self.metric.evaluate_pitches(report['reference_pitches'],report['generated_pitches'])
            self.assertEqual(again.to_dict(),report['scores'])
            with self.assertRaises(FileExistsError):
                run_demo(source,output,start=0,seconds=4)

    def test_demo_two_segments_and_external_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.mid'
            save_midi(self.notes,source)
            two = run_demo(source,Path(directory)/'two',mode='two-segments',start=0,second_start=4,seconds=3)
            self.assertNotEqual(two['reference_pitches'],two['generated_pitches'])
            same = run_demo(source,Path(directory)/'same',mode='compare',generated_path=source,start=0,second_start=0,seconds=3)
            self.assertAlmostEqual(same['scores']['score'],1)
            with self.assertRaises(ValueError):
                run_demo(source,Path(directory)/'missing',mode='compare',start=0,seconds=3)

    def test_random_start_reproducible_and_second_segment_is_disjoint(self):
        from improvnet.metrics.pitch_similarity.midi import random_segment_start
        a = random_segment_start(self.notes,2,random.Random(4))
        self.assertEqual(a,random_segment_start(self.notes,2,random.Random(4)))
        b = random_segment_start(self.notes,2,random.Random(5),exclude=(a,a+2))
        self.assertTrue(b+2<=a or b>=a+2)

    def test_bad_note_events_and_empty_midi_rejected(self):
        for kwargs in ({'pitch':128},{'pitch':True},{'onset':-1},{'end':0},{'velocity':0},{'onset':float('nan')}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                NoteEvent(**{'pitch':60,'onset':0,'end':1,**kwargs})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_midi([],Path(directory)/'empty.mid')


if __name__=='__main__':
    unittest.main()
