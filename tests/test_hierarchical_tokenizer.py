"""Representation tests independent of training, CUDA and legacy tokenizers."""

from copy import deepcopy
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import mido
import numpy as np

from improvnet.tokenizer.hierarchical import (
    FORMAT_VERSION, EncodedPiece, GridConfig, HierarchicalTokenizer, Note,
    TempoMap, normalize_piano_notes, read_performance,
)
from improvnet.tokenizer.roundtrip import compare_notes, main as roundtrip_main


def track(events):
    """Absolute tick → MIDI delta ticks; preserve input ordering at ties."""
    output, previous = mido.MidiTrack(), 0
    for tick, message in sorted(events, key=lambda e: e[0]):
        output.append(message.copy(time=tick - previous))
        previous = tick
    output.append(mido.MetaMessage("end_of_track"))
    return output


def midi(notes, *, ppq=1000, programs=None, messages=()):
    # At 120bpm, 2*ppq ticks = 1 second.
    events = list(messages)
    for channel, program in (programs or {0: 0}).items():
        events.append((0, mido.Message("program_change", channel=channel, program=program)))
    for channel, pitch, velocity, start, end in notes:
        events.append((round(start * 2 * ppq), mido.Message("note_on", channel=channel, note=pitch, velocity=velocity)))
        events.append((round(end * 2 * ppq), mido.Message("note_off", channel=channel, note=pitch)))
    result = mido.MidiFile(type=1, ticks_per_beat=ppq)
    result.tracks.append(track(events))
    return result


class HierarchicalTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = HierarchicalTokenizer()

    def roundtrip(self, source):
        piece = self.tokenizer.encode(source)
        decoded = self.tokenizer.decode(piece, strict=True)
        comparison = compare_notes(read_performance(source)["notes"], read_performance(decoded)["notes"])
        self.assertTrue(comparison["pitch_velocity_instrument_counts_match"], comparison)
        return piece, decoded, comparison

    def test_note_holds_across_bin_boundary_without_extra_onset(self):
        piece, _, errors = self.roundtrip(midi([(0, 60, 92, .99, 1.07)]))
        self.assertEqual(piece.velocity.shape, (2, 128, 100))
        self.assertEqual(piece.onset.shape, piece.velocity.shape)
        self.assertEqual(piece.velocity[0, 60, 99], 92)
        self.assertTrue(np.all(piece.velocity[1, 60, :7] == 92))
        self.assertEqual(piece.onset[0, 60, 99], 1)
        self.assertFalse(piece.onset[1, 60].any())
        self.assertEqual(piece.coarse_onsets[:, 60].tolist(), [1, 0])
        self.assertEqual(piece.coarse_velocity[:, 60].tolist(), [92, 92])
        self.assertTrue(piece.valid_mask[1, :7].all())
        self.assertFalse(piece.valid_mask[1, 7:].any())
        self.assertLess(errors["max_onset_error_ms"], .01)

    def test_adjacent_repeated_notes_are_separate_attacks(self):
        piece, output, _ = self.roundtrip(midi([(0, 60, 64, 0, .2), (0, 60, 64, .2, .4),
                                               (0, 60, 95, .4, .6)]))
        self.assertEqual(piece.velocity[0, 60, 20], 64)
        self.assertEqual(piece.velocity[0, 60, 40], 95)
        self.assertEqual(np.flatnonzero(piece.onset[0, 60]).tolist(), [0, 20, 40])
        self.assertEqual([n.velocity for n in read_performance(output)["notes"]], [64, 64, 95])
        self.assertEqual(piece.coarse_onsets[0, 60], 3)

    def test_same_pitch_overlap_cuts_previous_key_duration(self):
        source = midi([(0, 60, 64, 0, .4), (0, 60, 80, .1, .5)])
        piece = self.tokenizer.encode(source)
        notes = read_performance(self.tokenizer.decode(piece, strict=True))["notes"]
        self.assertEqual(piece.metadata["normalization"]["overlap_normalizations"], 1)
        self.assertEqual(piece.metadata["normalization"]["same_frame_collisions"], 0)
        np.testing.assert_allclose([(n.start, n.end) for n in notes], [(0, .1), (.1, .5)])
        self.assertEqual([n.velocity for n in notes], [64, 80])
        self.assertTrue(np.all(piece.velocity[0, 60, :10] == 64))
        self.assertTrue(np.all(piece.velocity[0, 60, 10:50] == 80))

    def test_same_frame_collision_latest_true_time_attack_wins(self):
        short = midi([(0, 60, 64, .001, .003), (0, 60, 80, .004, .006)])
        piece = self.tokenizer.encode(short)
        notes = read_performance(self.tokenizer.decode(piece, strict=True))["notes"]
        self.assertEqual(piece.metadata["normalization"], dict(overlap_normalizations=0,
                         same_frame_collisions=1, collision_frames=1, short_notes_extended=0))
        self.assertEqual(piece.metadata["source_piano_notes"], 2)
        self.assertEqual(piece.metadata["piano_notes"], 1)
        self.assertEqual([(n.pitch, n.velocity, n.start, n.end) for n in notes], [(60, 80, 0, .01)])

    def test_exact_time_collision_uses_stable_source_order_not_max_velocity(self):
        piece = self.tokenizer.encode(midi([(0, 60, 120, .1, .3), (0, 60, 30, .1, .4)]))
        notes = read_performance(self.tokenizer.decode(piece, strict=True))["notes"]
        self.assertEqual([n.velocity for n in notes], [30])
        self.assertAlmostEqual(notes[0].end, .4)
        self.assertEqual(piece.metadata["normalization"]["same_frame_collisions"], 1)
        self.assertEqual(piece.metadata["normalization"]["overlap_normalizations"], 1)

    def test_overlap_tail_does_not_resume_after_new_attack(self):
        # Test directly: FIFO MIDI pairing cannot disambiguate nested unison offs.
        notes = [Note(0, 0, 0, 60, 80, 0, .9, 1), Note(0, 0, 0, 60, 40, .1, .2, 2)]
        normalized, report = normalize_piano_notes(notes, GridConfig())
        self.assertEqual([(n.start, n.end, n.velocity) for n in normalized], [(0, .1, 80), (.1, .2, 40)])
        self.assertEqual(report["overlap_normalizations"], 1)
        # Displaced same-frame attack must not donate its long release either.
        notes[1] = Note(0, 0, 0, 60, 40, .004, .02, 2)
        normalized, report = normalize_piano_notes(notes, GridConfig())
        self.assertEqual([(n.start, n.end, n.velocity) for n in normalized], [(0, .02, 40)])
        self.assertEqual(report["same_frame_collisions"], 1)

    def test_collision_at_coarse_boundary_and_multiple_collision_counts(self):
        source = midi([(0, 60, 90, .995, .999), (0, 60, 45, 1.004, 1.008),
                       (0, 72, 80, 0, .001), (0, 72, 70, .002, .003), (0, 72, 60, .004, .005)])
        piece = self.tokenizer.encode(source)
        self.assertEqual(piece.onset[1, 60, 0], 1)
        self.assertEqual(piece.velocity[1, 60, 0], 45)
        self.assertEqual(piece.coarse_onsets[:, 60].tolist(), [0, 1])
        self.assertEqual(piece.metadata["normalization"]["same_frame_collisions"], 3)
        self.assertEqual(piece.metadata["normalization"]["collision_frames"], 2)
        self.assertEqual(len(read_performance(self.tokenizer.decode(piece, strict=True))["notes"]), 2)

    def test_chords_independent_pitches_and_velocity_runs(self):
        piece, output, _ = self.roundtrip(midi([(0, p, v, .1, .4) for p, v in [(48, 55), (60, 70), (64, 90), (67, 110)]]))
        for pitch, velocity in [(48, 55), (60, 70), (64, 90), (67, 110)]:
            self.assertTrue(np.all(piece.velocity[0, pitch, 10:40] == velocity))
            self.assertEqual(np.flatnonzero(piece.onset[0, pitch]).tolist(), [10])
        self.assertEqual(len(read_performance(output)["notes"]), 4)
        self.assertFalse(any(piece.metadata["normalization"].values()))

    def test_same_velocity_reattack_exactly_at_boundary(self):
        piece, output, _ = self.roundtrip(midi([(0, 60, 80, .9, 1), (0, 60, 80, 1, 1.1)]))
        self.assertEqual(piece.onset[1, 60, 0], 1)
        self.assertEqual(piece.coarse_onsets[:, 60].tolist(), [1, 1])
        self.assertEqual(len(read_performance(output)["notes"]), 2)

    def test_very_short_notes_extend_one_frame_and_padding_stays_zero(self):
        piece, output, _ = self.roundtrip(midi([(0, 60, 83, .006, .007)]))
        self.assertEqual(piece.n_frames, 2)
        self.assertEqual(piece.metadata["normalization"]["short_notes_extended"], 1)
        self.assertEqual(piece.velocity[0, 60, 1], 83)
        self.assertEqual(piece.onset[0, 60, 1], 1)
        self.assertFalse(piece.velocity[..., 2:].any())
        self.assertFalse(piece.onset[..., 2:].any())
        self.assertFalse(piece.pedal[..., 2:].any())
        note = read_performance(output)["notes"][0]
        self.assertAlmostEqual(note.start, .01)
        self.assertAlmostEqual(note.end, .02)

    def test_aligned_context_preserves_drums_and_cross_bin_sustains(self):
        source = midi([(0, 60, 80, 0, .5), (2, 72, 99, .95, 2.05), (9, 36, 74, 1.0, 1.2)],
                      programs={0: 0, 2: 40})
        piece, _, errors = self.roundtrip(source)
        self.assertEqual(piece.events.shape, (2, 11))
        self.assertEqual(piece.event_ptr.tolist(), [0, 1, 2, 2])
        self.assertEqual(piece.events[0, 6:10].tolist(), [0, 95, 2, 5])
        self.assertEqual(piece.events[1, 3], 1)
        self.assertEqual(len(piece.context_for_bin(1)), 2)
        self.assertEqual(len(piece.context_for_bin(2)), 1)
        self.assertLess(errors["max_offset_error_ms"], .01)
        with self.assertRaises(IndexError):
            piece.context_for_bin(3)

    def test_default_target_selection_other_pianos_are_conditioning(self):
        source = midi([(0, 60, 80, 0, .1), (1, 64, 70, 0, .1), (1, 67, 75, .1, .2)], programs={0: 0, 1: 0})
        piece, _, _ = self.roundtrip(source)
        self.assertEqual(piece.metadata["target_channel"], 1)
        self.assertEqual(piece.metadata["piano_notes"], 2)
        self.assertEqual(len(piece.events), 1)
        selected = self.tokenizer.encode(source, piano_channel=0)
        self.assertEqual(selected.metadata["piano_notes"], 1)
        with self.assertRaises(ValueError):
            self.tokenizer.encode(source, piano_channel=9)

    def test_no_piano_selects_real_other_instrument(self):
        piece, _, _ = self.roundtrip(midi([(2, 60, 80, 0, .2)], programs={2: 40}))
        self.assertTrue(piece.metadata["target_present"])
        self.assertTrue(piece.velocity.any())
        self.assertTrue(piece.onset.any())
        self.assertEqual(len(piece.events), 0)
        self.assertEqual(piece.metadata["target_channel"], 2)
        self.assertEqual(piece.metadata["target_program"], 40)
        self.assertEqual(piece.metadata["target_kind"], "pitched")
        self.assertEqual(piece.metadata["source_target_notes"], 1)
        self.assertEqual(piece.metadata["piano_notes"], 0)
        with self.assertRaisesRegex(ValueError, "non-piano"):
            self.tokenizer.encode(midi([(2, 60, 80, 0, .2)], programs={2: 40}), piano_channel=2)

    def test_empty_midi_still_has_empty_target(self):
        piece, _, _ = self.roundtrip(midi([]))
        self.assertFalse(piece.metadata["target_present"])
        self.assertEqual(piece.metadata["target_selection"], "empty")
        self.assertFalse(piece.velocity.any())
        self.assertFalse(piece.onset.any())

    def test_piano_priority_independent_of_seed_and_nonpiano_note_count(self):
        source = midi([(0, 60, 80, 0, .1)] + [(2, 70, 80, i * .1, (i + 1) * .1) for i in range(10)],
                      programs={0: 0, 2: 40})
        for seed in range(5):
            piece = self.tokenizer.encode(source, seed=seed)
            self.assertEqual(piece.metadata["target_channel"], 0)
            self.assertEqual(piece.metadata["target_selection"], "piano-priority")
            self.assertEqual(len(piece.events), 10)

    def test_seeded_fallback_can_choose_each_instrument_including_drums(self):
        source = midi([(2, 60, 80, 0, .1), (3, 72, 75, 0, .1), (9, 36, 90, 0, .1)],
                      programs={2: 40, 3: 73})
        choices = set()
        for seed in range(20):
            a, b = (self.tokenizer.encode(source, seed=seed) for _ in range(2))
            self.assertEqual(a.metadata["target_channel"], b.metadata["target_channel"])
            np.testing.assert_array_equal(a.velocity, b.velocity)
            self.assertEqual(a.metadata["target_selection"], "random-fallback")
            self.assertEqual(len(a.events), 2)
            self.assertTrue(compare_notes(read_performance(source)["notes"],
                            read_performance(self.tokenizer.decode(a, strict=True))["notes"])["pitch_velocity_instrument_counts_match"])
            choices.add(a.metadata["target_channel"])
        self.assertEqual(choices, {2, 3, 9})

    def test_explicit_nonpiano_override_zero_pedal_and_original_program(self):
        source = midi([(0, 60, 80, 0, .1), (2, 72, 92, .99, 1.1)], programs={0: 0, 2: 40}, messages=[
            (0, mido.Message("control_change", channel=2, control=64, value=95))])
        piece = self.tokenizer.encode(source, target_channel=2)
        self.assertEqual(piece.metadata["target_selection"], "explicit")
        self.assertFalse(piece.metadata["pedal_enabled"])
        self.assertFalse(piece.pedal.any())
        self.assertFalse(piece.coarse_pedal.any())
        self.assertTrue(any(m["message"].get("channel") == 2 and m["message"].get("control") == 64 for m in piece.metadata["messages"]))
        notes = read_performance(self.tokenizer.decode(piece, target_only=True, strict=True))["notes"]
        self.assertEqual([(n.channel, n.program, n.pitch) for n in notes], [(2, 40, 72)])
        self.assertAlmostEqual(notes[0].start, .99)
        self.assertAlmostEqual(notes[0].end, 1.1)
        piece.pedal[0, 0] = 10
        piece.refresh_coarse()
        with self.assertRaisesRegex(ValueError, "disabled"):
            piece.validate()

    def test_drum_roll_hits_retrigger_and_decode_on_percussion_channel(self):
        source = midi([(9, 36, 100, .99, 1), (9, 36, 100, 1, 1.01), (9, 38, 75, 1, 1.02)],
                      messages=[(0, mido.Message("control_change", channel=9, control=64, value=127))])
        piece = self.tokenizer.encode(source)
        self.assertTrue(piece.metadata["target_is_drum"])
        self.assertFalse(piece.metadata["target_is_piano"])
        self.assertFalse(piece.pedal.any())
        self.assertEqual(piece.onset[1, 36, 0], 1)
        self.assertEqual(piece.onset[1, 38, 0], 1)
        notes = read_performance(self.tokenizer.decode(piece, target_only=True, strict=True))["notes"]
        self.assertEqual([(n.channel, n.pitch, n.velocity) for n in notes], [(9, 36, 100), (9, 36, 100), (9, 38, 75)])
        self.assertEqual(piece.events.shape, (0, 11))

    def test_explicit_program_selects_one_part_on_program_changing_channel(self):
        source = midi([(2, 60, 80, 0, .1), (2, 72, 90, .2, .3)], programs={2: 40},
                      messages=[(300, mido.Message("program_change", channel=2, program=73))])
        piece = self.tokenizer.encode(source, target_channel=2, target_program=73)
        self.assertEqual(piece.metadata["target_program"], 73)
        self.assertEqual(piece.events[:, 2].tolist(), [40])
        self.assertEqual(piece.metadata["target_notes"], 1)
        self.assertTrue(compare_notes(read_performance(source)["notes"],
                        read_performance(self.tokenizer.decode(piece, strict=True))["notes"])["pitch_velocity_instrument_counts_match"])
        notes = read_performance(self.tokenizer.decode(piece, target_only=True, strict=True))["notes"]
        self.assertEqual([(n.pitch, n.program) for n in notes], [(72, 73)])

    def test_invalid_target_args_and_missing_part_are_rejected(self):
        source = midi([(0, 60, 80, 0, .1)])
        for args in (dict(target_channel=16), dict(target_channel=-1), dict(target_channel=True),
                     dict(target_channel=2), dict(target_program=0), dict(target_channel=0, target_program=128),
                     dict(target_channel=0, target_program=73), dict(target_channel=0, piano_channel=0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.tokenizer.encode(source, **args)

    def test_v2_without_new_selection_metadata_still_decodes(self):
        piece = self.tokenizer.encode(midi([(0, 60, 80, 0, .1)]))
        for key in ("target_is_piano", "target_is_drum", "target_kind", "target_selection",
                    "selection_seed", "pedal_enabled", "source_target_notes", "target_notes"):
            del piece.metadata[key]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "legacy_v2.npz"
            piece.save(path)
            loaded = EncodedPiece.load(path)
            self.assertEqual(len(read_performance(self.tokenizer.decode(loaded, strict=True))["notes"]), 1)

    def test_cli_drum_target_and_target_only_filename(self):
        with tempfile.TemporaryDirectory() as root:
            source, output = Path(root) / "source.mid", Path(root) / "output"
            midi([(0, 60, 80, 0, .1), (9, 36, 90, .1, .2)]).save(source)
            with redirect_stdout(io.StringIO()):
                report = roundtrip_main(["--input", str(source), "--output-dir", str(output), "--target-channel", "9", "--seed", "42"])
            self.assertEqual(report["target_kind"], "drums")
            self.assertTrue(report["normalized_reference_matches"])
            self.assertTrue((output / "target_only.mid").is_file())
            self.assertFalse((output / "piano_only.mid").exists())
            self.assertEqual([n.channel for n in read_performance(output / "target_only.mid")["notes"]], [9])

    def test_full_pitch_velocity_range(self):
        piece, _, _ = self.roundtrip(midi([(0, 0, 1, 0, .1), (0, 127, 127, 0, .1)]))
        self.assertEqual(piece.velocity[0, 0, 0], 1)
        self.assertEqual(piece.velocity[0, 127, 0], 127)
        self.assertEqual(piece.onset[0, 0, 0], 1)
        self.assertEqual(piece.onset[0, 127, 0], 1)

    def test_strict_missing_onset_and_deterministic_generated_fallback(self):
        piece = self.tokenizer.encode(midi([(0, 60, 80, 0, .1)]))
        piece.onset[:] = 0
        piece.refresh_coarse()
        with self.assertRaisesRegex(ValueError, "without onset"):
            self.tokenizer.decode(piece, strict=True)
        notes = read_performance(self.tokenizer.decode(piece))["notes"]
        self.assertEqual([n.velocity for n in notes], [80])
        self.assertTrue(all(n.end > n.start for n in notes))

    def test_velocity_change_requires_onset_in_strict_mode(self):
        piece = self.tokenizer.encode(midi([(0, 60, 80, 0, .1)]))
        piece.velocity[0, 60, 5:10] = 40
        piece.refresh_coarse()
        with self.assertRaisesRegex(ValueError, "Velocity changed without onset"):
            self.tokenizer.decode(piece, strict=True)
        self.assertEqual([n.velocity for n in read_performance(self.tokenizer.decode(piece))["notes"]], [80])
        piece.onset[0, 60, 5] = 1
        piece.refresh_coarse()
        notes = read_performance(self.tokenizer.decode(piece, strict=True))["notes"]
        self.assertEqual([n.velocity for n in notes], [80, 40])

    def test_zero_velocity_onset_rejected_by_strict_decoder(self):
        piece = self.tokenizer.encode(midi([(0, 60, 80, 0, .1)]))
        piece.onset[0, 62, 5] = 1
        piece.refresh_coarse()
        with self.assertRaisesRegex(ValueError, "Onset at zero velocity"):
            self.tokenizer.decode(piece, strict=True)
        self.assertEqual(len(read_performance(self.tokenizer.decode(piece))["notes"]), 1)

    def test_pedal_preserves_7bit_values_without_extending_key_duration(self):
        controls = [(100, mido.Message("control_change", control=64, value=127)),
                    (2100, mido.Message("control_change", control=64, value=63)),
                    (2500, mido.Message("control_change", control=64, value=0))]
        source = midi([(0, 60, 80, 0, .2)], messages=controls)
        piece, output, _ = self.roundtrip(source)
        self.assertEqual(piece.pedal[0, 5], 127)
        self.assertEqual(piece.pedal[1, 5], 63)
        self.assertEqual(piece.pedal[1, 25], 0)
        pedal_values = [m.value for t in output.tracks for m in t if m.type == "control_change" and m.control == 64]
        self.assertEqual(pedal_values, [127, 63, 0])
        self.assertAlmostEqual(read_performance(output)["notes"][0].end, .2)
        frozen = HierarchicalTokenizer(GridConfig(include_pedal=False)).encode(source)
        self.assertFalse(frozen.pedal.any())
        self.assertTrue(any(m["message"].get("control") == 64 for m in frozen.metadata["messages"]))

    def test_pedal_carries_bins_and_final_state_closes_without_extending_notes(self):
        source = midi([(0, 60, 80, .9, 1.1)], messages=[
            (1600, mido.Message("control_change", control=64, value=96)),
            (3200, mido.MetaMessage("marker", text="trailing silence"))])
        piece, output, _ = self.roundtrip(source)
        self.assertTrue(np.all(piece.pedal[1, :60] == 96))
        self.assertFalse(piece.pedal[1, 60:].any())
        self.assertEqual(piece.coarse_pedal.tolist(), [96, 96])
        values = [m.value for t in output.tracks for m in t if m.type == "control_change" and m.control == 64]
        self.assertEqual(values, [96, 0])
        self.assertAlmostEqual(read_performance(output)["notes"][0].end, 1.1)

    def test_pedal_same_frame_last_value_wins(self):
        source = midi([(0, 60, 80, 0, .1)], messages=[
            (2, mido.Message("control_change", control=64, value=127)),
            (8, mido.Message("control_change", control=64, value=23))])
        piece = self.tokenizer.encode(source)
        self.assertEqual(piece.pedal[0, 0], 23)
        self.assertEqual(piece.coarse_pedal[0], 23)

    def test_tempo_change_is_global_and_output_tick_resolution_is_high(self):
        source = mido.MidiFile(type=1, ticks_per_beat=120)
        source.tracks.append(track([(0, mido.MetaMessage("set_tempo", tempo=500_000)),
                                   (240, mido.MetaMessage("set_tempo", tempo=1_000_000)),
                                   (0, mido.MetaMessage("time_signature", numerator=3, denominator=4))]))
        source.tracks.append(track([(0, mido.Message("program_change", program=0)),
                                   (119, mido.Message("note_on", note=60, velocity=91)),
                                   (360, mido.Message("note_on", note=60, velocity=0)),
                                   (180, mido.Message("pitchwheel", pitch=1234))]))
        piece, output, errors = self.roundtrip(source)
        self.assertEqual(piece.metadata["tempo_map"], [(0, 500_000), (240, 1_000_000)])
        self.assertGreaterEqual(output.ticks_per_beat, 9600)
        self.assertLessEqual(errors["max_onset_error_ms"], 5.1)
        self.assertLessEqual(errors["max_offset_error_ms"], 5.1)
        self.assertTrue(any(m.type == "pitchwheel" and m.pitch == 1234 for t in output.tracks for m in t))
        self.assertEqual([m.tempo for t in output.tracks for m in t if m.type == "set_tempo"], [500_000, 1_000_000])
        tempo = TempoMap(120, [(0, 500_000), (240, 1_000_000)])
        self.assertEqual(tempo.to_tick(2.), 360)
        self.assertEqual(tempo.to_seconds(360), 2.)

    def test_npz_roundtrip_and_canvas_edits_not_source_notes_drive_decode(self):
        piece = self.tokenizer.encode(midi([(0, 60, 80, 0, .2), (2, 72, 95, 0, .2)], programs={0: 0, 2: 40}))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "piece.npz"
            piece.save(path)
            with self.assertRaises(FileExistsError):
                piece.save(path)
            reloaded = EncodedPiece.load(path)
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(set(data.files), {"velocity", "onset", "pedal", "valid_mask", "coarse_velocity",
                                                   "coarse_onsets", "coarse_pedal", "events", "event_ptr", "metadata"})
                header = json.loads(str(data["metadata"]))
                self.assertEqual(header["format"], FORMAT_VERSION)
                self.assertNotIn("lanes", header["config"])
            np.testing.assert_array_equal(reloaded.velocity, piece.velocity)
            np.testing.assert_array_equal(reloaded.onset, piece.onset)
            for array in (reloaded.velocity, reloaded.onset):
                array[:, 62] = array[:, 60]
                array[:, 60] = 0
            reloaded.refresh_coarse()
            notes = read_performance(self.tokenizer.decode(reloaded, strict=True))["notes"]
            self.assertEqual({n.pitch for n in notes}, {62, 72})
            self.assertFalse(any(m["message"]["type"].startswith("note_") for m in piece.metadata["messages"]))
            piano_only = read_performance(self.tokenizer.decode(reloaded, piano_only=True, strict=True))["notes"]
            self.assertEqual([n.pitch for n in piano_only], [62])

    def test_silence_is_valid_padding_is_not_and_bad_values_are_rejected(self):
        source = midi([(0, 60, 80, .5, .6)])
        piece = self.tokenizer.encode(source)
        self.assertTrue(piece.valid_mask[0, :60].all())
        self.assertFalse(piece.velocity[..., :50].any())
        for field in ("velocity", "onset", "pedal"):
            broken = deepcopy(piece)
            array = getattr(broken, field)
            array[(0, 60, 80) if array.ndim == 3 else (0, 80)] = 1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "padd"):
                broken.validate()
        broken = deepcopy(piece)
        broken.valid_mask[0, 2] = False
        with self.assertRaises(ValueError):
            broken.validate()
        broken = deepcopy(piece)
        broken.velocity[0, 60, 55] = 128
        with self.assertRaises(ValueError):
            broken.validate()
        broken = deepcopy(piece)
        broken.onset[0, 60, 55] = 2
        with self.assertRaises(ValueError):
            broken.validate()

    def test_parameterized_bins_and_duration_safety(self):
        config = GridConfig(coarse_ms=500, fine_ms=5)
        piece = HierarchicalTokenizer(config).encode(midi([(0, 60, 80, .49, .52)]))
        self.assertEqual(piece.velocity.shape, (2, 128, 100))
        self.assertEqual(piece.onset[0, 60, 98], 1)
        self.assertFalse(piece.onset[1, 60].any())
        self.assertAlmostEqual(read_performance(HierarchicalTokenizer(config).decode(piece, strict=True))["notes"][0].end, .52)
        for kwargs in (dict(coarse_ms=999), dict(fine_ms=0), dict(coarse_ms=True), dict(min_output_ppq=40000)):
            with self.assertRaises(ValueError):
                GridConfig(**kwargs)
        with self.assertRaisesRegex(ValueError, "duration"):
            HierarchicalTokenizer(GridConfig(max_frames=10)).encode(midi([(0, 60, 80, 0, .2)]))

    def test_coarse_summaries_exactly_match_fine_arrays_and_stale_edits_fail(self):
        source = midi([(0, 60, 70, .9, 1.1), (0, 60, 30, 1.1, 1.2), (0, 64, 100, 0, .1)],
                      messages=[(1000, mido.Message("control_change", control=64, value=41))])
        piece = self.tokenizer.encode(source)
        np.testing.assert_array_equal(piece.coarse_velocity, piece.velocity.max(axis=2))
        np.testing.assert_array_equal(piece.coarse_onsets, piece.onset.sum(axis=2, dtype=np.int32))
        np.testing.assert_array_equal(piece.coarse_pedal, piece.pedal.max(axis=1))
        self.assertEqual(piece.coarse_onsets[:, 60].tolist(), [1, 1])
        self.assertEqual(piece.coarse_velocity[:, 60].tolist(), [70, 70])
        self.assertEqual(piece.coarse_pedal.tolist(), [41, 41])
        for field in ("coarse_velocity", "coarse_onsets", "coarse_pedal"):
            broken = deepcopy(piece)
            getattr(broken, field).flat[0] += 1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Coarse"):
                broken.validate()
            broken.refresh_coarse().validate()

    def test_image_broadcasts_pedal_only_for_display(self):
        piece = self.tokenizer.encode(midi([(0, 60, 80, 0, .1)], messages=[
            (0, mido.Message("control_change", control=64, value=71))]))
        view = piece.image()
        self.assertEqual(view.shape, (1, 3, 128, 100))
        np.testing.assert_array_equal(view[:, 0], piece.velocity)
        np.testing.assert_array_equal(view[:, 1], piece.onset)
        np.testing.assert_array_equal(view[:, 2, 0], piece.pedal)
        self.assertEqual(piece.pedal.shape, (1, 100))
        view[:, 0] = 0
        self.assertTrue(piece.velocity.any())  # Display edits do not mutate storage.

    def test_accompaniment_overlap_is_not_normalized_and_half_open_carry(self):
        source = midi([(0, 60, 80, 0, .1), (2, 72, 80, .5, 1),
                       (2, 72, 45, .7, 1.5), (9, 36, 90, 1, 1.1)], programs={0: 0, 2: 40})
        piece, output, _ = self.roundtrip(source)
        self.assertEqual(len(piece.events), 3)
        # A note ending at 1s is not carried into bin 1; overlapping later note is.
        self.assertEqual(piece.context_for_bin(1)[:, 1].tolist(), [2, 9])
        self.assertEqual(len(read_performance(output)["notes"]), 4)
        self.assertEqual(piece.metadata["normalization"]["overlap_normalizations"], 0)

    def test_multi_track_piano_selection_and_context_track_identity(self):
        source = mido.MidiFile(type=1, ticks_per_beat=1000)
        source.tracks = [track([(0, mido.Message("program_change", program=0)),
                               (0, mido.Message("note_on", note=60, velocity=80)),
                               (400, mido.Message("note_off", note=60))]),
                         track([(0, mido.Message("note_on", note=64, velocity=90)),
                                (400, mido.Message("note_off", note=64))]),
                         track([(0, mido.Message("program_change", channel=2, program=40)),
                                (1900, mido.Message("note_on", channel=2, note=72, velocity=70)),
                                (2100, mido.Message("note_off", channel=2, note=72))])]
        piece, output, _ = self.roundtrip(source)
        self.assertEqual(piece.metadata["piano_notes"], 2)
        self.assertEqual(piece.events[0, :3].tolist(), [2, 2, 40])
        self.assertEqual(piece.context_for_bin(1)[0, 0], 2)
        self.assertEqual([n.track for n in read_performance(output)["notes"] if n.channel == 2], [2])

    def test_v1_npz_rejected_explicitly_without_modifying_file(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "old.npz"
            with path.open("xb") as handle:
                np.savez_compressed(handle, metadata=np.array(json.dumps(dict(
                    format="hierarchical_piano_events_v1", config=asdict(GridConfig()), metadata={}))))
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "re-encode"):
                EncodedPiece.load(path)
            self.assertEqual(path.read_bytes(), before)

    def test_roundtrip_cli_reports_normalization_without_false_preservation_claim(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source.mid"
            output = Path(root) / "result"
            midi([(0, 60, 80, 0, .3), (0, 60, 45, .1, .4),
                  (0, 72, 80, .001, .003), (0, 72, 35, .004, .006)]).save(source)
            with redirect_stdout(io.StringIO()):
                report = roundtrip_main(["--input", str(source), "--output-dir", str(output)])
            self.assertTrue(report["normalized_reference_matches"])
            self.assertFalse(report["comparison_to_source"]["pitch_velocity_instrument_counts_match"])
            self.assertIsNone(report["comparison_to_source"]["max_onset_error_ms"])
            self.assertIsNone(report["comparison_to_source"]["max_offset_error_ms"])
            self.assertEqual(report["normalization"]["overlap_normalizations"], 1)
            self.assertEqual(report["normalization"]["same_frame_collisions"], 1)
            self.assertEqual(json.loads((output / "summary.json").read_text()), report)
            self.assertEqual(len(read_performance(output / "piano_only.mid")["notes"]), 3)

    def test_random_valid_frame_runs_roundtrip_and_no_bin_splits(self):
        # Independent randomized reference with known frame-aligned notes;
        # exercises chords, gaps, reattacks and many boundary crossings.
        rng = np.random.default_rng(128)
        for trial in range(25):
            notes, velocities, onsets = [], np.zeros((128, 400), np.uint8), np.zeros((128, 400), np.uint8)
            for pitch in range(58, 66):
                start = int(rng.integers(0, 20))
                for _ in range(5):
                    end = start + int(rng.integers(1, 60))
                    velocity = int(rng.integers(1, 128))
                    notes.append((0, pitch, velocity, start / 100, end / 100))
                    velocities[pitch, start:end] = velocity
                    onsets[pitch, start] = 1
                    start = end + int(rng.integers(0, 10))
            with self.subTest(trial=trial):
                piece, _, _ = self.roundtrip(midi(notes))
                frames = piece.velocity.shape[0] * 100
                np.testing.assert_array_equal(piece.velocity.transpose(1, 0, 2).reshape(128, -1), velocities[:, :frames])
                np.testing.assert_array_equal(piece.onset.transpose(1, 0, 2).reshape(128, -1), onsets[:, :frames])

    def test_malformed_source_rejected_not_silently_repaired(self):
        source = mido.MidiFile()
        source.tracks.append(track([(0, mido.Message("note_on", note=60, velocity=80))]))
        with self.assertRaisesRegex(ValueError, "dangling"):
            self.tokenizer.encode(source)
        source.type = 2
        with self.assertRaisesRegex(ValueError, "types 0/1"):
            self.tokenizer.encode(source)

    def test_training_recovery_keeps_paired_notes_drops_unresolved_tail(self):
        source = midi([(0, 60, 80, 0, .2)], messages=[
            (1000, mido.Message("note_on", note=100, velocity=90)),
            (2000, mido.MetaMessage("marker", text="end"))])
        with self.assertRaisesRegex(ValueError, "dangling"):
            read_performance(source)
        parsed = read_performance(source, recover=True)
        self.assertEqual([(n.pitch, n.velocity, n.end) for n in parsed["notes"]], [(60, 80, .2)])
        self.assertEqual(parsed["end_seconds"], 1)
        piece = self.tokenizer.encode(source, recover=True)
        self.assertFalse(piece.velocity[:, 100].any())
        self.assertTrue(piece.velocity[:, 60].any())

    def test_training_recovery_caps_unreleased_retriggers_not_at_file_end(self):
        source = midi([], messages=[
            (0, mido.Message("note_on", note=60, velocity=80)),
            (400, mido.Message("note_on", note=60, velocity=90)),
            (2000, mido.MetaMessage("marker", text="end"))])
        notes = read_performance(source, recover=True)["notes"]
        self.assertEqual([(n.start, n.end, n.velocity) for n in notes], [(0, .2, 80)])

    def test_training_recovery_unchanged_for_valid_overlapping_source(self):
        source = midi([(0, 60, 80, 0, .3), (0, 60, 90, .1, .4), (2, 65, 70, .2, .5)],
                      programs={0: 0, 2: 40})
        self.assertEqual(read_performance(source)["notes"], read_performance(source, recover=True)["notes"])
        strict, recovered = self.tokenizer.encode(source), self.tokenizer.encode(source, recover=True)
        np.testing.assert_array_equal(strict.velocity, recovered.velocity)
        np.testing.assert_array_equal(strict.events, recovered.events)

    def test_unreleased_drums_use_one_frame_not_whole_track(self):
        source = midi([], messages=[
            (0, mido.Message("note_on", channel=9, note=36, velocity=90)),
            (10, mido.Message("note_on", channel=9, note=36, velocity=100)),
            (4000, mido.MetaMessage("marker", text="end"))])
        notes = read_performance(source, recover=True)["notes"]
        self.assertEqual([(n.start, n.end) for n in notes], [(0, .005), (.005, .015)])
        piece = HierarchicalTokenizer(GridConfig(fine_ms=20)).encode(source, recover=True)
        self.assertLessEqual(np.count_nonzero(piece.velocity), 2)

    def test_bad_sysex_recovery_preserves_notes_controllers_tempo_and_source(self):
        source = midi([(0, 60, 80, .1, .3)], messages=[
            (0, mido.MetaMessage("set_tempo", tempo=500000)),
            (100, mido.Message("sysex", data=(0x43, 1, 2))),
            (200, mido.Message("control_change", control=64, value=73)),
            (300, mido.Message("sysex", data=(0x43, 3, 4)))])
        buffer = io.BytesIO()
        source.save(file=buffer)
        raw = buffer.getvalue().replace(b"\xf0\x04\x43\x01\x02\xf7", b"\xf0\x04\x43\x80\x02\xf7")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_sysex.mid"
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                self.tokenizer.encode(path)
            parsed = read_performance(path, recover=True)
            expected = read_performance(source)
            # Removing non-musical events shifts internal source-order indices,
            # but must leave every musical field unchanged.
            self.assertEqual([replace(n, order=0) for n in parsed["notes"]],
                             [replace(n, order=0) for n in expected["notes"]])
            self.assertEqual(parsed["tempo_map"], expected["tempo_map"])
            sysex = [m for m in parsed["messages"] if m["message"]["type"] == "sysex"]
            self.assertEqual(len(sysex), 1)
            self.assertEqual(sysex[0]["tick"], 300)
            pedal = [m for m in parsed["messages"] if m["message"]["type"] == "control_change"]
            self.assertEqual((pedal[0]["tick"], pedal[0]["message"]["value"]), (200, 73))
            original, recovered = self.tokenizer.encode(source), self.tokenizer.encode(path, recover=True)
            np.testing.assert_array_equal(original.velocity, recovered.velocity)
            np.testing.assert_array_equal(original.pedal, recovered.pedal)
            self.assertEqual(path.read_bytes(), raw)

    def test_recovery_never_clips_invalid_note_data(self):
        source = midi([(0, 60, 80, .1, .3)], messages=[(0, mido.Message("sysex", data=(0x43, 1, 2)))])
        buffer = io.BytesIO()
        source.save(file=buffer)
        raw = buffer.getvalue().replace(b"\xf0\x04\x43\x01\x02\xf7", b"\xf0\x04\x43\x80\x02\xf7")
        raw = raw.replace(b"\x90\x3c\x50", b"\x90\x3c\xff")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid_note.mid"
            path.write_bytes(raw)
            with self.assertRaises(OSError):
                read_performance(path, recover=True)

    def test_sysex_with_omitted_following_channel_status(self):
        source = midi([(0, 60, 80, .2, .3)], messages=[
            (50, mido.Message("control_change", control=64, value=0)),
            (100, mido.Message("sysex", data=(0x43, 1, 2))),
            (200, mido.Message("control_change", control=64, value=73))])
        buffer = io.BytesIO()
        source.save(file=buffer)
        raw = buffer.getvalue()
        # XG-style omission: CC after SysEx reuses the earlier B0 status.
        original_size = len(raw)
        raw = raw.replace(b"\x64\xb0\x40\x49", b"\x64\x40\x49", 1)
        self.assertEqual(len(raw), original_size - 1)
        raw = raw[:18] + (len(raw) - 22).to_bytes(4, "big") + raw[22:]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "xg_running_status.mid"
            for bad_payload in (False, True):
                with self.subTest(bad_payload=bad_payload):
                    data = raw.replace(b"\xf0\x04\x43\x01\x02\xf7", b"\xf0\x04\x43\x80\x02\xf7") if bad_payload else raw
                    path.write_bytes(data)
                    expected, recovered = self.tokenizer.encode(source), self.tokenizer.encode(path, recover=True)
                    np.testing.assert_array_equal(expected.velocity, recovered.velocity)
                    np.testing.assert_array_equal(expected.pedal, recovered.pedal)

    def test_trailing_removed_sysex_preserves_track_duration(self):
        # A source without an end marker, ending in a bad SysEx after 100 ticks.
        payload = b"\x00\x90\x3c\x50\x64\x80\x3c\x00\x64\xf0\x02\x80\xf7"
        raw = b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x03\xe8MTrk" + len(payload).to_bytes(4, "big") + payload
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trailing_sysex.mid"
            path.write_bytes(raw)
            parsed = read_performance(path, recover=True)
            self.assertAlmostEqual(parsed["end_seconds"], .1)
            self.assertAlmostEqual(parsed["notes"][0].end, .05)

    def test_rmid_container_unwrapped_only_in_recovery_mode(self):
        source = midi([(0, 60, 85, .1, .3)], messages=[(100, mido.Message("control_change", control=64, value=73))])
        buffer = io.BytesIO()
        source.save(file=buffer)
        smf = buffer.getvalue()
        def chunk(kind, payload):
            return kind + len(payload).to_bytes(4, "little") + payload + (b"\x00" if len(payload) % 2 else b"")
        body = b"RMID" + chunk(b"JUNK", b"abc") + chunk(b"data", smf)
        raw = b"RIFF" + len(body).to_bytes(4, "little") + body
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wrapped.mid"
            path.write_bytes(raw)
            with self.assertRaises(OSError):
                self.tokenizer.encode(path)
            expected, recovered = self.tokenizer.encode(source), self.tokenizer.encode(path, recover=True)
            np.testing.assert_array_equal(expected.velocity, recovered.velocity)
            np.testing.assert_array_equal(expected.pedal, recovered.pedal)
            self.assertEqual(path.read_bytes(), raw)
            path.write_bytes(raw[:-1])
            with self.assertRaises(EOFError):
                self.tokenizer.encode(path, recover=True)

    def test_bad_key_signature_metadata_removed_without_guessing_key(self):
        for key_data in ((254, 255), (8, 0), (0,), ()):
            with self.subTest(data=key_data), tempfile.TemporaryDirectory() as tmp:
                source = midi([(0, 60, 85, .1, .3)], messages=[
                    (100, mido.UnknownMetaMessage(type_byte=0x59, data=key_data)),
                    (200, mido.MetaMessage("key_signature", key="Dm")),
                    (200, mido.Message("control_change", control=64, value=73))])
                path = Path(tmp) / "bad_key.mid"
                source.save(path)
                with self.assertRaises((mido.KeySignatureError, IndexError)):
                    read_performance(path)
                parsed = read_performance(path, recover=True)
                self.assertEqual([(n.pitch, n.start, n.end, n.velocity) for n in parsed["notes"]], [(60, .1, .3, 85)])
                keys = [m for m in parsed["messages"] if m["message"]["type"] == "key_signature"]
                self.assertEqual([(m["tick"], m["message"]["key"]) for m in keys], [(200, "Dm")])
                self.assertEqual(next(m["tick"] for m in parsed["messages"] if m["message"]["type"] == "control_change"), 200)

    def test_invalid_release_velocity_normalized_including_running_status(self):
        source = midi([(0, 60, 80, 0, .1), (0, 64, 90, 0, .1)])
        buffer = io.BytesIO()
        source.save(file=buffer)
        raw = buffer.getvalue()
        # Mido serializes the second note_off using running status.
        self.assertIn(b"\x80\x3c\x40\x00\x40\x40", raw)
        raw = raw.replace(b"\x80\x3c\x40\x00\x40\x40", b"\x80\x3c\xf0\x00\x40\xff")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "release_velocity.mid"
            path.write_bytes(raw)
            with self.assertRaises(OSError):
                read_performance(path)
            expected, recovered = self.tokenizer.encode(source), self.tokenizer.encode(path, recover=True)
            np.testing.assert_array_equal(expected.velocity, recovered.velocity)
            np.testing.assert_array_equal(expected.onset, recovered.onset)
            self.assertEqual(path.read_bytes(), raw)

    def test_invalid_bank_select_removed_without_changing_pedal_or_program(self):
        for control in (0, 32):
            with self.subTest(control=control), tempfile.TemporaryDirectory() as tmp:
                source = midi([(0, 60, 85, .1, .3)], messages=[
                    (100, mido.Message("control_change", control=control, value=1)),
                    (200, mido.Message("control_change", control=64, value=73)),
                    (250, mido.MetaMessage("marker", text="retain"))])
                buffer = io.BytesIO()
                source.save(file=buffer)
                raw = buffer.getvalue().replace(bytes((0xb0, control, 1)), bytes((0xb0, control, 255)))
                path = Path(tmp) / "bank_select.mid"
                path.write_bytes(raw)
                with self.assertRaises(OSError):
                    read_performance(path)
                parsed = read_performance(path, recover=True)
                self.assertFalse(any(m["message"].get("control") == control for m in parsed["messages"]))
                self.assertEqual(next(m["tick"] for m in parsed["messages"] if m["message"].get("control") == 64), 200)
                expected, recovered = self.tokenizer.encode(source), self.tokenizer.encode(path, recover=True)
                np.testing.assert_array_equal(expected.velocity, recovered.velocity)
                np.testing.assert_array_equal(expected.pedal, recovered.pedal)
                self.assertEqual(expected.metadata["target_program"], recovered.metadata["target_program"])

    def test_recovery_does_not_change_invalid_pitch_or_other_controller(self):
        source = midi([(0, 60, 80, .1, .3)], messages=[(0, mido.Message("control_change", control=64, value=73))])
        buffer = io.BytesIO()
        source.save(file=buffer)
        for original, invalid in ((b"\x90\x3c\x50", b"\x90\xfb\x50"),
                                  (b"\xb0\x40\x49", b"\xb0\x40\xff"),
                                  (b"\x80\x3c\x40", b"\x80\xfb\xff")):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "unrecoverable.mid"
                path.write_bytes(buffer.getvalue().replace(original, invalid))
                with self.assertRaises(OSError):
                    read_performance(path, recover=True)

    def test_bad_smpte_offset_removed_preserving_valid_offset_and_timing(self):
        for payload in ((128, 0, 0, 0, 0), (0,), ()):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as tmp:
                source = midi([(0, 60, 85, .1, .3)], messages=[
                    (100, mido.UnknownMetaMessage(type_byte=0x54, data=payload)),
                    (200, mido.MetaMessage("smpte_offset", frame_rate=25, hours=1)),
                    (250, mido.Message("control_change", control=64, value=73))])
                path = Path(tmp) / "offset.mid"
                source.save(path)
                raw = path.read_bytes()
                with self.assertRaises((KeyError, IndexError)):
                    read_performance(path)
                parsed = read_performance(path, recover=True)
                self.assertEqual([(n.pitch, n.start, n.end, n.velocity) for n in parsed["notes"]],
                                 [(60, .1, .3, 85)])
                offsets = [m for m in parsed["messages"] if m["message"]["type"] == "smpte_offset"]
                self.assertEqual([(m["tick"], m["message"]["hours"]) for m in offsets], [(200, 1)])
                self.assertEqual(next(m["tick"] for m in parsed["messages"] if m["message"].get("control") == 64), 250)
                self.tokenizer.encode(path, recover=True).validate()
                self.assertEqual(path.read_bytes(), raw)

    def test_bad_pitchwheel_removed_including_running_status(self):
        source = midi([(0, 60, 85, .2, .4)], messages=[
            (100, mido.Message("pitchwheel", pitch=0)),
            (150, mido.Message("pitchwheel", pitch=0)),
            (200, mido.Message("pitchwheel", pitch=1024)),
            (250, mido.Message("control_change", control=64, value=73))])
        buffer = io.BytesIO()
        source.save(file=buffer)
        raw = buffer.getvalue()
        self.assertIn(b"\xe0\x00\x40\x32\x00\x40", raw)
        # Invalid low byte with explicit status; invalid high byte with running
        # status. A high-bit first byte WITHOUT status would be ambiguous.
        raw = raw.replace(b"\xe0\x00\x40\x32\x00\x40", b"\xe0\xff\x40\x32\x00\x80")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bend.mid"
            path.write_bytes(raw)
            with self.assertRaises(OSError):
                read_performance(path)
            parsed = read_performance(path, recover=True)
            bends = [m for m in parsed["messages"] if m["message"]["type"] == "pitchwheel"]
            self.assertEqual([(m["tick"], m["message"]["pitch"]) for m in bends], [(200, 1024)])
            expected, recovered = self.tokenizer.encode(source), self.tokenizer.encode(path, recover=True)
            np.testing.assert_array_equal(expected.velocity, recovered.velocity)
            np.testing.assert_array_equal(expected.onset, recovered.onset)
            np.testing.assert_array_equal(expected.pedal, recovered.pedal)
            self.assertEqual(path.read_bytes(), raw)

    def test_bad_volume_removed_preserving_valid_volume_and_running_pedal(self):
        source = midi([(0, 60, 85, .1, .3)], messages=[
            (100, mido.Message("control_change", control=7, value=1)),
            (150, mido.Message("control_change", control=7, value=1)),
            (200, mido.Message("control_change", control=7, value=100)),
            (250, mido.Message("control_change", control=64, value=73))])
        buffer = io.BytesIO()
        source.save(file=buffer)
        raw = buffer.getvalue()
        self.assertIn(b"\xb0\x07\x01\x32\x07\x01", raw)
        raw = raw.replace(b"\xb0\x07\x01\x32\x07\x01", b"\xb0\x07\x95\x32\x07\xff")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "volume.mid"
            path.write_bytes(raw)
            with self.assertRaises(OSError):
                read_performance(path)
            parsed = read_performance(path, recover=True)
            volumes = [m for m in parsed["messages"] if m["message"].get("control") == 7]
            self.assertEqual([(m["tick"], m["message"]["value"]) for m in volumes], [(200, 100)])
            self.assertEqual(next(m["tick"] for m in parsed["messages"] if m["message"].get("control") == 64), 250)
            expected, recovered = self.tokenizer.encode(source), self.tokenizer.encode(path, recover=True)
            np.testing.assert_array_equal(expected.velocity, recovered.velocity)
            np.testing.assert_array_equal(expected.pedal, recovered.pedal)
            self.assertEqual(path.read_bytes(), raw)

    def test_new_recoveries_do_not_salvage_truncated_events(self):
        # Recoverable invalid offset followed by an incomplete pitchwheel.
        payload = b"\x00\xff\x54\x05\x80\x00\x00\x00\x00\x00\xe0\x7f"
        raw = b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x03\xe8MTrk" + len(payload).to_bytes(4, "big") + payload
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "truncated.mid"
            path.write_bytes(raw)
            with self.assertRaisesRegex(EOFError, "Truncated MIDI event"):
                read_performance(path, recover=True)

    def test_zero_duration_notes_skipped_counted_without_disrupting_next_note(self):
        source = midi([(0, 60, 99, .1, .1), (0, 60, 75, .2, .3),
                       (2, 72, 55, .4, .4)], programs={0: 0, 2: 40})
        parsed = read_performance(source)
        self.assertEqual(parsed["skipped_nonpositive_notes"], 2)
        self.assertEqual(parsed["unmatched_offs"], 0)
        piece, output, _ = self.roundtrip(source)
        self.assertEqual(piece.metadata["skipped_nonpositive_notes"], 2)
        self.assertEqual(piece.metadata["source_notes"], 1)
        self.assertEqual(len(piece.events), 0)
        self.assertEqual([n.velocity for n in read_performance(output)["notes"]], [75])

    def test_zero_duration_velocity_zero_off_and_all_notes_off(self):
        for closing in (mido.Message("note_on", note=60, velocity=0),
                        mido.Message("control_change", control=123, value=0)):
            source = mido.MidiFile()
            source.tracks.append(track([(0, mido.Message("note_on", note=60, velocity=80)), (0, closing)]))
            piece = self.tokenizer.encode(source)
            with self.subTest(closing=closing):
                self.assertEqual(piece.metadata["skipped_nonpositive_notes"], 1)
                self.assertFalse(piece.velocity.any())
                self.assertEqual(read_performance(self.tokenizer.decode(piece, strict=True))["notes"], [])

    def test_negative_delta_time_rejected_even_if_absolute_tick_stays_positive(self):
        source = mido.MidiFile()
        source.tracks.append(mido.MidiTrack([
            mido.Message("note_on", note=60, velocity=80, time=100),
            mido.Message("note_off", note=60, time=-1)]))
        with self.assertRaisesRegex(ValueError, "Negative MIDI delta"):
            self.tokenizer.encode(source)


if __name__ == "__main__":
    unittest.main()
