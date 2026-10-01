"""CPU-only syntax rewards; no model, checkpoint, or training job required."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from improvnet.metrics import MidiSyntaxMetric, summarize_syntax
from improvnet.tokenizer.absolute import AbsTokenizer
from improvnet.tokenizer.midi import MidiDict


def event(instrument="Acoustic Piano", pitch=60, velocity=80, onset=0, duration=100):
    return [("instrument", instrument), ("note", pitch, velocity),
            ("onset", onset), ("dur", duration)]


class MidiSyntaxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AbsTokenizer()
        cls.metric = MidiSyntaxMetric(cls.tokenizer)

    def assert_valid(self, tokens, **kwargs):
        result = self.metric.evaluate_sequence(tokens, **kwargs)
        self.assertEqual(result.score, 1, result.issue)
        self.assertTrue(result.syntax_valid)
        self.assertTrue(result.midi_compiles)
        return result

    def assert_invalid(self, tokens, code, **kwargs):
        result = self.metric.evaluate_sequence(tokens, **kwargs)
        self.assertEqual(result.validity_score, 0)
        self.assertGreaterEqual(result.score, 0)
        self.assertLess(result.score, 1)
        self.assertEqual(result.issue.code, code, result)
        return result

    def test_full_sequence_drums_chords_and_markers(self):
        tokens = ["<S>"] + event() + event("drum", 36) + ["<T>", "<D>"] + event(onset=200) + ["<E>"]
        result = self.assert_valid(tokens)
        self.assertEqual((result.note_events, result.sounding_note_events), (3, 3))
        self.assertEqual(result.num_tokens, len(tokens))

    def test_ids_and_serialized_token_lists(self):
        tokens = ["<S>"] + event() + ["<E>"]
        ids = [self.tokenizer.tok_to_id[t] for t in tokens]
        self.assert_valid(ids)
        self.assert_valid([list(t) if isinstance(t, tuple) else t for t in tokens])
        self.assert_invalid([self.tokenizer.vocab_size], "unknown_token_id")
        self.assert_invalid([True], "invalid_token")

    def test_consecutive_time_tokens_are_a_zero_reward_quality_failure(self):
        for count in (2, 3, 255):
            tokens = ["<S>"] + ["<T>"] * count + event() + ["<E>"]
            result = self.assert_invalid(tokens, "consecutive_time_tokens")
            self.assertEqual(result.score, 0.)
            self.assertTrue(result.syntax_valid)  # Long rests are legal grammar.
            self.assertTrue(result.midi_compiles)
        self.assert_valid(["<S>", "<T>"] + event() + ["<T>"] + event() + ["<E>"])
        near_all_time = ["<S>"] + ["<T>"] * 255 + [("note", 60, 80)] + ["<E>"]
        result = self.metric.evaluate_sequence(near_all_time)
        self.assertEqual(result.score, 0.)
        self.assertTrue(any(issue.code == "consecutive_time_tokens" for issue in result.issues))

    def test_time_run_guard_checks_generated_boundaries_not_clean_history(self):
        history = ["<S>", "<T>", "<T>"] + event()
        self.assertEqual(self.metric.evaluate_canvas(event(), history=history).score, 1.)
        result = self.metric.evaluate_canvas(["<T>"] + event(), history=["<S>"] + event() + ["<T>"])
        self.assertTrue(result.syntax_valid)
        self.assertFalse(result.valid)
        self.assertEqual(result.score, 0.)
        self.assertEqual(result.issue.code, "consecutive_time_tokens")

    def test_torch_tensors_when_available(self):
        # All grammar/MIDI tests remain usable without a model-training runtime.
        try:
            import torch
        except ImportError:
            self.skipTest("A compatible PyTorch installation is not available")
        ids = torch.tensor([self.tokenizer.tok_to_id[t] for t in ["<S>"] + event() + ["<E>"]])
        self.assert_valid(ids)
        results = self.metric.evaluate_batch(ids.repeat(2, 1))
        self.assertEqual([result.score for result in results], [1, 1])
        with self.assertRaises(ValueError):
            self.metric.evaluate_sequence(torch.zeros(2, 3, dtype=torch.long))

    def test_raw_fields_must_be_integer_and_in_vocabulary(self):
        for field in (("note", 128, 80), ("note", 60, 81), ("onset", 5000),
                      ("dur", -10), ("instrument", "nonexistent"), ("drum", 36)):
            with self.subTest(field=field):
                self.assert_invalid([field], "out_of_vocabulary")
        for field in (("note", 60., 80), ("onset", 10.), ("dur", True)):
            with self.subTest(field=field):
                self.assert_invalid([field], "invalid_token")
        self.assert_invalid([("instrument", [])], "out_of_vocabulary")

    def test_wrappers_and_metadata(self):
        prefix = self.tokenizer.prefix_tokens[0]
        self.assert_valid([prefix, "<S>"] + event() + ["<E>"])
        self.assert_invalid(event() + ["<E>"], "missing_start")
        self.assert_invalid(["<S>"] + event(), "missing_end")
        self.assert_invalid(["<S>", "<S>"] + event() + ["<E>"], "start_position")
        self.assert_invalid(["<S>", prefix] + event() + ["<E>"], "prefix_position")
        self.assert_invalid(["<S>"] + event() + ["<E>", "<T>"], "tokens_after_end")
        self.assert_valid(event(), require_bos=False, require_eos=False)

    def test_noise_tokens_are_not_implicitly_removed(self):
        for token in ("<P>", "<MASK>", "<SEP>", "<BLANK>", "<U>"):
            with self.subTest(token=token):
                result = self.assert_invalid(["<S>"] + event() + [token, "<E>"], "unresolved_token")
                self.assertIsNone(result.midi_compiles)

    def test_event_field_order_and_incomplete_tails(self):
        for tokens in (["<S>"] + event()[1:] + ["<E>"],
                       ["<S>"] + event()[:2] + ["<D>"] + event()[2:] + ["<E>"],
                       ["<S>"] + [event()[0], event()[2], event()[1], event()[3], "<E>"],
                       ["<S>"] + event()[:3] + ["<E>"]):
            with self.subTest(tokens=tokens):
                self.assert_invalid(tokens, "event_order")
        self.assert_invalid(["<S>"] + event()[:3], "incomplete_event")
        self.assert_invalid(["<S>", "<D>", "<D>"] + event() + ["<E>"], "duplicate_dim")

    def test_permissive_decoder_is_not_a_validity_check(self):
        malformed = ["<S>"] + event() + ["<MASK>"] + event()[:2] + ["<E>"]
        # Existing decoder accepts this and discards the invalid tail/noise.
        self.tokenizer.detokenize(malformed).to_midi()
        self.assert_invalid(malformed, "unresolved_token")

    def test_empty_or_nonsounding_output_cannot_earn_default_reward(self):
        for tokens in (["<S>", "<E>"], ["<S>", "<T>", "<T>", "<E>"],
                       ["<S>"] + event(velocity=0) + ["<E>"],
                       ["<S>"] + event(duration=0) + ["<E>"]):
            with self.subTest(tokens=tokens):
                result = self.assert_invalid(tokens, "insufficient_notes")
                self.assertTrue(result.syntax_valid)
        # Explicit pure-grammar mode may allow empty but compilable MIDI.
        pure = MidiSyntaxMetric(self.tokenizer, min_sounding_notes=0)
        self.assertEqual(pure.evaluate_sequence(["<S>", "<E>"]).score, 1)

    def test_padding_requires_explicit_right_padding_mask(self):
        tokens = ["<S>"] + event() + ["<E>", "<P>", "<P>"]
        self.assert_valid(tokens, valid_mask=[True] * 6 + [False] * 2)
        self.assert_invalid(tokens, "tokens_after_end")
        for mask in ([True], [True, False, True] + [False] * 5, [2] * 8):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                self.metric.evaluate_sequence(tokens, valid_mask=mask)

    def test_canvas_mid_event_boundaries_and_explicit_suffix(self):
        original = ["<S>"] + event() + event(pitch=64) + ["<E>"]
        saved = deepcopy(original)
        for start in range(1, 5):
            for end in range(start + 1, 5):
                with self.subTest(start=start, end=end):
                    result = self.metric.evaluate_canvas(original[start:end], history=original[:start],
                                                         boundary_suffix=original[end:5])
                    self.assertEqual(result.score, 1, result.issue)
        self.assertEqual(original, saved)
        result = self.metric.evaluate_canvas(original[1:4], history=original[:1])
        self.assertEqual(result.issue.code, "incomplete_event")
        with self.assertRaises(ValueError):
            self.metric.evaluate_canvas(event(), boundary_suffix=event())

    def test_exact_256_token_canvas_cut_through_duration(self):
        source = ["<S>"] + event() * 130 + ["<E>"]
        start, end = 2, 258
        self.assertEqual(len(source[start:end]), 256)
        result = self.metric.evaluate_canvas(source[start:end], history=source[:start],
                                             boundary_suffix=source[end:261])
        self.assertEqual(result.score, 1, result.issue)
        result = self.metric.evaluate_canvas(source[start:end], history=source[:start])
        self.assertEqual(result.validity_score, 0)

    def test_cropped_clean_history_only(self):
        history = event()[1:] + event()
        generated = event()
        self.assertEqual(self.metric.evaluate_canvas(generated, history=history).validity_score, 0)
        self.assertEqual(self.metric.evaluate_canvas(generated, history=history, allow_cropped_history=True).score, 1)
        self.assertEqual(self.metric.evaluate_canvas(event()[1:], allow_cropped_history=True).score, 0)
        self.assertEqual(history, event()[1:] + event())

    def test_history_cannot_hide_empty_canvas(self):
        history = ["<S>"] + event()
        for generated in ([], ["<T>"], ["<T>", "<E>"], event(velocity=0), event(duration=0)):
            with self.subTest(generated=generated):
                self.assertEqual(self.metric.evaluate_canvas(generated, history=history).score, 0)
        self.assertEqual(self.metric.evaluate_canvas(["<E>"], history=history).score, 1)
        self.assertEqual(self.metric.evaluate_canvas(["<E>"]).score, 0)

    def test_canvas_padding_and_invalid_suffix(self):
        result = self.metric.evaluate_canvas(event() + ["<P>"] * 3, valid_mask=[1] * 4 + [0] * 3)
        self.assertEqual(result.score, 1, result.issue)
        result = self.metric.evaluate_canvas(event()[:3], boundary_suffix=[("dur", -1)])
        self.assertEqual(result.issue.code, "out_of_vocabulary")
        self.assertEqual(result.issue.token_index, 3)

    def test_compilation_errors_and_discarded_events(self):
        tokens = ["<S>"] + event() + ["<E>"]
        with patch.object(self.tokenizer, "detokenize", side_effect=RuntimeError("test converter failure")):
            result = self.assert_invalid(tokens, "midi_compilation")
            self.assertTrue(result.syntax_valid)
            self.assertFalse(result.midi_compiles)
        decoded = self.tokenizer.detokenize(tokens)
        decoded.note_msgs.clear()
        with patch.object(self.tokenizer, "detokenize", return_value=decoded):
            self.assert_invalid(tokens, "midi_compilation")

    def test_actual_midi_channel_limit(self):
        tokens = ["<S>"]
        for instrument in self.tokenizer.instruments_nd[:16]:
            tokens += event(instrument)
        tokens += ["<E>"]
        result = self.assert_invalid(tokens, "midi_compilation")
        self.assertTrue(result.syntax_valid)

    def test_optional_chronological_order(self):
        tokens = ["<S>"] + event(onset=100) + event(onset=0) + ["<E>"]
        self.assert_valid(tokens)
        strict = MidiSyntaxMetric(self.tokenizer, check_time_order=True)
        self.assertEqual(strict.evaluate_sequence(tokens).issue.code, "time_order")
        ordered = ["<S>"] + event(onset=100) + ["<T>"] + event(onset=0) + ["<E>"]
        self.assertEqual(strict.evaluate_sequence(ordered).score, 1)

    def test_delimiter_and_no_pedal_configuration(self):
        path = Path(__file__).resolve().parents[1] / "improvnet/tokenizer/config.json"
        with path.open() as handle:
            config = json.load(handle)
        config["tokenizer"]["abs"].update(include_delimiter=True)
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory) / "config.json"
            with temporary.open("w") as handle:
                json.dump(config, handle)
            tokenizer = AbsTokenizer(temporary)
        metric = MidiSyntaxMetric(tokenizer)
        tokens = ["<S>"] + event() + ["<X>", "<E>"]
        result = metric.evaluate_sequence(tokens)
        self.assertEqual(result.score, 1, result.issue)
        self.assertEqual(metric.evaluate_sequence(tokens[:-2] + ["<X>", "<X>", "<E>"]).issue.code, "duplicate_delimiter")
        self.assert_invalid(tokens, "out_of_vocabulary")
        self.assert_invalid(["<S>"] + event() + ["<PED_ON>", "<E>"], "out_of_vocabulary")
        with patch.object(tokenizer, "include_pedal", True), self.assertRaises(ValueError):
            MidiSyntaxMetric(tokenizer)

    def test_graded_score_and_all_error_diagnostics(self):
        good = ["<S>"] + event() * 3 + ["<E>"]
        one = good.copy()
        one[4] = ("dur", -1)
        two = one.copy()
        two[8] = ("dur", -1)
        clean, partly, worse = [self.metric.evaluate_sequence(x) for x in (good, one, two)]
        self.assertGreater(clean.score, partly.score)
        self.assertGreater(partly.score, worse.score)
        self.assertGreater(worse.score, 0)
        self.assertEqual((partly.error_count, worse.error_count), (1, 2))
        self.assertEqual(len([i for i in worse.issues if i.code == "out_of_vocabulary"]), 2)
        self.assertFalse(partly.valid)
        self.assertIsNone(partly.midi_compiles)
        self.assertAlmostEqual(partly.score, 1 - 1 / len(one))
        summary = summarize_syntax([partly, worse])
        self.assertEqual(summary["valid_rate"], 0)
        self.assertGreater(summary["mean_score"], 0)

    def test_missing_field_is_a_graded_negative_not_silently_repaired(self):
        result = self.metric.evaluate_sequence(["<S>"] + event()[:3] + ["<E>"])
        self.assertFalse(result.valid)
        self.assertEqual(result.error_count, 1)
        self.assertAlmostEqual(result.score, .8)
        self.assertIsNone(result.midi_compiles)

    def test_clean_history_does_not_dilute_canvas_faults(self):
        damaged = event() + event()
        damaged[3] = ("dur", -1)
        short = self.metric.evaluate_canvas(damaged, history=["<S>"])
        long = self.metric.evaluate_canvas(damaged, history=["<S>"] + event() * 20)
        self.assertEqual(short.score, long.score)
        self.assertEqual(long.scored_tokens, 8)
        self.assertAlmostEqual(long.score, .875)

    def test_batch_rates_and_noise_bin_boundaries(self):
        good = ["<S>"] + event() + ["<E>"]
        bad = ["<S>", "<T>", "<E>"]
        results = self.metric.evaluate_batch([good, bad, good, bad], noise_levels=[0, .25, .5, 1])
        summary = summarize_syntax(results)
        self.assertEqual(summary["valid_rate"], .5)
        self.assertEqual(summary["failures"], {"insufficient_notes": 2})
        self.assertEqual([b["count"] for b in summary["noise_bins"]], [1, 1, 1, 1])
        self.assertEqual([b["valid_rate"] for b in summary["noise_bins"]], [1, 0, 1, 0])
        canvas_results = self.metric.evaluate_batch([event(), ["<E>"]], canvas=True,
                                                  histories=[["<S>"], ["<S>"] + event()])
        self.assertEqual([r.score for r in canvas_results], [1, 1])
        self.assertIsNone(summarize_syntax([])["valid_rate"])

    def test_argument_validation(self):
        with self.assertRaises(ValueError):
            MidiSyntaxMetric(min_sounding_notes=-1)
        with self.assertRaises(ValueError):
            MidiSyntaxMetric(min_sounding_notes=True)
        for noise in (-.1, 1.1, float("nan"), float("inf")):
            with self.subTest(noise=noise), self.assertRaises(ValueError):
                self.metric.evaluate_sequence([], noise_level=noise)
        with self.assertRaises(ValueError):
            self.metric.evaluate_batch([event()], noise_levels=[])
        with self.assertRaises(ValueError):
            self.metric.evaluate_batch([event()], histories=[[]])
        for edges in ((0, 1, 1), (.1, .5, 1), (0, float("nan"), 1)):
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                summarize_syntax([], noise_bins=edges)

    def test_real_example_midi_round_trips(self):
        directory = Path(__file__).resolve().parents[1] / "improvnet/inference/examples"
        paths = sorted(directory.glob("*.mid"))
        self.assertGreater(len(paths), 0)
        for path in paths:
            with self.subTest(path=path.name):
                tokens = self.tokenizer.tokenize(MidiDict.from_midi(str(path)))
                result = self.metric.evaluate_sequence(tokens)
                self.assertTrue(result.syntax_valid, result.issue)
                self.assertTrue(result.midi_compiles, result.issue)
                consecutive = any(a == b == "<T>" for a, b in zip(tokens, tokens[1:]))
                self.assertEqual(result.score, 0. if consecutive else 1.)


if __name__ == "__main__":
    unittest.main()
