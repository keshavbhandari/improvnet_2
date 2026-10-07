"""CPU invariants for typed, promptless, multi-block style transfer."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch

from improvnet.archive.uniform_diffusion.inference.uniform_diffusion_inference import (
    SamplerConfig, SlotVocabulary, closed_source, constrained_probabilities,
    entropy_selection, generate_batch, padded_ids, refine_stage, typed_corrupt,
)
from improvnet.archive.uniform_diffusion.inference.uniform_diffusion_evaluate import parse_args
from improvnet.archive.uniform_diffusion.inference.uniform_diffusion_evaluation_metrics import inspect_syntax
from improvnet.metrics.midi_syntax import MidiSyntaxMetric
from improvnet.utils.ar_utils import ProcessData


def event(pitch=60, instrument="Acoustic Piano"):
    return [("instrument", instrument), ("note", pitch, 80), ("onset", 0), ("dur", 100)]


class FakeModel:
    def __init__(self, processor):
        self.processor = processor
        self.config = SimpleNamespace(canvas_size=256, vocab_size=processor.tokenizer.vocab_size)
        self.calls, self.prefills, self.appends = [], [], []

    def encode_context(self, ids, **kwargs):
        (self.prefills if kwargs.get("cache") is None else self.appends).append((ids.clone(), kwargs))
        return SimpleNamespace(cache=object())

    def denoise(self, ids, **kwargs):
        self.calls.append((ids.clone(), kwargs))
        logits = torch.full((*ids.shape, self.config.vocab_size), -40.)
        # A structurally broken model still cannot emit T into note/timing slots.
        logits[..., self.processor.tokenizer.tok_to_id["<T>"]] = 1000.
        for token in (("note", 72, 80), ("onset", 200), ("dur", 200)):
            logits[..., self.processor.tokenizer.tok_to_id[token]] = 40.
        return logits


class ConstrainedInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.processor = ProcessData()

    def setUp(self):
        self.slots = SlotVocabulary(self.processor.tokenizer)

    def ids(self, sequence):
        return self.processor.tokens_to_tensor(sequence)[None]

    def stage(self, anchor, stage="timing", noise=1., sampler="flow", model=None, valid=None, **changes):
        model = model or FakeModel(self.processor)
        config = SamplerConfig(steps=4, sampler=sampler, **changes)
        valid = torch.ones_like(anchor, dtype=torch.bool) if valid is None else valid
        result = refine_stage(model, None, anchor, valid, self.slots, stage=stage, noise=noise, config=config,
            generators=[torch.Generator().manual_seed(7+i) for i in range(len(anchor))],
            sampler_generator=torch.Generator().manual_seed(9))
        return model, result

    def test_configuration_limits_and_defaults(self):
        self.assertEqual(SamplerConfig().steps, 48)
        self.assertEqual(parse_args([]).steps, [48])
        for value in (0, 49, True, 1.5):
            with self.assertRaises(ValueError):
                SamplerConfig(steps=value)
        for value in (float("nan"), 0, -1):
            with self.assertRaises(ValueError):
                SamplerConfig(temperature_start=value)

    def test_type_constraints_and_structural_point_masses(self):
        source = ["<S>"] + event() + ["<T>", "<D>", "<E>"]
        anchor = self.ids(source)
        for stage, indices in (("timing", [3, 4]), ("pitch-velocity", [2])):
            editable, allowed = self.slots.constraints(anchor, torch.ones_like(anchor, dtype=torch.bool), stage)
            self.assertEqual(editable.nonzero()[:, 1].tolist(), indices)
            for index in range(len(source)):
                if index not in indices:
                    self.assertEqual(allowed[0, index].nonzero().flatten().tolist(), [int(anchor[0, index])])
                else:
                    categories = {self.processor.tokenizer.id_to_tok[i][0] for i in allowed[0, index].nonzero().flatten().tolist()}
                    self.assertEqual(categories, {source[index][0]})

    def test_no_silent_candidates(self):
        for role in (2, 3):
            self.assertTrue(all(self.processor.tokenizer.id_to_tok[i][-1] > 0 for i in self.slots.candidates[role].nonzero().flatten().tolist()))

    def test_corruption_is_nested_and_frozen_slots_never_change(self):
        anchor = self.ids(["<S>"] + event()*64 + ["<D>", "<E>"])
        editable, allowed = self.slots.constraints(anchor, torch.ones_like(anchor, dtype=torch.bool), "timing")
        low, replaced_low = typed_corrupt(anchor, editable, allowed, .2, torch.Generator().manual_seed(5))
        high, replaced_high = typed_corrupt(anchor, editable, allowed, .8, torch.Generator().manual_seed(5))
        self.assertTrue(bool((~replaced_low | replaced_high).all()))
        torch.testing.assert_close(low[replaced_low], high[replaced_low])
        torch.testing.assert_close(high[~editable], anchor[~editable])
        self.assertFalse(bool(replaced_high[~editable].any()))
        self.assertTrue(bool(allowed.gather(-1, high[..., None]).all()))

    def test_timing_then_notes_and_self_conditioning_clamps(self):
        anchor = self.ids(["<S>"] + event() + ["<T>", "<D>", "<E>"])
        original = anchor.clone()
        model, (timing, _, traces, _) = self.stage(anchor)
        decoded = self.processor.tokenizer.decode(timing[0].tolist())
        self.assertEqual(decoded, ["<S>", event()[0], event()[1], ("onset", 200), ("dur", 200), "<T>", "<D>", "<E>"])
        _, (notes, _, _, _) = self.stage(timing, stage="pitch-velocity")
        final = self.processor.tokenizer.decode(notes[0].tolist())
        self.assertEqual(final[2], ("note", 72, 80))
        self.assertEqual(final[:2]+final[3:], decoded[:2]+decoded[3:])
        self.assertEqual(len(traces[0]), 2)
        _, allowed = self.slots.constraints(anchor, torch.ones_like(anchor, dtype=torch.bool), "timing")
        sc = model.calls[1][1]["self_conditioning_logits"].softmax(-1)
        self.assertTrue(bool((sc[~allowed] == 0).all()))
        self.assertEqual(model.calls[1][1]["self_conditioning_temperature"], 1.)
        torch.testing.assert_close(anchor, original)

    def test_effective_noise_scales_by_editable_valid_count_not_padding(self):
        anchor = self.ids(event())
        valid = torch.ones_like(anchor, dtype=torch.bool)
        _, (_, _, timing, _) = self.stage(anchor, noise=.8)
        _, (_, _, notes, _) = self.stage(anchor, stage="pitch-velocity", noise=.8)
        self.assertAlmostEqual(timing[0][0]["t_effective"], .4, places=6)
        self.assertAlmostEqual(notes[0][0]["t_effective"], .2, places=6)
        padded = torch.cat((anchor, torch.full((1, 8), self.processor.tokenizer.pad_id, dtype=torch.long)), -1)
        mask = torch.cat((valid, torch.zeros(1, 8, dtype=torch.bool)), -1)
        _, (out, _, trace, _) = self.stage(padded, noise=.8, valid=mask)
        self.assertAlmostEqual(trace[0][0]["t_effective"], .4, places=6)
        torch.testing.assert_close(out[~mask], padded[~mask])

    def test_zero_noise_skips_model_and_both_stages(self):
        anchor = self.ids(event())
        model, (result, initial, trace, changes) = self.stage(anchor, noise=0)
        self.assertEqual(model.calls, [])
        self.assertEqual(trace, [[]])
        torch.testing.assert_close(result, anchor)
        torch.testing.assert_close(initial, anchor)
        self.assertEqual(changes[0]["replaced_positions"], 0)

    def test_drums_have_same_fields_and_keep_instrument(self):
        anchor = self.ids(event(36, "drum"))
        _, (timing, _, _, _) = self.stage(anchor)
        _, (result, _, _, _) = self.stage(timing, stage="pitch-velocity")
        decoded = self.processor.tokenizer.decode(result[0].tolist())
        self.assertEqual(decoded, event(72, "drum")[:2] + [("onset", 200), ("dur", 200)])

    def test_partial_event_roles_do_not_depend_on_slice_offset(self):
        source = ["<S>"] + event()*128 + ["<E>"]
        # Position 255 is onset, 256 is duration; the event straddles the cut.
        for start, stage in ((0, "timing"), (256, "timing"), (256, "pitch-velocity")):
            anchor = self.ids(source[start:start+256])
            editable, allowed = self.slots.constraints(anchor, torch.ones_like(anchor, dtype=torch.bool), stage)
            for position in editable[0].nonzero().flatten().tolist():
                expected = source[start+position][0]
                actual = {self.processor.tokenizer.id_to_tok[i][0] for i in allowed[0, position].nonzero().flatten().tolist()}
                self.assertEqual(actual, {expected})
        self.assertEqual(source[256][0], "dur")

    def test_entropy_selection_bound_and_padding(self):
        entropy = torch.tensor([[.2, .3, .8, 0.], [.1, .2, .3, .4]])
        editable = torch.tensor([[1, 1, 1, 0], [0, 0, 0, 0]], dtype=torch.bool)
        self.assertEqual(entropy_selection(entropy, editable, .2).tolist(), [[True, True, False, False], [False]*4])

    def test_constraints_reject_invalid_stage_masks_and_logits(self):
        anchor = self.ids(event())
        with self.assertRaises(ValueError):
            self.slots.constraints(anchor, torch.ones_like(anchor), "timing")
        with self.assertRaises(ValueError):
            self.slots.constraints(anchor, anchor.bool(), "instrument")
        with self.assertRaises(ValueError):
            constrained_probabilities(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2, dtype=torch.bool), [1.])

    def test_probability_normalization_and_final_temperature_sampling(self):
        allowed = torch.tensor([[[True, False, True]]])
        scores, probs, entropy = constrained_probabilities(torch.zeros(1, 1, 3), allowed, [.5])
        torch.testing.assert_close(probs, torch.tensor([[[.5, 0., .5]]]))
        self.assertAlmostEqual(float(entropy), 0.69314718056, places=6)
        class EqualModel(FakeModel):
            def denoise(self, ids, **kwargs):
                logits = torch.full((*ids.shape, self.config.vocab_size), -100.)
                for token in (("note", 60, 80), ("note", 72, 80)):
                    logits[..., self.processor.tokenizer.tok_to_id[token]] = 0.
                return logits
        model = EqualModel(self.processor)
        anchor = self.ids(event()*64)
        _, (result, _, _, _) = self.stage(anchor, stage="pitch-velocity", model=model, adaptive_stopping=False)
        pitches = {t[1] for t in self.processor.tokenizer.decode(result[0].tolist()) if isinstance(t, tuple) and t[0] == "note"}
        self.assertEqual(pitches, {60, 72})

    def test_source_endpoint_is_closed_before_not_after_generation(self):
        source = ["<S>"] + event()*600 + ["<D>", "<E>"]
        chosen = closed_source(source, 2048)
        self.assertEqual(len(chosen), 2045)
        self.assertEqual(chosen[-1][0], "dur")
        self.assertEqual(len(source), 2403)
        self.assertEqual(closed_source(["<S>"]+event()+["<E>"]+event(), 256), ["<S>"]+event()+["<E>"])

    def test_promptless_multiblock_history_is_generated_and_never_gt(self):
        source = ["<S>"] + event()*130 + ["<D>", "<E>"]
        example = dict(example_id="example_00", sample_index=0, source_genre="classical", tokens=source)
        untouched = deepcopy(example)
        for sampler in ("flow", "entropy"):
            model = FakeModel(self.processor)
            result = generate_batch(model, self.processor, [example], noise=1., config=SamplerConfig(steps=3, sampler=sampler), max_blocks=3)[0]
            self.assertEqual(len(result["blocks"]), 3)
            self.assertEqual(result["stop_reason"], "source_end_token")
            self.assertEqual(model.prefills[0][0].shape, (1, 0))
            self.assertEqual(int(model.prefills[0][1]["genre"][0]), self.processor.get_genre_id("jazz"))
            self.assertEqual(len(model.appends), 2)
            generated_first = self.processor.tokenizer.decode(model.appends[0][0][0].tolist())
            self.assertEqual(generated_first, result["generated_tokens"][:256])
            self.assertNotEqual(generated_first, source[:256])
            self.assertIsNone(model.calls[0][1]["self_conditioning_logits"])
            self.assertIsNone(model.calls[2][1]["self_conditioning_logits"])
            grade = MidiSyntaxMetric(self.processor.tokenizer).evaluate_sequence(result["generated_tokens"])
            self.assertTrue(grade.valid, grade)
            self.assertTrue(grade.midi_compiles)
            self.assertEqual(result["generated_tokens"][-2:], ["<D>", "<E>"])
        self.assertEqual(example, untouched)

    def test_per_row_stopping_retains_finished_row_and_empty_placeholders(self):
        class ChangingModel(FakeModel):
            def denoise(self, ids, **kwargs):
                step = len(self.calls)
                logits = super().denoise(ids, **kwargs)
                if step < 3:
                    logits[1, :, self.processor.tokenizer.tok_to_id[("onset", 200)]] = -40.
                    logits[1, :, self.processor.tokenizer.tok_to_id[("onset", 0 if step%2 else 100)]] = 40.
                return logits
        anchor = self.ids(event()).repeat(3, 1)
        valid = torch.ones_like(anchor, dtype=torch.bool)
        valid[2] = False
        model, (result, _, traces, _) = self.stage(anchor, model=ChangingModel(self.processor), valid=valid)
        self.assertEqual(list(map(len, traces)), [2, 4, 0])
        self.assertFalse(bool(model.calls[2][1]["attention_mask"][0].any()))
        torch.testing.assert_close(result[2], anchor[2])

    def test_partial_last_block_no_padding_emitted_and_syntax_across_cut(self):
        source = ["<S>"]+event()*70+["<E>"]
        examples = [dict(example_id="example_00", sample_index=0, source_genre="classical", tokens=source)]
        result = generate_batch(FakeModel(self.processor), self.processor, examples, noise=.2, config=SamplerConfig(steps=2), max_blocks=2)[0]
        self.assertEqual(len(result["generated_tokens"]), len(source))
        self.assertNotIn(self.processor.tokenizer.pad_tok, result["generated_tokens"])
        report = dict(examples=examples, results=[result])
        inspect_syntax(report, self.processor)
        self.assertTrue(result["frozen_slots_preserved"])
        self.assertTrue(result["syntax"]["valid"])
        self.assertTrue(result["blocks"][0]["syntax"]["prefix_valid"])
        self.assertFalse(result["blocks"][0]["syntax"]["valid"])
        self.assertTrue(result["blocks"][1]["syntax"]["valid"])

    def test_padded_ids_explicit_mask_and_width(self):
        ids, valid = padded_ids([event(), []], self.processor, "cpu", 256)
        self.assertEqual(valid.sum(-1).tolist(), [4, 0])
        self.assertEqual(ids[0, 4:].unique().tolist(), [self.processor.tokenizer.pad_id])
        with self.assertRaises(ValueError):
            padded_ids([event()], self.processor, "cpu", 3)

    def test_timing_only_skips_note_stage_and_preserves_every_note(self):
        source = ["<S>"] + event()*70 + ["<D>", "<E>"]
        examples = [dict(example_id="example_00", sample_index=0, source_genre="classical", tokens=source)]
        model = FakeModel(self.processor)
        result = generate_batch(model, self.processor, examples, noise=1., config=SamplerConfig(steps=3),
                                max_blocks=2, genre="blues", strategy="timing")[0]
        self.assertEqual(len(model.calls), 4)  # Two timing passes per block; no note passes.
        self.assertEqual(int(model.prefills[0][1]["genre"][0]), self.processor.get_genre_id("blues"))
        self.assertEqual(result["max_steps"], 3)
        self.assertEqual(result["strategy"], "timing")
        self.assertTrue(all(set(b["stages"]) == {"timing"} for b in result["blocks"]))
        self.assertTrue(all(a == b for a, b in zip(source, result["generated_tokens"])
                            if not isinstance(a, tuple) or a[0] not in ("onset", "dur")))
        self.assertTrue(any(a != b for a, b in zip(source, result["generated_tokens"])))


if __name__ == "__main__":
    unittest.main()
