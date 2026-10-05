"""Selection, metric support, real-model padding and output safety tests."""

from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from improvnet.inference.uniform_diffusion_evaluate import main, parse_args
from improvnet.inference.uniform_diffusion_inference import SamplerConfig, generate_batch
from improvnet.inference.uniform_diffusion_evaluation_metrics import (
    compact_report, inspect_syntax, score_metrics, select_test_examples, statistics_for, summarize, write_report,
)
from improvnet.model.uniform_diffusion_config import UniformDiffusionConfig
from improvnet.model.uniform_diffusion_model import UniformDiffusionModel
from improvnet.utils.ar_utils import ProcessData


def event(pitch=60):
    return [("instrument", "Acoustic Piano"), ("note", pitch, 80), ("onset", 0), ("dur", 100)]


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.processor = ProcessData()

    def test_heldout_selection_is_classical_deduplicated_beginning_and_closed(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"data.jsonl"
            rows = [dict(midi_filepath=name, genre=genre, tokens=["<S>"]+event()*140+["<E>"])
                    for name, genre in (("j.mid", "jazz"), ("a.mid", "Classical"), ("a.mid", "classical"), ("b.mid", "classical"))]
            offsets = []
            with path.open("wb") as handle:
                for row in rows:
                    offsets.append(handle.tell())
                    handle.write((json.dumps(row)+"\n").encode())
            settings = dict(JSONL_FILES=[str(path)], DATA_SPLIT_SEED=42, DATA_SPLIT_RATIOS=[.97, .02, .01])
            with patch("improvnet.utils.ar_utils.load_split_byte_index", return_value=offsets) as index:
                examples, selection = select_test_examples(settings, self.processor, count=2, max_blocks=2, seed=3)
            self.assertEqual({e["midi_filepath"] for e in examples}, {"a.mid", "b.mid"})
            self.assertEqual(index.call_args.args[1], "test")
            self.assertTrue(all(e["target_start"] == 0 and e["tokens"][0] == "<S>" for e in examples))
            self.assertTrue(all(e["tokens"][-1][0] == "dur" and len(e["tokens"]) <= 512 for e in examples))
            self.assertTrue(all(e["source_syntax"]["valid"] for e in examples))
            self.assertEqual(selection["candidate_test_records"], 4)

    def test_selection_requires_checkpoint_split_provenance(self):
        with self.assertRaises(ValueError):
            select_test_examples({}, self.processor, count=1, max_blocks=1, seed=42)

    def test_cli_rejects_out_of_scope_limits_and_duplicates(self):
        for args in (["--steps", "49"], ["--max-blocks", "9"], ["--noise-levels", ".2", ".2"], ["--noise-levels", "nan"], ["--batch-size", "5"]):
            with self.subTest(args=args), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                parse_args(args)

    def test_existing_output_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as root:
            dummy = Path(root)/"checkpoint.pt"
            dummy.touch()
            output = Path(root)/"out"
            output.mkdir()
            (output/"keep.mid").touch()
            with patch("improvnet.inference.uniform_diffusion_evaluate.load_model") as loader, self.assertRaises(FileExistsError):
                main(["--model", str(dummy), "--genre-checkpoint", str(dummy), "--continuity-checkpoint", str(dummy), "--output-dir", str(output)])
            loader.assert_not_called()
            self.assertTrue((output/"keep.mid").exists())

    def test_statistics_missing_scores_not_zeros(self):
        self.assertEqual(statistics_for([None])["count"], 0)
        self.assertIsNone(statistics_for([None])["mean"])
        self.assertEqual(statistics_for([.2, None, .8])["mean"], .5)
        with self.assertRaises(ValueError):
            statistics_for([float("nan")])

    def test_metric_uses_generated_history_paired_baseline_first_block_na(self):
        source = ["<S>"]+event()+event()+["<E>"]
        generated = ["<S>"]+event(72)+event(72)+["<E>"]
        example = dict(example_id="example_00", tokens=source)
        row = dict(example_id="example_00", initial_noise=.2, source_genre="classical", target_genre="jazz", generated_tokens=generated,
                   blocks=[dict(block=1, source_start=0, source_end=5), dict(block=2, source_start=5, source_end=10)])
        report = dict(examples=[example], results=[row])
        inspect_syntax(report, self.processor)
        class Genre:
            def probabilities(self, tokens, **kwargs):
                jazz = .4 if any(isinstance(t, tuple) and t[:2] == ("note", 72) for t in tokens) else .1
                return dict(jazz=jazz, classical=1-jazz, blues=0.)
        class Continuity:
            def score_boundary(self, previous, current, **kwargs):
                a = next(t[1] for t in reversed(previous) if isinstance(t, tuple) and t[0] == "note")
                b = next(t[1] for t in current if isinstance(t, tuple) and t[0] == "note")
                return .9 if a == b == 72 else .8 if a == b == 60 else .1
        def load(path, kind, device, tokenizer):
            return (Genre() if kind == "genre" else Continuity()), {"trained_steps": 1}
        with patch("improvnet.inference.uniform_diffusion_evaluation_metrics.load_encoder", side_effect=load):
            score_metrics(report, SimpleNamespace(genre_checkpoint="genre", continuity_checkpoint="continuity"), self.processor, torch.device("cpu"))
        first, second = row["blocks"]
        self.assertEqual(first["metrics"]["continuity"]["status"], "not_applicable")
        self.assertEqual(second["metrics"]["continuity"]["score"], .9)
        self.assertEqual(second["metrics"]["continuity"]["reference_score"], .8)
        self.assertEqual(second["metrics"]["continuity"]["source_history_score"], .1)
        self.assertAlmostEqual(row["metrics"]["genre"]["jazz_delta_from_reference"], .3)
        self.assertEqual(row["metrics"]["continuity"]["boundaries"], 1)

    def test_invalid_results_remain_in_syntax_denominator_not_metric_means(self):
        row = dict(initial_noise=.2, syntax=dict(valid=True, midi_compiles=True, score=1.), frozen_slots_preserved=True,
                   blocks=[dict(stages={"timing": dict(actual_steps=3), "pitch-velocity": dict(actual_steps=4)})],
                   metrics=dict(pitch=dict(score=.6), genre=dict(jazz_probability=.3), continuity=dict(score=.8)))
        invalid = deepcopy(row)
        invalid["syntax"].update(valid=False, midi_compiles=False, score=0.)
        invalid["metrics"]["genre"]["jazz_probability"] = .99
        summary = summarize([row, invalid])[0]
        self.assertEqual(summary["pieces"], 2)
        self.assertEqual(summary["valid"], 1)
        self.assertEqual(summary["syntax_score"]["mean"], .5)
        self.assertEqual(summary["jazz_probability"]["count"], 1)
        self.assertEqual(summary["jazz_probability"]["mean"], .3)

    def test_real_model_batched_padding_cache_and_midi_roundtrip(self):
        torch.manual_seed(4)
        config = UniformDiffusionConfig(embed_dim=32, n_heads=4, n_kv_heads=2, n_layers=1, dropout=0., gradient_checkpointing=False)
        model = UniformDiffusionModel(config).eval()
        sources = [["<S>"]+event()*70+["<D>", "<E>"], ["<S>"]+event()+["<E>"]]
        examples = [dict(example_id=f"example_{i:02d}", sample_index=i, source_genre="classical", tokens=tokens) for i, tokens in enumerate(sources)]
        weights_before = {name: value.clone() for name, value in model.state_dict().items()}
        rows = generate_batch(model, self.processor, examples, noise=1., config=SamplerConfig(steps=2), max_blocks=2)
        report = dict(examples=examples, results=rows)
        inspect_syntax(report, self.processor)
        for row, source in zip(rows, sources):
            self.assertTrue(row["syntax"]["valid"], row["syntax"])
            self.assertTrue(row["frozen_slots_preserved"])
            self.assertEqual(len(row["generated_tokens"]), len(source))
            midi = self.processor.tokens_to_midi(row["generated_tokens"])
            self.assertEqual(sum(m.type == "note_on" and m.velocity > 0 for track in midi.tracks for m in track),
                             sum(isinstance(t, tuple) and t[0] == "note" for t in source))
        for name, weight in model.state_dict().items():
            torch.testing.assert_close(weight, weights_before[name])

    def test_step_sweep_parser_and_separate_summary_groups(self):
        args = parse_args(["--steps", "16", "32", "48", "--strategy", "timing", "--metrics", "genre"])
        self.assertEqual(args.steps, [16, 32, 48])
        self.assertFalse(args.save_details)
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--steps", "16", "16"])
        row = dict(initial_noise=.2, max_steps=16, syntax=dict(valid=True, midi_compiles=True, score=1.),
                   frozen_slots_preserved=True, blocks=[], metrics={name: {} for name in ("pitch", "genre", "continuity")})
        other = dict(row, max_steps=48)
        self.assertEqual([s["steps"] for s in summarize([row, other])], [16, 48])

    def test_compact_json_has_no_tokens_traces_or_duplicate_raw_files(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            genre = dict(jazz_probability=1e-7, blues_probability=.12, reference_jazz_probability=2e-7, reference_blues_probability=.02)
            result = dict(example_id="example_00", initial_noise=.4, max_steps=16,
                          syntax=dict(valid=True), pitch_velocity_preserved=True, timing_changed_fraction=.25,
                          musical_diagnostics=dict(onset_decrease_rate=0.),
                          blocks=[dict(stages=dict(timing=dict(actual_steps=16, trace=[{"huge": "ignored"}]*500)))],
                          generated_tokens=["huge"]*10000, output_midi=str(directory/"output.mid"), metrics=dict(genre=genre))
            report = dict(results=[result], examples=[dict(tokens=["huge"]*10000)], selection=dict(input="source.mid"),
                          target_genre="blues", strategy="timing", checkpoint=dict(step=40000), checkpoint_path="model.pt",
                          metric_checkpoints=dict(genre=dict(path="genre.pt")), max_blocks=8, seed=42,
                          sampler_config=dict(steps=16, sampler="flow"), step_budgets=[16, 32, 48], noise_levels=[.4],
                          metrics_requested="genre", elapsed_seconds=1., generation_elapsed_seconds=1.)
            write_report(directory, report)
            content = (directory/"scores.json").read_text()
            self.assertLess(len(content), 2000)
            for forbidden in ("generated_tokens", "tokens", "trace", "model_config", "probabilities"):
                self.assertNotIn(forbidden, content)
            self.assertFalse((directory/"configuration.json").exists())
            self.assertFalse((directory/"generation.json").exists())
            compact = json.loads(content)
            self.assertEqual(compact["results"][0]["blues"], .12)
            self.assertEqual(compact["results"][0]["jazz"], 1e-7)
            self.assertEqual(compact["results"][0]["midi"], "output.mid")
            self.assertEqual(compact["steps"], [16, 32, 48])


if __name__ == "__main__":
    unittest.main()
