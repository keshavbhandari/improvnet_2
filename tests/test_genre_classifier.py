"""CPU model/data/calibration checks and two tiny synthetic optimizer steps."""

from collections import Counter
from dataclasses import asdict, replace
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from improvnet.tokenizer.absolute import AbsTokenizer
from improvnet.metrics.genre_classifier.audit import audit_dataset
from improvnet.metrics.genre_classifier.data import BalancedGenreSampler, GenreCollator, GenreDataset, load_manifest
from improvnet.metrics.genre_classifier.evaluation import class_weights, classification_metrics, evaluate, fit_temperature
from improvnet.metrics.genre_classifier.genre_classifier_model import GenreClassifier, GenreClassifierConfig
from improvnet.metrics.genre_classifier.scoring import GenreClassifierMetric
from improvnet.metrics.genre_classifier.tokenization import classifier_tokens, transpose_melodic_notes, vocabulary_digest
from improvnet.metrics.genre_classifier.genre_classifier_train import format_classification_metrics, log_classification_metrics, prepare_run_directory, train, validate_config
from test_genre_classifier_audit import tiny_records, write_records


class GenreClassifierTests(unittest.TestCase):
    def test_console_metrics_include_groups_genres_and_empty_support(self):
        metrics = classification_metrics([[8, 2, 0], [1, 9, 0], [0, 0, 0]], ["classical", "jazz", "blues"])
        text = format_classification_metrics(metrics)
        self.assertIn("all: accuracy=0.850, macro_f1=", text)
        self.assertIn("jazz_blues: accuracy=0.900, macro_f1=", text)
        self.assertIn("remaining: accuracy=0.800, macro_f1=", text)
        self.assertIn("jazz: accuracy=0.900, f1=", text)
        self.assertIn("blues: accuracy=N/A, f1=N/A", text)
        empty = classification_metrics([[0, 0], [0, 0]], ["jazz", "blues"])
        self.assertIn("remaining: accuracy=N/A, macro_f1=N/A", format_classification_metrics(empty))

    def test_interrupted_run_directory_is_reusable_and_preserves_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(prepare_run_directory(root, resuming=False))
            (root / "metrics.jsonl").write_text("old log\n")
            (root / "best_checkpoint.pt").write_bytes(b"old best")
            (root / "calibrated_checkpoint.pt").write_bytes(b"old calibration")
            (root / "tensorboard").mkdir()
            (root / "tensorboard" / "events").write_bytes(b"old events")
            self.assertIsNone(prepare_run_directory(root, resuming=True))
            self.assertTrue((root / "metrics.jsonl").exists())
            archive = prepare_run_directory(root, resuming=False)
            self.assertEqual((archive / "metrics.jsonl").read_text(), "old log\n")
            self.assertEqual((archive / "best_checkpoint.pt").read_bytes(), b"old best")
            self.assertEqual((archive / "tensorboard" / "events").read_bytes(), b"old events")
            self.assertFalse((root / "metrics.jsonl").exists())
            self.assertIsNone(prepare_run_directory(root, resuming=False))
            (root / "metrics.jsonl").write_text("second log\n")
            second = prepare_run_directory(root, resuming=False)
            self.assertNotEqual(second, archive)
            self.assertTrue(archive.exists())

    def test_missing_manifest_explains_required_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing_manifest.json"
            with self.assertRaises(FileNotFoundError) as caught:
                load_manifest(path)
            self.assertIn(str(path), str(caught.exception))
            self.assertIn("python -m improvnet.metrics.genre_classifier.audit", str(caught.exception))
            self.assertIn("--output-prefix", str(caught.exception))

    def test_values_only_settings_build_model_config_and_validate_lengths(self):
        import ast
        from improvnet.metrics.genre_classifier import genre_classifier_config as defaults
        tree = ast.parse(Path(defaults.__file__).read_text())
        self.assertFalse(any(isinstance(node, (ast.ClassDef, ast.FunctionDef)) for node in ast.walk(tree)))
        self.assertFalse(hasattr(defaults, "MODEL"))
        settings = SimpleNamespace(**{key: value for key, value in vars(defaults).items() if key.isupper()})
        settings.EMBED_DIM, settings.N_HEADS, settings.N_KV_HEADS, settings.N_LAYERS = 32, 4, 2, 1
        settings.MIN_LENGTH, settings.MAX_LENGTH, settings.EVAL_LENGTH = 8, 32, 16
        validate_config(settings)
        config = GenreClassifierConfig.from_settings(settings, vocab_size=100, num_genres=3, pad_id=2)
        self.assertEqual((config.embed_dim, config.n_heads, config.n_kv_heads, config.n_layers, config.max_length), (32, 4, 2, 1, 32))
        self.assertEqual(GenreClassifierConfig(**asdict(config)), config)
        for key, value in (("MIN_LENGTH", 33), ("MAX_LENGTH", 0), ("EVAL_LENGTH", 33), ("MIN_LENGTH", True)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_config(SimpleNamespace(**{**vars(settings), key: value}))

    def setUp(self):
        torch.manual_seed(14)
        self.tokenizer = AbsTokenizer()
        self.labels = ["classical", "jazz", "rock"]
        self.config = GenreClassifierConfig(vocab_size=self.tokenizer.vocab_size, num_genres=3,
                                           pad_id=self.tokenizer.pad_id, embed_dim=32, n_heads=4,
                                           n_layers=2, max_length=16, dropout=0.)
        self.model = GenreClassifier(self.config).eval()
        self.tokens = ["<S>", ("instrument", "Acoustic Piano"), ("note", 60, 80), ("onset", 0), ("dur", 100), "<E>"]
        self.ids = torch.tensor([self.tokenizer.tok_to_id[t] for t in self.tokens])

    def test_mixed_lengths_padding_and_left_padding_are_invariant(self):
        short, long = self.ids[:4], self.ids
        batch = GenreCollator(self.tokenizer.pad_id)([{"ids": short, "label": 0}, {"ids": long, "label": 1}])
        actual = self.model(batch["ids"], batch["attention_mask"])
        torch.testing.assert_close(actual[0], self.model(short[None])[0], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(actual[1], self.model(long[None])[0], atol=1e-6, rtol=1e-5)
        changed = batch["ids"].clone()
        changed[~batch["attention_mask"]] = 99999
        torch.testing.assert_close(actual, self.model(changed, batch["attention_mask"]))
        left = torch.cat((torch.full((2,), self.tokenizer.pad_id), short))[None]
        torch.testing.assert_close(self.model(left)[0], actual[0], atol=1e-6, rtol=1e-5)

    def test_bidirectional_context_and_backward(self):
        changed = self.ids.clone()
        changed[-1] = self.tokenizer.tok_to_id[("note", 67, 80)]
        _, one = self.model(self.ids[None], return_embedding=True)
        _, two = self.model(changed[None], return_embedding=True)
        self.assertFalse(torch.allclose(one, two))
        loss = F.cross_entropy(self.model(torch.stack((self.ids, changed))), torch.tensor([0, 1]))
        loss.backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in self.model.parameters()))

    def test_invalid_inputs_and_config(self):
        for ids in (torch.empty(1, 0, dtype=torch.long), torch.full((1, 3), self.tokenizer.pad_id),
                    self.ids.float()[None], torch.zeros(1, 17, dtype=torch.long)):
            with self.subTest(shape=ids.shape), self.assertRaises(ValueError):
                self.model(ids)
        with self.assertRaises(ValueError):
            self.model(self.ids[None], torch.ones(1, len(self.ids)))
        with self.assertRaises(ValueError):
            GenreClassifierConfig(vocab_size=32, num_genres=1)

    def test_defaults_match_uniform_backbone_without_allocating_real_weights(self):
        from improvnet.model.uniform_diffusion_config import UniformDiffusionConfig
        classifier, uniform = GenreClassifierConfig(), UniformDiffusionConfig()
        for field in ("embed_dim", "n_heads", "n_kv_heads", "n_layers", "ffn_mult", "dropout", "gradient_checkpointing", "vocab_size"):
            self.assertEqual(getattr(classifier, field), getattr(uniform, field), field)
        with torch.device("meta"):
            model = GenreClassifier(classifier)
        self.assertEqual(len(model.layers), 20)
        self.assertEqual(model.layers[0].ffn.w1.out_features, 5504)
        self.assertEqual(model.layers[0].attn.head_dim, 128)
        self.assertEqual(model.head.out_features, 12)
        self.assertGreater(sum(p.numel() for p in model.parameters()), 800000000)
        for kwargs in ({"n_heads": 4, "n_kv_heads": 3}, {"ffn_mult": float("nan")}):
            with self.assertRaises(ValueError):
                GenreClassifierConfig(**kwargs)

    def test_gqa_and_gradient_checkpointing_match_dense_backward(self):
        cfg = replace(self.config, n_kv_heads=2, gradient_checkpointing=True)
        checked = GenreClassifier(cfg).train()
        direct = GenreClassifier(replace(cfg, gradient_checkpointing=False)).train()
        direct.load_state_dict(checked.state_dict())
        for model in (checked, direct):
            F.cross_entropy(model(self.ids[None]), torch.tensor([1])).backward()
        torch.testing.assert_close(checked(self.ids[None]), direct(self.ids[None]))
        for one, two in zip(checked.parameters(), direct.parameters()):
            torch.testing.assert_close(one.grad, two.grad)

    def test_block_matches_uniform_backbone_math_with_identical_weights(self):
        from improvnet.model.uniform_diffusion_model import SharedBlock
        from improvnet.metrics.genre_classifier.genre_classifier_model import EncoderBlock
        cfg = replace(self.config, n_kv_heads=2)
        classifier, uniform = EncoderBlock(cfg).eval(), SharedBlock(cfg).eval()
        classifier.load_state_dict(uniform.state_dict(), strict=True)
        x = torch.randn(2, 6, cfg.embed_dim)
        mask = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])
        positions = torch.arange(6).expand(2, -1)
        ours = classifier(x, mask, positions)
        reference, _, _ = uniform(x, positions, mask[:, None, None, :], is_causal=False)
        torch.testing.assert_close(ours[mask], reference[mask], rtol=1e-5, atol=1e-6)

    def test_balanced_sampler_focus_distribution_and_distributed_draws(self):
        counts = {"classical": 1000, "jazz": 50, "blues": 5, "rock": 20, "folk": 7}
        labels = sorted(counts)
        rows = [(0, index, label) for label, count in counts.items() for index in range(count)]
        data = SimpleNamespace(labels=labels, rows=rows, class_counts=counts,
                               label_to_id={label: i for i, label in enumerate(labels)})
        sampler = BalancedGenreSampler(data, 20000)
        draws = list(sampler)
        histogram = Counter(rows[index][2] for index, _ in draws)
        for label in labels:
            expected = (4. if label in ("jazz", "blues") else 1.) / 11.
            self.assertAlmostEqual(histogram[label] / len(draws), expected, delta=.015)
        one = list(BalancedGenreSampler(data, 20000, world_size=2, rank=0))
        two = list(BalancedGenreSampler(data, 20000, world_size=2, rank=1))
        combined = [value for pair in zip(one, two) for value in pair]
        self.assertEqual(combined, draws)
        sampler.set_epoch(1)
        self.assertNotEqual(list(sampler)[:10], draws[:10])

    def test_caps_only_training_and_repeated_draws_get_variable_crops(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            write_records(path, tiny_records())
            audit_dataset([path], Path(directory) / "manifest", progress_every=0)
            manifest = Path(directory) / "manifest.json"
            data = GenreDataset(manifest, "train", min_length=8, max_length=32, eval_length=16, class_caps={"classical": 1})
            try:
                self.assertEqual(data.class_counts["classical"], 1)
                self.assertEqual(data.class_counts["jazz"], 3)
                torch.testing.assert_close(data[(0, 5)]["ids"], data[(0, 5)]["ids"])
                self.assertGreater(len({len(data[(0, i)]["ids"]) for i in range(20)}), 1)
            finally:
                data.close()
            with self.assertRaises(ValueError):
                GenreDataset(manifest, "validation", class_caps={"classical": 1})

    def test_token_reader_matches_stored_schema_and_strips_labels(self):
        flat = self.tokens + [("prefix", "genre", "jazz"), ("prefix", "composer", "bach")]
        self.assertEqual(classifier_tokens(flat, self.tokenizer), self.tokens)
        stored = [["<S>"] * 5, [["instrument", "Acoustic Piano"], ["pitch", 60], ["velocity", 80], ["onset", 0], ["dur", 100]], ["<E>"] * 5]
        self.assertEqual(classifier_tokens(stored, self.tokenizer), self.tokens)
        legacy = ["<S>", ["Acoustic Piano", 60, 80], ["onset", 0], ["dur", 100], "<E>"]
        self.assertEqual(classifier_tokens(legacy, self.tokenizer), self.tokens)
        with self.assertRaises(ValueError):
            classifier_tokens(["<S>", "<T>", "<E>"], self.tokenizer)

    def test_transposition_preserves_intervals_drums_and_other_fields(self):
        import random
        from unittest.mock import Mock
        tokens = self.tokens[:-1] + [("instrument", "Acoustic Piano"), ("note", 64, 80),
                                    ("onset", 10), ("dur", 100),
                                    ("instrument", "drum"), ("note", 36, 80),
                                    ("onset", 10), ("dur", 100), "<E>"]
        rng = Mock()
        rng.randint.return_value = 7
        actual = transpose_melodic_notes(tokens, rng)
        rng.randint.assert_called_once_with(-7, 7)
        self.assertEqual(actual[2], ("note", 67, 80))
        self.assertEqual(actual[6], ("note", 71, 80))
        self.assertEqual(actual[10], ("note", 36, 80))
        self.assertEqual(tokens[2], ("note", 60, 80))
        for index in set(range(len(tokens))) - {2, 6}:
            self.assertEqual(tokens[index], actual[index])
        bounds = [("instrument", "Acoustic Piano"), ("note", 0, 80), ("note", 127, 80)]
        self.assertEqual(transpose_melodic_notes(bounds, random.Random(42)), bounds)

    def test_training_augments_but_heldout_and_disabled_training_stay_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            write_records(path, tiny_records())
            audit_dataset([path], Path(directory) / "manifest", progress_every=0)
            datasets = [GenreDataset(Path(directory) / "manifest.json", split, min_length=8, max_length=32,
                                     eval_length=16, pitch_augmentation=enabled)
                        for split, enabled in (("train", True), ("train", False), ("validation", True), ("test", True))]
            try:
                def pitches(row):
                    return [token[1] for i in row["ids"].tolist()
                            if isinstance(token := self.tokenizer.id_to_tok[i], tuple) and token[0] == "note"]
                augmented = {p for draw in range(20) for p in pitches(datasets[0][(0, draw)])}
                self.assertTrue(any(p != 60 for p in augmented))
                for dataset in datasets[1:]:
                    self.assertEqual(set(pitches(dataset[0])), {60})
                torch.testing.assert_close(datasets[0][(0, 4)]["ids"], datasets[0][(0, 4)]["ids"])
            finally:
                for dataset in datasets:
                    dataset.close()

    def test_transposition_resolves_drums_before_single_token_crops(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = tiny_records()
            mix = ["<S>"] + [("instrument", "drum"), ("note", 36, 80), ("onset", 0), ("dur", 100)] * 10
            mix += [("instrument", "Acoustic Piano"), ("note", 60, 80), ("onset", 0), ("dur", 100)] * 10 + ["<E>"]
            for row in rows:
                row["tokens"] = mix
            path = Path(directory) / "data.jsonl"
            write_records(path, rows)
            audit_dataset([path], Path(directory) / "manifest", progress_every=0)
            data = GenreDataset(Path(directory) / "manifest.json", "train", min_length=1, max_length=1, eval_length=1)
            try:
                pitches = {self.tokenizer.id_to_tok[int(data[(0, draw)]["ids"][0])][1] for draw in range(100)}
                self.assertIn(36, pitches)
                self.assertTrue(any(p != 60 for p in pitches - {36}))
                self.assertLessEqual(pitches, {36} | set(range(53, 68)))
            finally:
                data.close()

    def test_scoring_strips_metadata_and_ignores_explicit_padding(self):
        metric = GenreClassifierMetric(self.model, self.tokenizer, self.labels)
        baseline = metric.probabilities(self.ids)
        self.assertAlmostEqual(sum(baseline.values()), 1., places=6)
        self.assertEqual(metric.score(self.ids), baseline["jazz"])
        leaked = [("prefix", "genre", "jazz")] + self.tokens
        self.assertEqual(metric.probabilities(leaked), baseline)
        padded = torch.cat((self.ids, torch.tensor([99999, 99999])))
        self.assertEqual(metric.probabilities(padded, valid_mask=[1] * len(self.ids) + [0, 0]), baseline)
        for sequence in (["<T>"], self.tokens + ["<MASK>"], self.tokens + ["<P>"]):
            with self.assertRaises(ValueError):
                metric.score(sequence)
        with self.assertRaises(ValueError):
            metric.score(self.ids, target_genre="unknown")

    def test_long_sequences_are_scored_not_truncated(self):
        metric = GenreClassifierMetric(self.model, self.tokenizer, self.labels)
        result = metric.probabilities(self.ids.repeat(6), window_batch_size=2)
        self.assertAlmostEqual(sum(result.values()), 1., places=6)
        self.assertTrue(all(0 <= p <= 1 for p in result.values()))

    def test_batched_scoring_keeps_owner_order_and_matches_individual_scores(self):
        metric = GenreClassifierMetric(self.model, self.tokenizer, self.labels)
        inputs = [self.ids, self.ids.repeat(6), self.ids[1:5]]
        actual = metric.probabilities_batch(inputs, window_batch_size=2)
        for row, sequence in zip(actual, inputs):
            expected = metric.probabilities(sequence)
            for label in self.labels:
                self.assertAlmostEqual(row[label], expected[label], places=6)
        self.assertEqual(metric.score_batch([]), [])
        scores = metric.score_batch(inputs)
        for score, row in zip(scores, actual):
            self.assertAlmostEqual(score, row["jazz"], places=6)
        with self.assertRaises(ValueError):
            metric.probabilities_batch(inputs, valid_masks=[])

    def test_metrics_and_weight_prior_correction(self):
        metrics = classification_metrics([[8, 2], [1, 9]], ["classical", "jazz"])
        self.assertEqual(metrics["accuracy"], .85)
        self.assertEqual(metrics["per_genre"]["jazz"]["support"], 10)
        self.assertGreater(metrics["macro_f1"], .8)
        weights = class_weights([100, 1], "sqrt")
        self.assertAlmostEqual(float(weights[1] / weights[0]), 10., places=5)
        natural = torch.tensor([.7, .3])
        weighted_logits = (natural * weights).log()
        torch.testing.assert_close((weighted_logits - weights.log()).softmax(0), natural)
        with self.assertRaises(ValueError):
            class_weights([100, 0])

    def test_group_metrics_keep_cross_group_errors_and_log_to_tensorboard(self):
        from unittest.mock import Mock
        # Rows=true genres, columns=predictions. Include cross-group errors.
        metrics = classification_metrics([[6, 2, 2], [1, 8, 1], [3, 2, 5]], ["jazz", "blues", "classical"])
        groups = metrics["groups"]
        self.assertAlmostEqual(groups["all"]["accuracy"], 19 / 30)
        self.assertAlmostEqual(groups["jazz_blues"]["accuracy"], 14 / 20)
        self.assertAlmostEqual(groups["jazz_blues"]["macro_f1"], (12 / 20 + 16 / 22) / 2)
        self.assertAlmostEqual(groups["remaining"]["accuracy"], .5)
        self.assertAlmostEqual(groups["remaining"]["macro_f1"], 10 / 18)
        self.assertAlmostEqual(metrics["per_genre"]["jazz"]["accuracy"], .6)
        writer = Mock()
        log_classification_metrics(writer, "validation", metrics, 100)
        tags = {call.args[0] for call in writer.add_scalar.call_args_list}
        self.assertIn("validation/groups/jazz_blues/accuracy", tags)
        self.assertIn("validation/groups/remaining/macro_f1", tags)
        self.assertIn("validation/accuracy/jazz", tags)
        self.assertIn("validation/f1/blues", tags)
        empty = classification_metrics([[3, 0], [0, 0]], ["classical", "jazz"])
        self.assertIsNone(empty["groups"]["jazz_blues"]["accuracy"])
        self.assertIsNone(empty["groups"]["jazz_blues"]["macro_f1"])
        writer.reset_mock()
        log_classification_metrics(writer, "train", empty, 1)
        self.assertFalse(any(call.args[1] is None for call in writer.add_scalar.call_args_list))

    def test_temperature_fits_inference_tensors_without_autograd_errors(self):
        with torch.inference_mode():
            logits = torch.tensor([[10., 0.]] * 8 + [[0., 10.]] * 2)
            targets = torch.zeros(10, dtype=torch.long)
        result = fit_temperature(logits, targets)
        self.assertGreater(result["temperature"], 1)
        self.assertLess(result["nll_after"], result["nll_before"])

    def test_checkpoint_mapping_vocabulary_and_temperature(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            payload = {"model_state_dict": self.model.state_dict(), "config": asdict(self.config),
                       "labels": self.labels, "trained_steps": 1, "temperature": 2.,
                       "vocabulary_digest": vocabulary_digest(self.tokenizer)}
            torch.save(payload, path)
            metric = GenreClassifierMetric.from_checkpoint(path)
            expected = (self.model(self.ids[None]).float() / 2).softmax(-1)[0].detach().tolist()
            for got, want in zip(metric.probabilities(self.ids).values(), expected):
                self.assertAlmostEqual(got, want, places=6)
            payload["trained_steps"] = 0
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                GenreClassifierMetric.from_checkpoint(path)

    def test_dataset_crops_are_deterministic_variable_and_contain_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            write_records(path, tiny_records())
            report = audit_dataset([path], Path(directory) / "manifest", progress_every=0)
            manifest = Path(directory) / "manifest.json"
            data = GenreDataset(manifest, "train", min_length=8, max_length=32, eval_length=16)
            validation = GenreDataset(manifest, "validation", min_length=8, max_length=32, eval_length=16)
            try:
                torch.testing.assert_close(data[0]["ids"], data[0]["ids"])
                lengths = set()
                for epoch in range(8):
                    data.set_epoch(epoch)
                    lengths.update(len(data[i]["ids"]) for i in range(len(data)))
                self.assertGreater(len(lengths), 1)
                validation.set_epoch(9)
                first = validation[0]
                validation.set_epoch(10)
                torch.testing.assert_close(first["ids"], validation[0]["ids"])
                self.assertEqual(len(first["ids"]), 16)
                with path.open("ab") as handle:
                    handle.write(b"\n")
                with self.assertRaises(ValueError):
                    load_manifest(manifest)
            finally:
                data.close()
                validation.close()

    def test_independent_training_loop_on_tiny_synthetic_cpu_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            write_records(path, tiny_records())
            audit_dataset([path], Path(directory) / "manifest", progress_every=0)
            from improvnet.metrics.genre_classifier import genre_classifier_config as defaults
            config = SimpleNamespace(**{key: value for key, value in vars(defaults).items() if key.isupper()})
            config.MANIFEST = str(Path(directory) / "manifest.json")
            config.SAVE_DIR = str(Path(directory) / "run")
            config.EMBED_DIM, config.N_LAYERS, config.N_HEADS, config.N_KV_HEADS = 16, 1, 2, 1
            config.MAX_LENGTH, config.DROPOUT = 32, 0.
            config.N_STEPS, config.BATCH_SIZE, config.VALIDATION_BATCH_SIZE = 2, 2, 2
            config.MIN_LENGTH, config.EVAL_LENGTH, config.NUM_WORKERS = 8, 16, 0
            config.WARMUP_STEPS, config.VALIDATE_EVERY, config.LOG_EVERY = 0, 1, 1
            config.DEVICE, config.TENSORBOARD, config.CALIBRATE = "cpu", False, True
            config.SAMPLES_PER_EPOCH = 20
            Path(config.SAVE_DIR).mkdir()
            (Path(config.SAVE_DIR) / "metrics.jsonl").write_text("interrupted debug log\n")
            (Path(config.SAVE_DIR) / "best_checkpoint.pt").write_bytes(b"orphaned checkpoint")
            train(config)
            self.assertEqual(len(list((Path(config.SAVE_DIR) / "previous_runs").iterdir())), 1)
            state = torch.load(Path(config.SAVE_DIR) / "calibrated_checkpoint.pt", map_location="cpu", weights_only=False)
            self.assertGreater(state["trained_steps"], 0)
            self.assertIn("macro_f1", state["validation"])
            self.assertIn("calibration", state)
            metric = GenreClassifierMetric.from_checkpoint(Path(config.SAVE_DIR) / "calibrated_checkpoint.pt")
            self.assertTrue(0 <= metric.score(self.ids) <= 1)

    def test_canonical_checkpoint_resume_replays_optimizer_and_sampling(self):
        from unittest.mock import patch
        from improvnet.metrics.genre_classifier import genre_classifier_config as defaults
        from improvnet.metrics.genre_classifier.genre_classifier_train import train as canonical_train
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            write_records(path, tiny_records())
            audit_dataset([path], Path(directory) / "manifest", progress_every=0)
            config = SimpleNamespace(**{key: value for key, value in vars(defaults).items() if key.isupper()})
            config.EMBED_DIM, config.N_HEADS, config.N_KV_HEADS, config.N_LAYERS = 16, 2, 1, 1
            config.MAX_LENGTH, config.DROPOUT = 32, .1
            config.MANIFEST, config.SAVE_DIR = str(Path(directory) / "manifest.json"), str(Path(directory) / "reference")
            config.DEVICE, config.NUM_WORKERS, config.TENSORBOARD, config.CALIBRATE = "cpu", 0, False, False
            config.MIN_LENGTH, config.EVAL_LENGTH = 8, 16
            config.BATCH_SIZE, config.ACCUM_STEPS, config.SAMPLES_PER_EPOCH = 2, 2, 14
            config.N_STEPS, config.WARMUP_STEPS, config.LOG_EVERY, config.VALIDATE_EVERY, config.SAVE_EVERY = 4, 0, 1, 1, 1
            canonical_train(config)
            reference = torch.load(Path(config.SAVE_DIR) / "latest_checkpoint.pt", weights_only=True)
            config.SAVE_DIR = str(Path(directory) / "resumed")
            request = Path(directory) / "request"
            request.touch()
            with patch.dict("os.environ", {"CHECKPOINT_REQUEST_FILE": str(request)}):
                canonical_train(config)  # Stop at step 1, after accumulation.
            saved = torch.load(Path(config.SAVE_DIR) / "latest_checkpoint.pt", weights_only=True)
            self.assertEqual(saved["trained_steps"], 1)
            # Simulate the pre-refactor settings layout; model config/weights are unchanged.
            for key in ("EMBED_DIM", "N_HEADS", "N_KV_HEADS", "N_LAYERS", "FFN_MULT", "DROPOUT", "GRADIENT_CHECKPOINTING", "MAX_LENGTH"):
                saved["train_config"].pop(key)
            torch.save(saved, Path(config.SAVE_DIR) / "latest_checkpoint.pt")
            canonical_train(config)
            resumed = torch.load(Path(config.SAVE_DIR) / "latest_checkpoint.pt", weights_only=True)
            self.assertEqual(resumed["trained_steps"], 4)
            self.assertEqual((resumed["epoch"], resumed["next_batch"]), (reference["epoch"], reference["next_batch"]))
            for key in reference["model_state_dict"]:
                torch.testing.assert_close(resumed["model_state_dict"][key], reference["model_state_dict"][key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
