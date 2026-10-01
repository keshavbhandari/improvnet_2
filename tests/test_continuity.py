"""Synthetic CPU continuity checks; no production training/checkpoint reads."""

from dataclasses import asdict
import json
import math
from pathlib import Path
import pickle
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from improvnet.tokenizer.absolute import AbsTokenizer
from improvnet.metrics.genre_classifier.audit import piece_split
from improvnet.metrics.continuity import continuity_config as defaults
from improvnet.metrics.continuity.audit import audit_dataset
from improvnet.metrics.continuity.continuity_model import ContinuityConfig, ContinuityEncoder
from improvnet.metrics.continuity.continuity_train import evaluate, settings_dict, train, validate_config
from improvnet.metrics.continuity.data import ContinuityDataset, PairBatchSampler, PairCollator, augment_excerpt, load_manifest, sample_pair
from improvnet.metrics.continuity.objective import contrastive_loss, summarize
from improvnet.metrics.continuity.scoring import ContinuityMetric


def records():
    rows = []
    for split in ("train", "validation", "test"):
        index, count = 0, 0
        while count < 5:
            identity = f"{split}-{index}"
            index += 1
            if piece_split("piece_id:" + identity) != split:
                continue
            tokens = ["<S>"] + [["instrument", "Acoustic Piano"], ["note", 60, 80], ["onset", 0], ["dur", 100]] * 160 + ["<E>"]
            rows.append({"piece_id": identity, "genre": None if count % 2 else "jazz", "midi_filepath": f"/music/{identity}.mid", "tokens": tokens})
            count += 1
    return rows


def make_manifest(directory):
    path = Path(directory) / "music.jsonl"
    offsets = []
    with path.open("wb") as handle:
        for row in records():
            offsets.append(handle.tell())
            handle.write((json.dumps(row) + "\n").encode())
    with path.with_name(path.stem + "_index.pkl").open("wb") as handle:
        pickle.dump(offsets, handle)
    audit_dataset([path], Path(directory) / "manifest", progress_every=0)
    return Path(directory) / "manifest.json"


def tiny_config(directory):
    config = SimpleNamespace(**settings_dict(defaults))
    config.MANIFEST, config.SAVE_DIR = str(Path(directory) / "manifest.json"), str(Path(directory) / "run")
    config.EMBED_DIM, config.N_HEADS, config.N_KV_HEADS, config.N_LAYERS, config.PROJECTION_DIM = 16, 2, 1, 1, 8
    config.MIN_LENGTH, config.MAX_LENGTH, config.EVAL_LENGTH = 8, 32, 16
    config.BATCH_SIZE, config.ACCUM_STEPS, config.SAMPLES_PER_EPOCH = 2, 2, 8
    config.VALIDATION_BATCH_SIZE = 2
    config.N_STEPS, config.WARMUP_STEPS, config.LOG_EVERY, config.VALIDATE_EVERY, config.SAVE_EVERY = 2, 0, 1, 1, 1
    config.DEVICE, config.NUM_WORKERS, config.TENSORBOARD = "cpu", 0, False
    return config


class ContinuityTests(unittest.TestCase):
    def test_excerpt_augmentation_bounds_and_drums_crossing_crop_boundaries(self):
        from unittest.mock import Mock
        tokens = [("instrument", "Acoustic Piano"), ("note", 0, 80), ("note", 127, 80),
                  ("note", 60, 80), ("note", 64, 80), ("instrument", "drum"), ("note", 36, 80)]
        rng = Mock()
        rng.randint.return_value = 3
        excerpt = augment_excerpt(tokens, 3, 5, rng)
        rng.randint.assert_called_once_with(-7, 7)
        self.assertEqual(excerpt, [("note", 63, 80), ("note", 67, 80)])
        self.assertEqual(augment_excerpt(tokens, 6, 7, random.Random(0)), [("note", 36, 80)])
        self.assertEqual(augment_excerpt(tokens, 1, 3, random.Random(0)), tokens[1:3])
        self.assertEqual(tokens[3][1], 60)

    def setUp(self):
        torch.manual_seed(13)
        self.tokenizer = AbsTokenizer()
        self.config = ContinuityConfig(vocab_size=self.tokenizer.vocab_size, pad_id=self.tokenizer.pad_id,
                                       embed_dim=16, n_heads=2, n_kv_heads=1, n_layers=1, projection_dim=8, dropout=0., max_length=512)
        self.model = ContinuityEncoder(self.config).eval()
        self.tokens = ["<S>", ("instrument", "Acoustic Piano"), ("note", 60, 80), ("onset", 0), ("dur", 100), "<E>"]
        self.ids = torch.tensor([self.tokenizer.tok_to_id[t] for t in self.tokens])

    def test_default_backbone_settings_and_values_only_config(self):
        import ast
        config = ContinuityConfig()
        self.assertEqual((config.embed_dim, config.n_layers, config.n_heads, config.n_kv_heads, config.max_length), (2048, 20, 16, 4, 512))
        self.assertEqual((defaults.MIN_LENGTH, defaults.EVAL_LENGTH), (128, 256))
        self.assertFalse(any(isinstance(node, (ast.ClassDef, ast.FunctionDef)) for node in ast.walk(ast.parse(Path(defaults.__file__).read_text()))))
        self.assertEqual(ContinuityConfig(**asdict(self.config)), self.config)
        with torch.device("meta"):
            model = ContinuityEncoder(config)
        self.assertEqual(model.layers[0].ffn.w1.out_features, 5504)
        self.assertFalse(any("genre" in name or "condition" in name for name, _ in model.named_parameters()))

    def test_nonoverlap_variable_lengths_and_validation_lengths(self):
        tokens = [("note", 60, 80)] * 2048
        lengths = set()
        for seed in range(40):
            a, b = sample_pair(tokens, random.Random(seed), 128, 512)
            self.assertLessEqual(a[1], b[0])
            lengths.update((a[1] - a[0], b[1] - b[0]))
            self.assertTrue(all(128 <= length <= 512 for length in (a[1] - a[0], b[1] - b[0])))
        self.assertGreater(len(lengths), 20)
        a, b = sample_pair(tokens, random.Random(42), 128, 512, 256)
        self.assertEqual((a[1] - a[0], b[1] - b[0]), (256, 256))
        self.assertIsNone(sample_pair(tokens[:200], random.Random(0), 128, 512))
        self.assertIsNone(sample_pair(["<T>"] * 2048, random.Random(0), 128, 512))

    def test_audit_includes_unknowns_and_guards_sources_and_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = make_manifest(directory)
            report = load_manifest(manifest)
            self.assertEqual(report["records"], 15)
            self.assertEqual(report["split_counts"], {"train": 5, "validation": 5, "test": 5})
            self.assertNotIn("labels", report)
            data = [ContinuityDataset(manifest, split) for split in ("train", "validation", "test")]
            for dataset in data:
                self.assertEqual(len(dataset), 5)
                dataset.close()
            with Path(report["sources"][0]["path"]).open("ab") as handle:
                handle.write(b"\n")
            with self.assertRaises(ValueError):
                load_manifest(manifest)

    def test_positive_views_use_independent_augmentation_and_validation_is_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = make_manifest(directory)
            data = ContinuityDataset(manifest, "train")
            pitches = []
            for draw in range(20):
                item = data[(0, draw)]
                self.assertLessEqual(item["ranges"][0][1], item["ranges"][1][0])
                pair = []
                for view in ("a", "b"):
                    notes = [self.tokenizer.id_to_tok[int(i)][1] for i in item[view] if isinstance(self.tokenizer.id_to_tok[int(i)], tuple) and self.tokenizer.id_to_tok[int(i)][0] == "note"]
                    pair.append(notes[0])
                pitches.append(pair)
            self.assertTrue(any(a != b for a, b in pitches))
            self.assertEqual(data[(0, 3)]["a"].tolist(), data[(0, 3)]["a"].tolist())
            data.close()
            val = ContinuityDataset(manifest, "validation")
            with patch("improvnet.metrics.continuity.data.augment_excerpt", side_effect=AssertionError("validation augmentation")):
                first = val[0]
                val.set_epoch(9)
                second = val[0]
            self.assertEqual((len(first["a"]), len(first["b"])), (256, 256))
            self.assertEqual(first["a"].tolist(), second["a"].tolist())
            val.close()

    def test_sampler_resume_and_ddp_shards_are_disjoint(self):
        a = PairBatchSampler(40, 24, 3, seed=3, rank=0, world_size=2)
        b = PairBatchSampler(40, 24, 3, seed=3, rank=1, world_size=2)
        complete = list(a)
        for left, right in zip(complete, b):
            self.assertFalse({i for i, _ in left} & {i for i, _ in right})
        a.start_batch = 2
        self.assertEqual(list(a), complete[2:])

    def test_padding_unit_norm_bidirectional_context_and_backward(self):
        batch = PairCollator(self.tokenizer.pad_id)([{"a": self.ids[:4], "b": self.ids, "piece_id": 0}, {"a": self.ids, "b": self.ids[:5], "piece_id": 1}])
        a, b = self.model(batch["a"], batch["a_mask"], batch["b"], batch["b_mask"])
        torch.testing.assert_close(a[0], self.model(self.ids[:4][None])[0], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(a.norm(dim=-1), torch.ones(2))
        self.model.train()
        a, b = self.model(batch["a"], batch["a_mask"], batch["b"], batch["b_mask"])
        loss, stats = contrastive_loss(a, b, batch["piece_ids"])
        loss.backward()
        self.assertTrue(torch.isfinite(self.model.token_emb.weight.grad).all())
        self.assertIn("hard_negative_margin", summarize(stats))
        with self.assertRaises(ValueError):
            self.model(torch.full((1, 4), self.tokenizer.pad_id))

    def test_hand_calculated_loss_and_duplicate_piece_negatives(self):
        a = torch.eye(2, requires_grad=True)
        b = torch.eye(2, requires_grad=True)
        loss, stats = contrastive_loss(a, b, torch.tensor([0, 1]), temperature=.5)
        self.assertAlmostEqual(float(loss.detach()), math.log1p(math.exp(-2)), places=6)
        values = summarize(stats)
        self.assertEqual(values["retrieval_accuracy"], 1)
        self.assertEqual(values["positive_similarity"], 1)
        self.assertEqual(values["negative_similarity"], 0)
        loss.backward()
        self.assertIsNotNone(a.grad)
        duplicated = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        loss, stats = contrastive_loss(duplicated, duplicated, torch.tensor([0, 0, 1]), temperature=.5)
        self.assertAlmostEqual(float(loss), (2 * math.log1p(math.exp(-2)) + math.log1p(2 * math.exp(-2))) / 3, places=6)
        self.assertEqual(summarize(stats)["retrieval_accuracy"], 1)
        with self.assertRaises(ValueError):
            contrastive_loss(a, b, torch.tensor([0, 0]))

    def test_scoring_symmetry_checkpoint_guard_and_invalid_music(self):
        metric = ContinuityMetric(self.model, self.tokenizer)
        self.assertAlmostEqual(metric.score(self.ids, self.ids), 1., places=5)
        self.assertAlmostEqual(metric.score(self.ids, self.ids[:4]), metric.score(self.ids[:4], self.ids), places=6)
        self.assertTrue(-1 <= metric.score_boundary(self.ids, self.ids, window_size=4) <= 1)
        with self.assertRaises(ValueError):
            metric.score(["<T>", "<E>"], self.ids)
        with self.assertRaises(ValueError):
            metric.score([self.tokenizer.tok_to_id[self.tokenizer.mask_tok]], self.ids)
        with self.assertRaises(ValueError):
            metric.score([("note", 60., 80)], self.ids)
        with self.assertRaises(ValueError):
            metric.score(self.ids[None], self.ids)
        padded = torch.cat((self.ids, torch.tensor([self.tokenizer.pad_id])))
        self.assertAlmostEqual(metric.score(padded, self.ids, previous_mask=[True] * len(self.ids) + [False]), 1., places=5)

    def test_training_resume_exact_weights_and_trained_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            make_manifest(directory)
            config = tiny_config(directory)
            train(config)
            reference = torch.load(Path(config.SAVE_DIR) / "latest_checkpoint.pt", weights_only=True)
            metric = ContinuityMetric.from_checkpoint(Path(config.SAVE_DIR) / "best_checkpoint.pt")
            self.assertAlmostEqual(metric.score(self.ids, self.ids), 1., places=5)
            config.SAVE_DIR = str(Path(directory) / "resume")
            request = Path(directory) / "request"
            request.touch()
            with patch.dict("os.environ", {"CHECKPOINT_REQUEST_FILE": str(request)}):
                train(config)
            self.assertEqual(torch.load(Path(config.SAVE_DIR) / "latest_checkpoint.pt", weights_only=True)["trained_steps"], 1)
            train(config)
            resumed = torch.load(Path(config.SAVE_DIR) / "latest_checkpoint.pt", weights_only=True)
            for key in reference["model_state_dict"]:
                torch.testing.assert_close(resumed["model_state_dict"][key], reference["model_state_dict"][key], atol=0, rtol=0)
            reference["trained_steps"] = 0
            torch.save(reference, Path(directory) / "untrained.pt")
            with self.assertRaises(ValueError):
                ContinuityMetric.from_checkpoint(Path(directory) / "untrained.pt")
            config.MAX_LENGTH = 64
            with self.assertRaises(ValueError):
                train(config)

    def test_invalid_config_and_missing_manifest(self):
        config = SimpleNamespace(**settings_dict(defaults))
        validate_config(config)
        for key, value in (("BATCH_SIZE", 1), ("TEMPERATURE", 0), ("MIN_LENGTH", 513)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(SimpleNamespace(**{**vars(config), key: value}))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "continuity.audit"):
                load_manifest(Path(directory) / "missing.json")


if __name__ == "__main__":
    unittest.main()
