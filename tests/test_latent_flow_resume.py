"""Resume-policy tests only; no training or production-checkpoint writes."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from improvnet.model.latent_flow_train import loader_options, main, resume_setting_changes, validate_config


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(BATCH_SIZE=8, LR=.0001, TARGET_BINS=8, NUM_WORKERS=0)
        self.model = {"d_model": 32, "target_bins": 8}
        self.state = dict(train_config=vars(self.config).copy(), world_size=1, model_config=self.model.copy())

    def test_exact_resume_is_unchanged(self):
        self.assertEqual(resume_setting_changes(self.state, self.config, self.model, 1), [])

    def test_batch_change_requires_explicit_flag(self):
        self.config.BATCH_SIZE = 16
        with self.assertRaisesRegex(ValueError, "allow-batch-size-change"):
            resume_setting_changes(self.state, self.config, self.model, 1)
        self.assertEqual(resume_setting_changes(self.state, self.config, self.model, 1,
                         allow_batch_size_change=True), ["BATCH_SIZE"])
        self.assertEqual(self.state["train_config"]["BATCH_SIZE"], 8)

    def test_flag_does_not_allow_other_settings(self):
        for field, value in (("LR", .001), ("TARGET_BINS", 16)):
            config = deepcopy(self.config)
            config.BATCH_SIZE = 16
            setattr(config, field, value)
            with self.assertRaises(ValueError):
                resume_setting_changes(self.state, config, self.model, 1, allow_batch_size_change=True)

    def test_world_and_architecture_stay_protected(self):
        for model, world in ((self.model, 2), ({"d_model": 64, "target_bins": 8}, 1)):
            with self.assertRaises(ValueError):
                resume_setting_changes(self.state, self.config, model, world, allow_batch_size_change=True)

    def test_cli_passes_flag_without_starting_training(self):
        with patch("improvnet.model.latent_flow_train.train") as train:
            main(["--batch-size", "16", "--allow-batch-size-change", "--cpu"])
            self.assertEqual(train.call_args.args[0].BATCH_SIZE, 16)
            self.assertTrue(train.call_args.kwargs["allow_batch_size_change"])

    def test_memory_settings_preserve_old_checkpoint_resume(self):
        self.config.NUM_WORKERS = 2
        self.config.PREFETCH_FACTOR = 1
        self.assertEqual(resume_setting_changes(self.state, self.config, self.model, 1), [])

    def test_prefetch_is_only_supplied_to_worker_loaders(self):
        options = loader_options(self.config, torch.device("cpu"))
        self.assertNotIn("prefetch_factor", options)
        self.config.NUM_WORKERS = 2
        self.assertEqual(loader_options(self.config, torch.device("cuda"))["prefetch_factor"], 1)
        self.config.PREFETCH_FACTOR = 3
        self.assertEqual(loader_options(self.config, torch.device("cpu"))["prefetch_factor"], 3)

    def test_invalid_prefetch_is_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "PREFETCH_FACTOR"):
                validate_config(SimpleNamespace(PREFETCH_FACTOR=value))

    def test_cli_loader_controls(self):
        with patch("improvnet.model.latent_flow_train.train") as train:
            main(["--num-workers", "0", "--prefetch-factor", "1", "--cpu"])
            self.assertEqual(train.call_args.args[0].NUM_WORKERS, 0)
            self.assertEqual(train.call_args.args[0].PREFETCH_FACTOR, 1)


if __name__ == "__main__":
    unittest.main()
