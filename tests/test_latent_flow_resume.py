"""Resume-policy tests only; no training or production-checkpoint writes."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from improvnet.model.latent_flow_train import main, resume_setting_changes


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


if __name__ == "__main__":
    unittest.main()
