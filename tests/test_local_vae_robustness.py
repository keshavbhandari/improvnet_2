"""Frozen decoder perturbation: scale, padding, paired noise and finite reports."""
from contextlib import redirect_stdout
import io
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from improvnet.inference.local_vae_inference import LatentStats, Scores
from improvnet.inference.local_vae_robustness import channel_scale, perturb, sweep, average_scores, plot_report
from improvnet.model.local_vae_model import LocalVAE, LocalVAEConfig, Posterior
from improvnet.tokenizer.hierarchical import GridConfig


class RobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_noise_scale_paired_direction_and_padding(self):
        mean = torch.ones(1, 2, 1, 3)
        epsilon = torch.tensor([[[[1., -1., 9.]], [[2., -2., 9.]]]])
        scale = torch.tensor([2., 4.])
        weights = torch.tensor([[[[1., .5, 0.]]]])
        baseline = perturb(mean, scale, epsilon, 0., weights)
        a = perturb(mean, scale, epsilon, .05, weights)
        b = perturb(mean, scale, epsilon, .1, weights)
        torch.testing.assert_close(a[0, :, 0, 0], torch.tensor([1.1, 1.4]))
        torch.testing.assert_close(b - baseline, 2 * (a - baseline))
        self.assertFalse(a[..., 2].any())
        torch.testing.assert_close(baseline[..., :2], mean[..., :2])

    def test_channel_scale_uses_valid_fraction_unrounded(self):
        mean = torch.tensor([[[[1., 3., 999.]]]])
        p = Posterior(mean, torch.zeros_like(mean), torch.tensor([[[[1., .5, 0.]]]]))
        stats = LatentStats(); stats.update(p)
        torch.testing.assert_close(channel_scale(stats), torch.tensor([np.sqrt(8 / 9)], dtype=torch.float32))

    def test_invalid_noise_rejected(self):
        mean = torch.zeros(1, 2, 1, 2)
        for alpha in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError): perturb(mean, torch.ones(2), mean, alpha, torch.ones(1, 1, 1, 2))
        with self.assertRaises(ValueError): perturb(mean, torch.ones(3), mean, .1, torch.ones(1, 1, 1, 2))
        with self.assertRaises(ValueError): channel_scale(LatentStats())

    def test_sweep_seeded_no_weight_updates_and_baseline_draws_equal(self):
        model = LocalVAE(LocalVAEConfig(base_channels=4, channel_multipliers=(1, 2), latent_channels=2, res_blocks=1)).eval()
        state = {k: v.clone() for k, v in model.state_dict().items()}
        reference = SimpleNamespace(config=GridConfig(), metadata={"pedal_enabled": False},
            velocity=np.zeros((2, 128, 100), np.uint8), onset=np.zeros((2, 128, 100), np.uint8),
            pedal=np.zeros((2, 100), np.uint8), valid_mask=np.ones((2, 100), bool))
        reference.velocity[0, 60, 20:] = 80; reference.velocity[1, 60, :20] = 80
        reference.onset[0, 60, 20] = 1
        reference.valid_mask[1, 21:] = False
        with torch.inference_mode():
            p = model.encode(torch.tensor(np.stack((reference.velocity / 127., reference.onset), axis=1)).float(),
                torch.zeros(2, 1, 100), torch.from_numpy(reference.valid_mask))
        windows = [dict(reference=reference, posterior=p, kind="pitched", row_id=1)]
        with redirect_stdout(io.StringIO()):
            a, _ = sweep(model, windows, torch.ones(2), [0., .1], 2, 42, 2, .5, .99)
            b, _ = sweep(model, windows, torch.ones(2), [0., .1], 2, 42, 2, .5, .99)
        self.assertEqual(a[0.][0].counts, a[0.][1].counts)
        self.assertEqual(a[.1][0].counts, b[.1][0].counts)
        for key, tensor in model.state_dict().items(): torch.testing.assert_close(tensor, state[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_empty_metrics_null(self):
        summary = average_scores([Scores(), Scores()])
        self.assertIsNone(summary["notes_f1"])

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Optional plotting dependency unavailable")
    def test_plot_saved(self):
        summary = average_scores([Scores(), Scores()])
        row = dict(alpha=0., **summary)
        with tempfile.TemporaryDirectory() as tmp:
            plot_report(dict(curve=[row, dict(alpha=.1, **summary)], posterior_control=summary,
                validation_files=2, thresholds={"onset": .99},
                by_target={"piano": [row, dict(alpha=.1, **summary)]}, target_files={"piano": 2}), Path(tmp))
            self.assertTrue((Path(tmp) / "robustness.png").exists())
            self.assertTrue((Path(tmp) / "by_instrument.png").exists())


if __name__ == "__main__":
    unittest.main()
