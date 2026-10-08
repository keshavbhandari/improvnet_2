"""Exact histogram counts, joint gating, recall safeguards and checkpoint defaults."""
from types import SimpleNamespace
import unittest

import torch

from improvnet.inference.local_vae_inference import resolve_thresholds
from improvnet.utils.local_vae_calibration import ThresholdCalibrator


def example(activity, onset, target_active, target_onset, valid=None):
    a, o = [torch.tensor(v, dtype=torch.float32).reshape(1, 1, -1) for v in (activity, onset)]
    targets = torch.stack([torch.tensor(v, dtype=torch.float32).reshape(1, -1)
                           for v in (target_active, target_onset)])[None]
    return SimpleNamespace(activity_logits=torch.logit(a), onset_logits=torch.logit(o)), dict(
        roll=targets, valid_mask=torch.tensor([valid or [True] * len(activity)]))


class CalibrationTests(unittest.TestCase):
    def test_joint_selection_gates_false_onsets_and_meets_recall_floor(self):
        r, batch = example([.95, .6, .6, .95], [.99, .95, .95, .1], [1, 0, 0, 1], [1, 0, 0, 0])
        c = ThresholdCalibrator([.5, .8], [.8, .98], defaults=(.5, .8), branches=("clean",))
        c.update(r, batch)
        selected = c.select(distributed=False)
        self.assertEqual(selected["activity_threshold"], .8)
        self.assertTrue(selected["recall_floor_met"])
        self.assertEqual(selected["scores"]["clean"]["onset"]["f1"], 1.)
        self.assertEqual(selected["scores"]["clean"]["activity"]["f1"], 1.)

    def test_recall_floor_prevents_precision_only_solution(self):
        r, batch = example([.95, .6, .55], [.99, .99, .99], [1, 1, 0], [1, 1, 0])
        c = ThresholdCalibrator([.5, .8], [.5], defaults=(.5, .5), branches=("clean",), min_recall=.99)
        c.update(r, batch)
        selected = c.select(distributed=False)
        self.assertEqual(selected["activity_threshold"], .5)
        self.assertEqual(selected["scores"]["clean"]["onset"]["recall"], 1.)

    def test_unattainable_recall_floor_is_explicit(self):
        r, batch = example([.1, .95], [.1, .99], [1, 0], [1, 0])
        c = ThresholdCalibrator([.5], [.5], defaults=(.5, .5), branches=("clean",))
        c.update(r, batch)
        selected = c.select(distributed=False)
        self.assertFalse(selected["recall_floor_met"])
        self.assertEqual(selected["status"], "recall_floor_unattainable")

    def test_padding_and_silent_validation_cannot_certify_perfect_reconstruction(self):
        r, batch = example([.1, .99], [.1, .99], [0, 1], [0, 1], valid=[True, False])
        r.activity_logits[..., 1] = float("nan")
        c = ThresholdCalibrator([.5], [.8], defaults=(.5, .8), branches=("clean",))
        c.update(r, batch)
        selected = c.select(distributed=False)
        self.assertEqual(int(c.counts.sum()), 1)
        self.assertEqual(selected["status"], "insufficient_positive_targets")
        self.assertFalse(selected["recall_floor_met"])

    def test_histograms_equal_brute_force_for_every_threshold_pair_and_branch(self):
        # Powers of two yield exact sigmoid/logit threshold-equality checks.
        thresholds = [.25, .5, .75]
        c = ThresholdCalibrator(thresholds, thresholds, defaults=(.5, .5))
        r, batch = example([.25, .5, .75, .9, .1], [.75, .5, .25, .9, .1],
                           [1, 1, 1, 0, 0], [1, 0, 1, 0, 0])
        c.update(r, batch, "clean")
        c.update(r, batch, "jitter")
        hist = c.counts.flip((-2, -1)).cumsum(-2).cumsum(-1).flip((-2, -1))
        target = batch["roll"][:, 1].bool()
        for b in range(2):
            for i, a in enumerate(thresholds):
                for j, o in enumerate(thresholds):
                    predicted = (r.activity_logits.sigmoid() >= a) & (r.onset_logits.sigmoid() >= o)
                    self.assertEqual(int(hist[b, 1, i + 1, j + 1]), int((predicted & target).sum()))
                    self.assertEqual(int(hist[b, 0, i + 1, j + 1]), int((predicted & ~target).sum()))

    def test_sharded_histograms_match_single_batch_selection(self):
        whole = ThresholdCalibrator([.5, .8], [.8, .98])
        shards = [ThresholdCalibrator([.5, .8], [.8, .98]) for _ in range(2)]
        for i in range(2):
            r, batch = example([.95, .6], [.99, .95], [1, 0], [1, 0])
            for branch in ("clean", "jitter"):
                whole.update(r, batch, branch)
                shards[i].update(r, batch, branch)
        shards[0].counts += shards[1].counts
        shards[0].active_counts += shards[1].active_counts
        self.assertEqual(whole.select(distributed=False), shards[0].select(distributed=False))

    def test_checkpoint_calibration_then_cli_override_and_legacy_fallback(self):
        model = SimpleNamespace(decoder_calibration=dict(activity_threshold=.8, onset_threshold=.995))
        args = SimpleNamespace(activity_threshold=None, onset_threshold=None)
        resolve_thresholds(args, model)
        self.assertEqual((args.activity_threshold, args.onset_threshold), (.8, .995))
        args = SimpleNamespace(activity_threshold=.6, onset_threshold=None)
        resolve_thresholds(args, model)
        self.assertEqual((args.activity_threshold, args.onset_threshold), (.6, .995))
        args = SimpleNamespace(activity_threshold=None, onset_threshold=None)
        resolve_thresholds(args, SimpleNamespace())
        self.assertEqual((args.activity_threshold, args.onset_threshold), (.5, .99))
        with self.assertRaisesRegex(ValueError, "finite"):
            resolve_thresholds(SimpleNamespace(activity_threshold=float("nan"), onset_threshold=None), model)

    def test_invalid_configuration_and_nonfinite_predictions(self):
        for kwargs in (dict(activity=[]), dict(onset=[1.]), dict(min_recall=1.1), dict(branches=())):
            settings = dict(activity=[.5], onset=[.9])
            settings.update(kwargs)
            with self.assertRaises(ValueError):
                ThresholdCalibrator(**settings)
        r, batch = example([.5], [.9], [1], [1])
        r.activity_logits[:] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            ThresholdCalibrator([.5], [.9], branches=("clean",)).update(r, batch)


if __name__ == "__main__":
    unittest.main()
