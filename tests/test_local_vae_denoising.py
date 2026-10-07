"""Event corruption, latent noise, loss balance and deterministic stress tests."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import random
import unittest

import numpy as np
import torch

from improvnet.model import local_vae_config as defaults
from improvnet.model.local_vae_model import LocalVAE, LocalVAEConfig, Posterior, balanced_bce
from improvnet.model.local_vae_train import batch_objective, evaluate, robustness_ramp, summarize, validate_config
from improvnet.tokenizer.hierarchical import EncodedPiece, GridConfig
from improvnet.utils.local_vae_corruption import corrupt_roll_bin, jitter_latent, latent_channel_std, target_note_rows
from improvnet.utils.local_vae_data import bin_tensors, collate_bins, transpose_bin


def config(**changes):
    values = {k: v for k, v in vars(defaults).items() if k.isupper()}
    values.update(changes)
    return SimpleNamespace(**values)


def piece(notes, *, n_frames=200, drum=False):
    """Notes are (pitch, velocity, absolute onset frame, absolute end frame)."""
    bins = (n_frames + 99) // 100
    velocity, onset = np.zeros((bins, 128, 100), np.uint8), np.zeros((bins, 128, 100), np.uint8)
    for pitch, value, start, end in notes:
        for frame in range(start, end):
            velocity[frame // 100, pitch, frame % 100] = value
        onset[start // 100, pitch, start % 100] = 1
    mask = np.arange(bins * 100).reshape(bins, 100) < n_frames
    pedal = np.full((bins, 100), 73 if not drum else 0, np.uint8) * mask
    return EncodedPiece(GridConfig(), velocity, onset, pedal, mask, velocity.max(-1),
        onset.sum(-1, dtype=np.int32), pedal.max(-1), np.zeros((0, 11), np.int64), np.zeros(bins + 1, np.int64),
        dict(target_track=0, target_channel=9 if drum else 0, target_program=0,
             target_is_drum=drum, target_present=True, pedal_enabled=not drum)).validate()


class ScriptedRNG:
    def __init__(self, operation, amount=1, selected=(0,)):
        self.operation, self.amount, self.selected, self.index = operation, amount, set(selected), -1

    def random(self):
        self.index += 1
        return 0. if self.index in self.selected else 1.

    def choices(self, choices, weights):
        return [self.operation]

    def choice(self, choices):
        if self.amount not in choices:
            raise AssertionError(f"Bad scripted choice {self.amount} for {choices}")
        return self.amount


def batch():
    source = piece([(60, 80, 95, 110), (64, 90, 120, 130)], n_frames=133)
    items = [bin_tensors(source, i) for i in (0, 1)]
    for i, item in enumerate(items):
        item["corrupted_roll"] = corrupt_roll_bin(source, i, config(), random.Random(12))
        item["retries"] = torch.tensor(0)
    return collate_bins(items)


class CorruptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_probability_exactly_preserves_chords_retriggers_carries_and_padding(self):
        source = piece([(60, 80, 90, 100), (60, 80, 100, 120), (64, 90, 85, 125),
                        (72, 50, 121, 122)], n_frames=133)
        self.assertEqual(len(target_note_rows(source)), 4)
        before = deepcopy(source)
        for i in (0, 1):
            result = corrupt_roll_bin(source, i, config(SEMANTIC_NOTE_PROB=0.), ScriptedRNG("addition"))
            torch.testing.assert_close(result, bin_tensors(source, i)["roll"], rtol=0, atol=0)
        np.testing.assert_array_equal(source.velocity, before.velocity)
        np.testing.assert_array_equal(source.onset, before.onset)
        np.testing.assert_array_equal(source.pedal, before.pedal)

    def test_timing_moves_onset_velocity_and_release_as_a_single_event(self):
        source = piece([(60, 81, 20, 35)])
        result = corrupt_roll_bin(source, 0, config(), ScriptedRNG("timing", 3))
        self.assertEqual(result[1, 60].nonzero().flatten().tolist(), [23])
        self.assertEqual(result[0, 60].nonzero().flatten().tolist(), list(range(23, 38)))
        self.assertAlmostEqual(result[0, 60, 23].item(), 81 / 127)

    def test_shifted_carry_has_no_fabricated_boundary_onset(self):
        source = piece([(60, 80, 95, 110)])
        result = corrupt_roll_bin(source, 1, config(), ScriptedRNG("timing", -2))
        self.assertFalse(result[1].any())
        self.assertEqual(result[0, 60].nonzero().flatten().tolist(), list(range(8)))

    def test_neighbouring_attack_can_move_into_the_bin_on_the_shared_clock(self):
        source = piece([(60, 80, 101, 110)])
        result = corrupt_roll_bin(source, 0, config(), ScriptedRNG("timing", -2))
        self.assertEqual(result[1, 60].nonzero().flatten().tolist(), [99])
        self.assertTrue(result[0, 60, 99] > 0)

    def test_crossing_into_current_bin_is_a_true_retrigger_not_a_carry(self):
        source = piece([(60, 80, 99, 110)])
        result = corrupt_roll_bin(source, 1, config(), ScriptedRNG("timing", 2))
        self.assertEqual(result[1, 60].nonzero().flatten().tolist(), [1])
        self.assertEqual(result[0, 60].nonzero().flatten().tolist(), list(range(1, 12)))

    def test_shift_cannot_erase_only_local_evidence_or_leave_piece_bounds(self):
        for source, amount in ((piece([(60, 80, 99, 110)]), 2), (piece([(60, 80, 0, 10)]), -1),
                               (piece([(60, 80, 120, 133)], n_frames=133), 1)):
            i = 1 if source.n_frames == 133 else 0
            result = corrupt_roll_bin(source, i, config(), ScriptedRNG("timing", amount))
            torch.testing.assert_close(result, bin_tensors(source, i)["roll"])

    def test_same_pitch_overlap_proposals_are_rejected(self):
        source = piece([(60, 80, 10, 20), (60, 90, 22, 30)])
        result = corrupt_roll_bin(source, 0, config(), ScriptedRNG("timing", 3))
        torch.testing.assert_close(result, bin_tensors(source, 0)["roll"])
        source = piece([(60, 80, 10, 30), (61, 90, 15, 40)])
        result = corrupt_roll_bin(source, 0, config(), ScriptedRNG("addition", 1))
        torch.testing.assert_close(result, bin_tensors(source, 0)["roll"])

    def test_pitch_shift_moves_both_channels_but_does_not_change_clean_target(self):
        source = piece([(60, 81, 20, 35), (65, 70, 30, 40)])
        clean = bin_tensors(source, 0)["roll"]
        result = corrupt_roll_bin(source, 0, config(), ScriptedRNG("pitch", 1))
        self.assertFalse(result[:, 60].any())
        torch.testing.assert_close(result[:, 61], clean[:, 60])
        torch.testing.assert_close(bin_tensors(source, 0)["roll"], clean)

    def test_isolated_notes_and_carries_are_not_pitch_shifted_or_deleted(self):
        for source, i in ((piece([(60, 80, 20, 35)]), 0), (piece([(60, 80, 90, 110)]), 1)):
            for operation in ("pitch", "deletion"):
                result = corrupt_roll_bin(source, i, config(), ScriptedRNG(operation))
                torch.testing.assert_close(result, bin_tensors(source, i)["roll"])

    def test_addition_retains_original_complete_note(self):
        source = piece([(60, 80, 20, 35)])
        result = corrupt_roll_bin(source, 0, config(), ScriptedRNG("addition", -1))
        torch.testing.assert_close(result[:, 60], bin_tensors(source, 0)["roll"][:, 60])
        torch.testing.assert_close(result[:, 59], result[:, 60])

    def test_deletion_is_rare_and_preserves_other_evidence(self):
        source = piece([(60 + i, 80, 10 + i * 10, 15 + i * 10) for i in range(4)])
        result = corrupt_roll_bin(source, 0, config(), ScriptedRNG("deletion", selected=range(4)))
        self.assertEqual(int(result[1].sum()), 3)  # At most one deletion here.
        self.assertFalse(result[:, 60].any())

    def test_midi_pitch_edges_do_not_wrap(self):
        source = piece([(127, 80, 10, 30), (60, 90, 35, 45)])
        for operation in ("pitch", "addition"):
            result = corrupt_roll_bin(source, 0, config(), ScriptedRNG(operation, 1))
            torch.testing.assert_close(result, bin_tensors(source, 0)["roll"])

    def test_drums_never_have_percussion_pitches_shifted_or_added(self):
        source = piece([(36, 80, 20, 35), (42, 90, 40, 45)], drum=True)
        for operation in ("pitch", "addition"):
            result = corrupt_roll_bin(source, 0, config(), ScriptedRNG(operation))
            torch.testing.assert_close(result, bin_tensors(source, 0)["roll"])
        result = corrupt_roll_bin(source, 0, config(), ScriptedRNG("timing", 1))
        self.assertEqual(result[1, 36].nonzero().flatten().tolist(), [21])

    def test_silent_bins_are_not_filled_and_randomness_is_reproducible(self):
        silent = piece([], n_frames=33)
        result = corrupt_roll_bin(silent, 0, config(), ScriptedRNG("addition"))
        self.assertFalse(result.any())
        source = piece([(60, 80, 10, 30), (64, 90, 40, 50)])
        a = corrupt_roll_bin(source, 0, config(), random.Random(19))
        b = corrupt_roll_bin(source, 0, config(), random.Random(19))
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_pitch_augmentation_moves_clean_and_corrupt_inputs_together(self):
        source = piece([(60, 80, 20, 35)])
        item = bin_tensors(source, 0)
        item["corrupted_roll"] = corrupt_roll_bin(source, 0, config(), ScriptedRNG("addition", 1))
        clean, corrupt = item["roll"].clone(), item["corrupted_roll"].clone()
        transpose_bin(item, 3)
        torch.testing.assert_close(item["roll"][:, 63], clean[:, 60])
        torch.testing.assert_close(item["corrupted_roll"][:, 64], corrupt[:, 61])

    def test_transposition_range_accounts_for_corrupted_pitches_without_clipping(self):
        source = piece([(126, 80, 20, 35)])
        item = bin_tensors(source, 0)
        item["corrupted_roll"] = corrupt_roll_bin(source, 0, config(), ScriptedRNG("addition", 1))
        with self.assertRaisesRegex(ValueError, "MIDI range"):
            transpose_bin(item, 1)
        self.assertTrue(item["corrupted_roll"][0, 127].any())

    def test_collision_with_a_long_note_outside_crop_halo_is_rejected(self):
        source = piece([(60, 80, 20, 160), (61, 70, 150, 180)])
        # Added 61 would conflict at 1.5s even though decoding this bin alone
        # could not see that conflict. Event edits stay globally valid.
        result = corrupt_roll_bin(source, 0, config(), ScriptedRNG("addition", 1))
        torch.testing.assert_close(result, bin_tensors(source, 0)["roll"])

    def test_same_frame_attack_collision_is_rejected_not_normalized_silently(self):
        source = piece([(60, 80, 20, 21), (60, 70, 21, 22)])
        result = corrupt_roll_bin(source, 0, config(), ScriptedRNG("timing", 1))
        torch.testing.assert_close(result, bin_tensors(source, 0)["roll"])


class ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        return LocalVAE(LocalVAEConfig(base_channels=4, channel_multipliers=(1, 2), latent_channels=2, res_blocks=1))

    def test_channel_scale_is_valid_weighted_and_detached(self):
        mean = torch.tensor([[[[1., 3., 999.]]]], requires_grad=True)
        p = Posterior(mean, torch.zeros_like(mean), torch.tensor([[[[1., .5, 0.]]]]))
        scale = latent_channel_std(p)
        torch.testing.assert_close(scale, torch.tensor([np.sqrt(8 / 9)], dtype=torch.float32))
        self.assertFalse(scale.requires_grad)

    def test_jitter_zero_identity_seed_reproducibility_linear_strength_and_padding(self):
        mean = torch.arange(12.).reshape(2, 1, 2, 3)
        p = Posterior(mean, torch.zeros_like(mean), torch.tensor([[[[1., 1., 0.]]], [[[1., 1., 0.]]]]))
        expected = mean.masked_fill(~(p.weights > 0), 0)
        zero = jitter_latent(mean, p, 0., fixed=True)
        torch.testing.assert_close(zero, expected)
        a = jitter_latent(mean, p, .05, fixed=True, generator=torch.Generator().manual_seed(9))
        b = jitter_latent(mean, p, .10, fixed=True, generator=torch.Generator().manual_seed(9))
        torch.testing.assert_close(b - expected, 2 * (a - expected), rtol=1e-5, atol=1e-6)
        self.assertFalse(b[..., 2].any())
        self.assertTrue(torch.isfinite(b).all())

    def test_no_new_state_keys_and_default_forward_remains_equivalent(self):
        model, data = self.model().eval(), batch()
        keys = tuple(model.state_dict())
        a = model(data["roll"], data["pedal"], data["valid_mask"], sample=False)
        b = model(data["roll"], data["pedal"], data["valid_mask"], sample=False,
                  corrupted_roll=data["corrupted_roll"], latent_jitter_std=.1, fixed_jitter=True)
        torch.testing.assert_close(a.posterior.mean, b.posterior.mean)
        torch.testing.assert_close(a.reconstruction.onset_logits, b.reconstruction.onset_logits)
        self.assertEqual(tuple(model.state_dict()), keys)
        self.assertIsNone(a.denoising)
        self.assertIsNone(a.jittered)
        model.load_state_dict(model.state_dict(), strict=True)

    def test_all_branches_have_finite_gradients_and_exact_weighted_objective(self):
        model, data = self.model(), batch()
        c = config(ROBUSTNESS_WARMUP_STEPS=2, KL_WARMUP_STEPS=2)
        loss, stats = batch_objective(model, data, c, 0)
        values = summarize(stats)
        expected = values["reconstruction"] + .5 * (
            c.SEMANTIC_WEIGHT * values["denoising_reconstruction"] + c.LATENT_JITTER_WEIGHT * values["jitter_reconstruction"]
        ) + .5 * c.KL_WEIGHT * values["kl"]
        self.assertAlmostEqual(loss.item(), expected, places=5)
        self.assertAlmostEqual(values["robust_reconstruction"],
            (values["reconstruction"] + .5 * values["denoising_reconstruction"] + .5 * values["jitter_reconstruction"]) / 2, places=5)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_auxiliary_branches_can_be_disabled(self):
        c, data = config(SEMANTIC_CORRUPTION=False, LATENT_JITTER_WEIGHT=0.), batch()
        loss, stats = batch_objective(self.model(), data, c, 0)
        values = summarize(stats)
        self.assertEqual(values["denoising_reconstruction"], 0.)
        self.assertEqual(values["jitter_reconstruction"], 0.)
        self.assertEqual(values["reconstruction"], values["robust_reconstruction"])

    def test_auxiliary_loss_targets_clean_not_corrupted_roll(self):
        model, data = self.model(), batch()
        data["corrupted_roll"] = torch.zeros_like(data["roll"])
        # Reconstruction terms only read the clean roll in the batch.
        from improvnet.model.local_vae_model import reconstruction_losses
        output = model(data["roll"], data["pedal"], data["valid_mask"], sample=False,
                       corrupted_roll=data["corrupted_roll"])
        clean = reconstruction_losses(output.denoising, data)
        wrong = reconstruction_losses(output.denoising, dict(data, roll=data["corrupted_roll"]))
        self.assertFalse(torch.equal(clean["onset"], wrong["onset"]))

    def test_validation_stress_noise_is_fixed_and_preserves_training_rng(self):
        data, model = batch(), self.model()
        c = config(MAX_VALIDATION_BATCHES=1)
        torch.manual_seed(73)
        state = torch.get_rng_state().clone()
        a = evaluate(model, [data], c, torch.device("cpu"), 2)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        b = evaluate(model, [data], c, torch.device("cpu"), 2)
        self.assertEqual(a, b)
        self.assertTrue(model.training)

    def test_sparse_class_balance_favours_false_positive_suppression(self):
        target = torch.tensor([[1., 0.]])
        mask = torch.ones_like(target, dtype=torch.bool)
        logits = torch.tensor([[10., 10.]])  # Correct positive, very wrong negative.
        equal = balanced_bce(logits, target, mask)
        precision_favouring = balanced_bce(logits, target, mask, positive_fraction=.35)
        self.assertTrue(precision_favouring > equal)
        # A silent patch still receives its entire negative-class loss.
        silent = torch.zeros_like(target)
        torch.testing.assert_close(balanced_bce(logits, silent, mask),
                                   balanced_bce(logits, silent, mask, positive_fraction=.35))

    def test_bad_robustness_configs_are_rejected_and_warmup_is_bounded(self):
        for change in (dict(SEMANTIC_NOTE_PROB=1.1), dict(SEMANTIC_OPERATION_WEIGHTS={"bogus": 1}),
                       dict(SEMANTIC_OPERATION_WEIGHTS={"timing": 0}), dict(SEMANTIC_MIN_NOTES_FOR_DELETION=1),
                       dict(LATENT_JITTER_MAX_STD=-1), dict(ACTIVITY_POSITIVE_FRACTION=0),
                       dict(ROBUSTNESS_WARMUP_STEPS=-1)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_config(config(**change))
        self.assertAlmostEqual(robustness_ramp(0, config(ROBUSTNESS_WARMUP_STEPS=100)), .01)
        self.assertEqual(robustness_ramp(200, config(ROBUSTNESS_WARMUP_STEPS=100)), 1.)


if __name__ == "__main__":
    unittest.main()
