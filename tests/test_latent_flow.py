"""Synthetic stage-2 tests; no real training or saved-checkpoint changes."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mido
import numpy as np
import torch

from improvnet.model import latent_flow_config as defaults
from improvnet.model.latent_flow_model import LatentFlowConfig, LatentFlowModel
from improvnet.model.latent_flow_train import batch_objective, model_config, validate_config
from improvnet.model.local_vae_model import LocalVAE, LocalVAEConfig
from improvnet.tokenizer.hierarchical import GridConfig, HierarchicalTokenizer, read_performance
from improvnet.utils.local_vae_data import build_midi_index
from improvnet.utils.latent_flow import (LatentNormalizer, checkpoint_digest, clean_from_velocity,
    encode_windows, flow_target, sample, weighted_loss)
from improvnet.utils.latent_flow_data import (LatentFlowDataset, accompaniment, collate_windows,
    genre_labels, transpose_window, window_tensors)
from improvnet.inference.latent_flow_inference import generate, load_model, reference_events, resize_piece
from improvnet.utils.note_decoding import NoteDecoding


def midi():
    result = mido.MidiFile(ticks_per_beat=1000)
    # MIDI type 0-style container: four parts share one physical track.
    messages = [(0, mido.Message("program_change", channel=i, program=p)) for i, p in enumerate((0, 32, 40, 48, 56))]
    for channel in range(5):
        for j in range(10):
            start = (j * .21 + .1) * 2000
            end = start + (1100 if j == 0 else 220)
            messages.extend(((round(start), mido.Message("note_on", channel=channel, note=60 + channel + j % 3, velocity=70 + channel)),
                             (round(end), mido.Message("note_off", channel=channel, note=60 + channel + j % 3))))
    track = mido.MidiTrack(); previous = 0
    for tick, message in sorted(messages, key=lambda e: e[0]):
        track.append(message.copy(time=tick - previous)); previous = tick
    track.append(mido.MetaMessage("end_of_track", time=2000))
    result.tracks.append(track)
    return result


def config(**kwargs):
    values = {k: v for k, v in vars(defaults).items() if k.isupper()}
    values.update(HISTORY_BINS=1, TARGET_BINS=2, D_MODEL=32, N_LAYERS=2, N_HEADS=4,
        PATCH_PITCH=4, PATCH_TIME=5, NUM_WORKERS=0, VAE_BATCH_SIZE=2,
        MAX_ACCOMPANIMENT_EVENTS=8, LATENT_CALIBRATION_SAMPLES=2, SAMPLES_PER_EPOCH=8,
        VALIDATION_SAMPLES=4, PITCH_AUGMENTATION=False, ARIA_METADATA=None,
        GIGAMIDI_METADATA=None, DATA_SPLIT_RATIOS={"train": .7, "validation": .3, "test": 0.})
    values.update(kwargs)
    return SimpleNamespace(**values)


def fixture(mode="full", start=1, *, latent_channels=8, **overrides):
    c = config(**overrides)
    vae = LocalVAE(LocalVAEConfig(base_channels=4, res_blocks=1,
                                latent_channels=latent_channels)).eval().requires_grad_(False)
    piece = HierarchicalTokenizer().encode(midi())
    item = window_tensors(piece, start, c, mode=mode)
    item["retries"] = torch.tensor(0)
    batch = collate_windows([item])
    model = LatentFlowModel(model_config(c, vae, ("<unconditioned>", "classical", "jazz")))
    normalizer = LatentNormalizer(torch.zeros(latent_channels), torch.ones(latent_channels))
    return c, vae, piece, batch, model, normalizer


class Stage2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_soft_pitch_and_carry_information(self):
        c, _, piece, _, _, _ = fixture()
        item = window_tensors(piece, 0, c)
        self.assertFalse(item["known"].any())
        self.assertFalse(item["valid_mask"][0].any())
        expected = torch.from_numpy(piece.velocity[:2].any(-1).astype(np.float32))
        torch.testing.assert_close(item["pitch"][1:], expected)
        silent = deepcopy(piece)
        silent.velocity[:] = 0; silent.onset[:] = 0; silent.refresh_coarse()
        self.assertTrue(window_tensors(silent, 0, c)["pitch_present"][1:].all())

    def test_track_limit_splits_shared_midi_container(self):
        c, _, piece, _, _, _ = fixture()
        item = window_tensors(piece, 1, c)
        self.assertEqual(len(item["accompaniment_info"]["parts"]), 3)
        self.assertEqual(len({p[1] for p in item["accompaniment_info"]["parts"]}), 3)
        self.assertLessEqual(len(item["events"]), c.MAX_ACCOMPANIMENT_EVENTS)
        self.assertGreater(item["accompaniment_info"]["dropped_events"], 0)
        self.assertLess(int(item["events"][:, 0].max()), 3)

    def test_carried_events_have_signed_positions(self):
        c, _, piece, _, _, _ = fixture()
        item = window_tensors(piece, 1, c)
        # origin bin 0, so choose a window beginning at bin 1 with no history slot.
        c.HISTORY_BINS = 0
        item = window_tensors(piece, 1, c)
        self.assertTrue((item["event_positions"][:, 0] < 0).any())

    def test_rhythm_has_no_pitch_and_budget_does_not_change_it(self):
        c, _, piece, _, _, _ = fixture()
        first = window_tensors(piece, 0, c, mode="rhythm")
        other = deepcopy(piece); other.events[:, 4] = (other.events[:, 4] + 5) % 128
        c.MAX_ACCOMPANIMENT_EVENTS = 1
        second = window_tensors(other, 0, c, mode="rhythm")
        torch.testing.assert_close(first["rhythm"], second["rhythm"])
        self.assertEqual(len(first["events"]), 0)
        self.assertTrue(first["rhythm"].any())

    def test_consistent_transposition_not_on_drums(self):
        _, _, _, batch, _, _ = fixture()
        c, _, piece, _, _, _ = fixture()
        item = window_tensors(piece, 1, c)
        original = deepcopy(item)
        transpose_window(item, 3)
        torch.testing.assert_close(item["roll"][:, :, 3:], original["roll"][:, :, :-3])
        torch.testing.assert_close(item["pitch"][:, 3:], original["pitch"][:, :-3])
        torch.testing.assert_close(item["events"][:, 3], original["events"][:, 3] + 3)
        torch.testing.assert_close(item["rhythm"], original["rhythm"])

    def test_nonempty_target_and_trailing_padding(self):
        c, vae, piece, _, _, _ = fixture()
        item = window_tensors(piece, len(piece.velocity) - 1, c)
        self.assertTrue(item["valid_mask"][c.HISTORY_BINS].any())
        self.assertFalse(item["valid_mask"][-1].any())
        batch = collate_windows([item])
        z, w = encode_windows(vae, batch, 1)
        self.assertFalse(w[:, -1].any())
        self.assertFalse(z[:, -1].any())

    def test_straight_flow_path_velocity_and_clean_history(self):
        c, vae, _, batch, _, norm = fixture()
        z, w = encode_windows(vae, batch)
        z = norm.normalize(z, w)
        for time in (0., .2, .5, 1.):
            epsilon = torch.randn_like(z)
            noisy, v, levels = flow_target(z, batch["known"], w, torch.tensor([time]), epsilon)
            expected_state = torch.where(batch["known"][..., None, None, None], z, (1-time)*epsilon + time*z).masked_fill(~(w > 0), 0)
            expected_velocity = (z-epsilon).masked_fill(batch["known"][..., None, None, None] | ~(w > 0), 0)
            torch.testing.assert_close(noisy, expected_state)
            torch.testing.assert_close(v, expected_velocity)
            torch.testing.assert_close(noisy[:, :1], z[:, :1])
            torch.testing.assert_close(clean_from_velocity(noisy, v, levels), z, atol=1e-6, rtol=1e-5)
            self.assertTrue((levels[:, :1] == 1).all())

    def test_loss_excludes_history_and_padding_even_nan(self):
        p = torch.zeros(1, 3, 2, 4, 5)
        w = torch.ones(1, 3, 1, 1, 5); w[:, -1] = 0
        known = torch.tensor([[True, False, False]])
        p[:, 0] = float("nan"); p[:, -1] = float("nan")
        self.assertEqual(weighted_loss(p, torch.ones_like(p), w, known).item(), 1.)

    def test_normalization_roundtrip_and_mask(self):
        z = torch.randn(2, 3, 8, 4, 5)
        w = torch.ones(2, 3, 1, 1, 5); w[:, :, :, :, -1] = 0
        norm = LatentNormalizer(torch.arange(8), torch.arange(8) + .5)
        clean = z.masked_fill(~(w > 0), 0)
        torch.testing.assert_close(norm.denormalize(norm.normalize(z, w), w), clean)

    def test_model_shapes_and_all_parameters_have_gradients(self):
        for mode in ("full", "rhythm", "none"):
            c, vae, _, batch, model, norm = fixture(mode)
            loss, stats = batch_objective(model, vae, norm, batch, c)
            loss.backward()
            self.assertTrue(torch.isfinite(stats).all())
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertTrue(all(p.grad is None for p in vae.parameters()))

    def test_vector_vae_one_token_per_bin_flow_and_sampling(self):
        c, _, piece, batch, _, _ = fixture("rhythm", PATCH_PITCH=1, PATCH_TIME=1)
        vae = LocalVAE(LocalVAEConfig(base_channels=4, res_blocks=1,
                       latent_layout="vector", latent_channels=256)).eval().requires_grad_(False)
        model = LatentFlowModel(model_config(c, vae, ("unconditioned", "jazz")))
        norm = LatentNormalizer(torch.zeros(256), torch.ones(256))
        z, w = encode_windows(vae, batch, 1)
        self.assertEqual(z.shape, (1, 3, 256, 1, 1))
        self.assertEqual(model.patchify(z).flatten(1, 3).shape, (1, 3, 256))
        loss, stats = batch_objective(model, vae, norm, batch, c)
        loss.backward()
        self.assertTrue(torch.isfinite(stats).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue(all(p.grad is None for p in vae.parameters()))
        model.eval()
        generated = sample(model, z, batch["known"], w, batch, steps=2)
        torch.testing.assert_close(generated[batch["known"]], z[batch["known"]])
        self.assertEqual(vae.decode(generated.flatten(0, 1)).velocity.shape, (3, 128, 100))
        output, baseline, reference, report = generate(model, vae, norm, piece, c,
            task="reconstruct", prompt_bins=1, max_bins=3, steps=2, strength=.2,
            progress=False, note_decoding=NoteDecoding('held', .5, 20))
        self.assertEqual(report["flow_architecture"]["tokens_per_bin"], 1)
        self.assertEqual(report["flow_architecture"]["tokens_per_window"], 3)
        torch.testing.assert_close(torch.from_numpy(output.velocity[:1]), torch.from_numpy(reference.velocity[:1]))

    def test_padding_and_inactive_modalities_are_invariant(self):
        c, vae, _, batch, model, _ = fixture("rhythm", start=0)
        model.eval()
        # Nonzero final projection lets us test conditioning influence, not only zero initialization.
        torch.nn.init.normal_(model.output.weight, std=.02)
        z, w = encode_windows(vae, batch)
        levels = torch.ones(1, 3) * .5; levels[batch["known"]] = 1
        a = model(z, levels, batch["known"], w, batch)
        other = deepcopy(batch)
        other["events"][:] = 999
        other["event_positions"][:] = float("nan")
        other["rhythm"][:, :, 0] = float("nan")
        noisy = z.clone(); noisy[:, 0] = float("nan")
        b = model(noisy, levels, batch["known"], w, other)
        torch.testing.assert_close(a, b)

    def test_partial_bin_rhythm_padding_is_excluded_from_attention(self):
        c, vae, _, batch, model, _ = fixture("rhythm")
        batch["valid_mask"][:, -1, 17:] = False
        batch["roll"][:, -1, :, :, 17:] = 0
        batch["pedal"][:, -1, :, 17:] = 0
        z, w = encode_windows(vae, batch)
        _, valid = model.conditions(batch, w.flatten(2).any(-1))
        mask = valid[:, -c.MAX_ACCOMPANIMENT_TRACKS * 3 * 10:].reshape(1, 3, 3, 10)
        self.assertTrue(mask[:, :, -1, :2].all())
        self.assertFalse(mask[:, :, -1, 2:].any())
        other = deepcopy(batch)
        other["rhythm"][:, :, -1, 17:] = float("nan")
        a, _ = model.conditions(batch, w.flatten(2).any(-1))
        b, _ = model.conditions(other, w.flatten(2).any(-1))
        torch.testing.assert_close(a, b)

    def test_flow_training_exact_resume_with_frozen_vector_vae(self):
        from improvnet.model.latent_flow_train import train
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root / "data"; folder.mkdir()
            for i in range(16):
                midi().save(folder / f"{i}.mid")
            c = config(DATASETS={"doug": str(folder)}, INDEX_PATH=str(root / "index.sqlite"),
                SAVE_DIR=str(root / "expected"), VAE_CHECKPOINT=str(root / "vae.pt"),
                BATCH_SIZE=2, ACCUM_STEPS=1, N_STEPS=3, WARMUP_STEPS=1,
                LOG_EVERY=1, VALIDATE_EVERY=2, SAVE_EVERY=1, MAX_VALIDATION_BATCHES=2,
                PATCH_PITCH=1, PATCH_TIME=1)
            with redirect_stdout(io.StringIO()):
                build_midi_index(c)
            vae = LocalVAE(LocalVAEConfig(base_channels=4, res_blocks=1,
                latent_layout="vector", latent_channels=512))
            torch.save(dict(format="local_vae_training_v1", model_config=asdict(vae.config),
                model_state_dict=vae.state_dict(), train_config=vars(c), step=0), c.VAE_CHECKPOINT)
            before = checkpoint_digest(c.VAE_CHECKPOINT)
            with redirect_stdout(io.StringIO()):
                train(c, force_cpu=True)
            expected = torch.load(Path(c.SAVE_DIR) / "latest_checkpoint.pt", weights_only=False)
            c.SAVE_DIR = str(root / "resumed")
            request = root / "request"; request.touch()
            with patch.dict(os.environ, CHECKPOINT_REQUEST_FILE=str(request)), redirect_stdout(io.StringIO()):
                train(c, force_cpu=True)
            request.unlink()
            with redirect_stdout(io.StringIO()):
                train(c, force_cpu=True)
            actual = torch.load(Path(c.SAVE_DIR) / "latest_checkpoint.pt", weights_only=False)
            self.assertEqual(actual["format"], "latent_flow_training_v1")
            self.assertEqual(actual["step"], 3)
            self.assertEqual(before, checkpoint_digest(c.VAE_CHECKPOINT))
            for key in ("model_state_dict", "ema_state_dict"):
                for name in expected[key]:
                    torch.testing.assert_close(actual[key][name], expected[key][name], rtol=0, atol=0)

    def test_augmented_dataset_and_calibration_ignore_non_targets(self):
        from improvnet.model.latent_flow_train import calibrate
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root / "data"; folder.mkdir()
            for i in range(16):
                midi().save(folder / f"{i}.mid")
            c = config(DATASETS={"doug": str(folder)}, INDEX_PATH=str(root / "index.sqlite"),
                       PITCH_AUGMENTATION=True, PITCH_SHIFT_RANGE=(3, 3))
            with redirect_stdout(io.StringIO()):
                build_midi_index(c)
            dataset = LatentFlowDataset(c)
            item = dataset[0]
            self.assertTrue(item["roll"][:, 0, 63:66].any())
            vae = LocalVAE(LocalVAEConfig(base_channels=4, res_blocks=1)).eval().requires_grad_(False)
            norm = calibrate(vae, dataset, c, torch.device("cpu"))
            self.assertTrue(torch.isfinite(norm.std).all())
            self.assertTrue((norm.std > 0).all())
            dataset.connection.close()

    def test_multi_hot_is_conditioning_not_vocabulary_constraint(self):
        c, vae, _, batch, model, _ = fixture("none")
        torch.nn.init.normal_(model.output.weight, std=.02)
        model.eval()
        z, w = encode_windows(vae, batch)
        levels = torch.ones(1, 3) * .3; levels[batch["known"]] = 1
        a = model(z, levels, batch["known"], w, batch)
        other = deepcopy(batch); other["pitch"].zero_()
        b = model(z, levels, batch["known"], w, other)
        self.assertFalse(torch.equal(a, b))
        self.assertTrue(b[:, 1:].abs().sum() > 0)  # Empty pitch set does not force silent output.

    def test_sampler_reclamps_prefix_every_pass_and_full_noise_ignores_source(self):
        c, vae, _, batch, model, norm = fixture()
        model.eval()
        z, w = encode_windows(vae, batch); z = norm.normalize(z, w)
        snapshots = []
        result = sample(model, z, batch["known"], w, batch, steps=4,
            generator=torch.Generator().manual_seed(9), callback=lambda i, t, x: snapshots.append(x.clone()))
        for snapshot in snapshots:
            torch.testing.assert_close(snapshot[:, :1], z[:, :1], rtol=0, atol=0)
        other = z.clone(); other[:, 1:] += 100
        repeated = sample(model, other, batch["known"], w, batch, steps=4, generator=torch.Generator().manual_seed(9))
        torch.testing.assert_close(result, repeated)

    def test_oracle_sampler_recovers_source_at_any_noise(self):
        _, vae, _, batch, _, norm = fixture()
        clean, w = encode_windows(vae, batch); clean = norm.normalize(clean, w)
        class Oracle:
            def __call__(self, z, levels, known, weights, conditioning):
                return (clean - z) / (1 - levels[..., None, None, None]).clamp_min(1e-8)
        for strength in (.2, .8, 1.):
            output = sample(Oracle(), clean, batch["known"], w, batch, steps=4, strength=strength)
            torch.testing.assert_close(output, clean, atol=2e-5, rtol=2e-5)

    def test_generation_and_midi_roundtrip_across_windows(self):
        c, vae, piece, _, model, norm = fixture("full")
        model.eval()
        output, baseline, ref, report = generate(model, vae, norm, piece, c, task="reconstruct", prompt_bins=1,
            max_bins=3, steps=2, strength=.2, progress=False, note_decoding=NoteDecoding('held', .5, 20))
        self.assertTrue(np.array_equal(output.velocity[0], piece.velocity[0]))
        self.assertTrue(np.array_equal(output.onset[0], piece.onset[0]))
        self.assertIn("vae_jitter_at_measured_residual", report)
        self.assertIn("latent_nrmse_vs_source", report)
        self.assertEqual(report['note_decoding']['mode'], 'held')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "generated.mid"
            HierarchicalTokenizer().decode(output).save(path)
            self.assertTrue(len(mido.MidiFile(path).tracks) > 0)

    def test_unit_patch_generation_reuses_latent_history_and_partial_window(self):
        c, vae, piece, _, model, norm = fixture("none", latent_channels=4,
                                               PATCH_PITCH=1, PATCH_TIME=1)
        calls = []

        def checked_sample(model, source, known, weights, conditioning, **kwargs):
            if calls:
                torch.testing.assert_close(source[:, :1], calls[-1][1][:, -1:], rtol=0, atol=0)
            result = sample(model, source, known, weights, conditioning, **kwargs)
            calls.append((source.clone(), result.clone(), weights.clone()))
            return result

        with patch("improvnet.inference.latent_flow_inference.sample", side_effect=checked_sample):
            output, _, _, report = generate(model.eval(), vae, norm, piece, c, task="reconstruct",
                prompt_bins=1, max_bins=3, steps=2, strength=.2, mode="none", progress=False,
                note_decoding=NoteDecoding("held", .5, 20))
        self.assertEqual(len(calls), 2)
        self.assertFalse(calls[-1][2][:, -1].any())
        np.testing.assert_array_equal(output.velocity[0], piece.velocity[0])
        self.assertEqual(report["flow_architecture"]["tokens_per_bin"], 800)
        self.assertEqual(report["flow_architecture"]["latent_channels"], 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unit-patch.mid"
            HierarchicalTokenizer().decode(output).save(path)
            self.assertTrue(mido.MidiFile(path).tracks)

    def test_generation_rejects_mismatched_window_settings_before_sampling(self):
        c, vae, piece, _, model, norm = fixture()
        for name, value in (("HISTORY_BINS", 8), ("TARGET_BINS", 32), ("PATCH_TIME", 1)):
            other = deepcopy(c)
            setattr(other, name, value)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                generate(model, vae, norm, piece, other, progress=False)

    def test_current_unit_patch_checkpoint_ignores_live_architecture_defaults(self):
        c, vae, _, _, model, norm = fixture(latent_channels=4, HISTORY_BINS=8, TARGET_BINS=32,
            PATCH_PITCH=1, PATCH_TIME=1, N_LAYERS=16)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vae_path, checkpoint = root / "vae.pt", root / "flow.pt"
            torch.save(dict(format="local_vae_training_v1", model_config=asdict(vae.config),
                model_state_dict=vae.state_dict(), step=35, train_config=dict(COARSE_MS=1000, FINE_MS=10,
                PIANO_PROGRAMS=tuple(range(8)), MAX_FRAMES=360000)), vae_path)
            c.VAE_CHECKPOINT = str(vae_path)
            state = dict(format="latent_flow_training_v1", model_config=asdict(model.config),
                train_config=vars(c), genres=["<unconditioned>", "classical", "jazz"],
                model_state_dict=model.state_dict(), ema_state_dict=model.state_dict(),
                normalizer=norm.state_dict(), vae_sha256=checkpoint_digest(vae_path), step=7)
            torch.save(state, checkpoint)
            with patch.multiple(defaults, HISTORY_BINS=32, TARGET_BINS=128,
                                PATCH_PITCH=4, PATCH_TIME=5, N_LAYERS=12):
                for use_ema in (True, False):
                    loaded, _, _, normalizer, saved, info = load_model(checkpoint, torch.device("cpu"), use_ema=use_ema)
                    self.assertEqual(loaded.config, model.config)
                    self.assertEqual((saved.HISTORY_BINS, saved.TARGET_BINS), (8, 32))
                    self.assertEqual(info["weights"], "ema" if use_ema else "raw")
                    torch.testing.assert_close(normalizer.std, norm.std)
                    z = torch.randn(1, 40, 4, 32, 25)
                    tokens = loaded.patchify(z)
                    self.assertEqual(tokens.shape, (1, 40, 32, 25, 4))
                    torch.testing.assert_close(loaded.unpatchify(tokens), z, rtol=0, atol=0)
            state["train_config"] = dict(vars(c), HISTORY_BINS=32)
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, "HISTORY_BINS"):
                load_model(checkpoint, torch.device("cpu"))

    def test_continuation_can_extend_an_entire_source_prompt(self):
        c, vae, piece, _, model, norm = fixture("none")
        prompt = resize_piece(piece, 1)
        output, _, _, _ = generate(model.eval(), vae, norm, prompt, c, task="continue",
            prompt_bins=1, max_bins=3, steps=2, mode="none", progress=False)
        self.assertEqual(len(output.velocity), 4)
        self.assertTrue(output.valid_mask.all())
        np.testing.assert_array_equal(output.velocity[0], prompt.velocity[0])

    def test_resize_and_external_reference_include_piano(self):
        _, _, piece, _, _, _ = fixture()
        resized = resize_piece(piece, 1)
        self.assertTrue(((resized.events[:, 8] * 100 + resized.events[:, 9]) <= resized.n_frames).all())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.mid"; midi().save(path)
            events = reference_events(path, GridConfig(), time_scale=2., offset_seconds=.1)
            self.assertTrue((events[:, 1] == 0).any())
            self.assertEqual(events[0, 6] * 100 + events[0, 7], 30)

    def test_invalid_args_and_checkpoint_protection(self):
        c = config(SAVE_DIR=str(Path(defaults.VAE_CHECKPOINT).parent))
        with self.assertRaises(ValueError):
            validate_config(c)
        with self.assertRaises(ValueError):
            LatentFlowConfig(max_tracks=4)
        with self.assertRaises(ValueError):
            LatentNormalizer([0], [0])

    def test_checkpoint_reload_and_wrong_vae_rejection(self):
        c, vae, _, _, model, norm = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / "vae.pt"
            torch.save(dict(format="local_vae_training_v1", model_config=asdict(vae.config),
                model_state_dict=vae.state_dict(), step=35, train_config=dict(COARSE_MS=1000, FINE_MS=10,
                PIANO_PROGRAMS=tuple(range(8)), MAX_FRAMES=360000)), path)
            c.VAE_CHECKPOINT = str(path)
            checkpoint = root / "stage2.pt"
            torch.save(dict(format="latent_flow_training_v1", model_config=asdict(model.config),
                train_config=vars(c), genres=["<unconditioned>", "classical", "jazz"],
                model_state_dict=model.state_dict(), ema_state_dict=model.state_dict(),
                normalizer=norm.state_dict(), vae_sha256=checkpoint_digest(path), step=7), checkpoint)
            loaded = load_model(checkpoint, torch.device("cpu"))
            self.assertEqual(loaded[-1]["step"], 7)
            with path.open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "different VAE"):
                load_model(checkpoint, torch.device("cpu"))

    def test_index_genres_and_reproducible_midi_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root / "data"; folder.mkdir()
            for i in range(16):
                midi().save(folder / f"{i}.mid")
            c = config(DATASETS={"doug": str(folder)}, INDEX_PATH=str(root / "index.sqlite"))
            with redirect_stdout(io.StringIO()):
                build_midi_index(c)
            self.assertEqual(genre_labels(c.INDEX_PATH), ("<unconditioned>", "jazz"))
            dataset = LatentFlowDataset(c)
            first, second = dataset[0], dataset[0]
            for key in first:
                if isinstance(first[key], torch.Tensor):
                    torch.testing.assert_close(first[key], second[key])
            self.assertTrue(first["valid_mask"][c.HISTORY_BINS:].any())
            dataset.connection.close()


if __name__ == "__main__":
    unittest.main()
