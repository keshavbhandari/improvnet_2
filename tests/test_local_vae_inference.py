"""CPU tests of reconstruction, mask-aware errors and held-out evaluation."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import mido
import numpy as np
import torch

from improvnet.inference.local_vae_inference import Scores, LatentStats, load_model, main, reconstruct
from improvnet.inference.local_vae_evaluate import main as evaluate, readiness, candidates
from improvnet.model import local_vae_config as defaults
from improvnet.model.local_vae_model import LocalVAE, LocalVAEConfig, Posterior, Reconstruction
from improvnet.utils.note_decoding import NoteDecoding
from improvnet.tokenizer.hierarchical import HierarchicalTokenizer, read_performance
from improvnet.utils.local_vae_data import build_midi_index


def source():
    result = mido.MidiFile(ticks_per_beat=1000)
    # 0.2--1.2s note crosses a bin; repeated note at 1.2s; accompaniment channel 2.
    events = [(0, mido.Message("program_change", program=0)),
              (0, mido.Message("program_change", channel=2, program=40)),
              (400, mido.Message("note_on", note=60, velocity=80)),
              (600, mido.Message("note_on", channel=2, note=72, velocity=90)),
              (1000, mido.Message("note_off", channel=2, note=72)),
              (2400, mido.Message("note_off", note=60)),
              (2400, mido.Message("note_on", note=60, velocity=70)),
              (2600, mido.Message("note_off", note=60))]
    track = mido.MidiTrack(); previous = 0
    for tick, msg in events:
        track.append(msg.copy(time=tick - previous)); previous = tick
    track.append(mido.MetaMessage("end_of_track")); result.tracks.append(track)
    return result


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.piece = HierarchicalTokenizer().encode(source())
        self.model = LocalVAE(LocalVAEConfig(base_channels=4, channel_multipliers=(1, 2),
                                             latent_channels=2, res_blocks=1)).eval()

    def test_perfect_metrics_and_cross_bin_notes(self):
        scores = Scores(); scores.arrays(self.piece, self.piece); scores.notes(self.piece, self.piece)
        report = scores.report()
        self.assertEqual(report["notes_f1"], 1)
        self.assertEqual(report["missed_notes"], 0)
        self.assertEqual(scores.counts["notes_tp"], 2)  # Not three bin-fragment notes.
        self.assertEqual(report["velocity_mae"], 0)

    def test_missing_extra_and_timing_notes(self):
        other = deepcopy(self.piece)
        other.velocity[:, 60] = 0; other.onset[:, 60] = 0
        other.velocity[0, 65, 10:20] = 50; other.onset[0, 65, 10] = 1
        scores = Scores(); scores.arrays(self.piece, other); scores.notes(self.piece, other)
        report = scores.report()
        self.assertEqual((report["missed_notes"], report["extra_notes"]), (2, 1))
        self.assertEqual(report["notes_f1"], 0)
        self.assertIsNone(report["onset_mae_ms"])
        self.assertAlmostEqual(report["velocity_mae"], scores.counts["velocity_error"] / scores.counts["active"], places=4)

    def test_activity_gap_creates_extra_note_despite_perfect_onset_predictions(self):
        other = deepcopy(self.piece)
        other.velocity[0, 60, 50] = 0  # One 10-ms gap splits a held note in MIDI decoding.
        scores = Scores(); scores.arrays(self.piece, other); scores.notes(self.piece, other)
        report = scores.report()
        self.assertEqual(report["onset_precision"], 1)
        self.assertEqual(report["onset_recall"], 1)
        self.assertEqual(report["reference_notes"], 2)
        self.assertEqual((report["extra_notes"], report["missed_notes"]), (1, 0))
        self.assertEqual(report["notes_precision"], .6667)
        self.assertEqual(report["notes_recall"], 1)
        self.assertEqual(report["notes_f1"], .8)

    def test_padding_and_unselected_bins_excluded(self):
        other = deepcopy(self.piece)
        other.velocity[1, :, 30:] = 127; other.onset[1, :, 30:] = 1
        other.velocity[0] = 0; other.onset[0] = 0
        scores = Scores(); scores.arrays(self.piece, other, np.array([1]))
        self.assertEqual(scores.report()["activity_f1"], 1)
        self.assertEqual(scores.report()["onset_f1"], 1)

    def test_no_pedal_is_null_not_perfect_zero(self):
        piece = deepcopy(self.piece); piece.metadata["pedal_enabled"] = False
        scores = Scores(); scores.arrays(piece, piece)
        self.assertIsNone(scores.report()["pedal_mae"])

    def test_mean_reconstruction_deterministic_and_context_unchanged(self):
        a, _, _, _ = reconstruct(self.model, self.piece, batch_size=1)
        b, _, sampled, zero = reconstruct(self.model, self.piece, diagnostics=True)
        np.testing.assert_array_equal(a.velocity, b.velocity)
        np.testing.assert_array_equal(self.piece.events, b.events)
        self.assertFalse(a.velocity[~self.piece.valid_mask[:, None].repeat(128, axis=1)].any())
        a.validate()
        self.assertIsNotNone(sampled); self.assertIsNotNone(zero)
        HierarchicalTokenizer().decode(a).save(file=io.BytesIO())

    def test_held_decoding_carries_across_inference_batches(self):
        class Echo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.marker = torch.nn.Parameter(torch.zeros(()))
                self.note_decoding = NoteDecoding('held', .5, 20)
                self.decoder_calibration = dict(activity_threshold=.95, onset_threshold=.995)

            def encode(self, roll, pedal, mask):
                return Posterior(roll, torch.full_like(roll, -20), mask[:, None, None].float())

            def decode(self, latent):
                activity = torch.where(latent[:, 0] > 0, .999, .001)
                activity[:, :, 99] = .01  # Uncertainty immediately before a bin seam.
                onset = torch.where(latent[:, 1] > 0, .999, .001)
                return Reconstruction(torch.logit(activity), torch.logit(onset), latent[:, 0],
                                      torch.zeros((len(latent), 1, 100)))

        model = Echo()
        a, _, _, _ = reconstruct(model, self.piece, batch_size=1)
        b, _, _, _ = reconstruct(model, self.piece, batch_size=2)
        np.testing.assert_array_equal(a.velocity, b.velocity)
        np.testing.assert_array_equal(a.onset, b.onset)
        scores = Scores(); scores.notes(self.piece, a)
        self.assertEqual(scores.report()['notes_with_offsets_f1'], 1)
        self.assertEqual(scores.report()['offset_mae_ms'], 0)
        legacy, _, _, _ = reconstruct(model, self.piece, note_decoding=NoteDecoding())
        scores = Scores(); scores.notes(self.piece, legacy)
        self.assertEqual(scores.report()['extra_notes'], 1)

    def test_onset_matching_tolerance_one_to_one_and_offset_errors(self):
        other = deepcopy(self.piece)
        other.velocity[:] = 0; other.onset[:] = 0
        # Attack 40ms late, release 30ms early; repeat remains unchanged.
        other.velocity[0, 60, 24:] = 80; other.onset[0, 60, 24] = 1
        other.velocity[1, 60, :17] = 80
        other.velocity[1, 60, 20:30] = 70; other.onset[1, 60, 20] = 1
        score = Scores(); score.notes(self.piece, other)
        self.assertEqual(score.report()["notes_f1"], 1)
        self.assertEqual(score.report()["onset_mae_ms"], 20)
        self.assertEqual(score.report()["offset_mae_ms"], 15)
        strict = Scores(); strict.notes(self.piece, other, tolerance_ms=30)
        self.assertEqual(strict.report()["missed_notes"], 1)
        self.assertEqual(strict.report()["extra_notes"], 1)

    def test_seeded_samples_and_bad_arguments(self):
        a, _, _, _ = reconstruct(self.model, self.piece, sample=True, generator=torch.Generator().manual_seed(7))
        b, _, _, _ = reconstruct(self.model, self.piece, sample=True, generator=torch.Generator().manual_seed(7))
        np.testing.assert_array_equal(a.velocity, b.velocity)
        for kwargs in (dict(batch_size=0), dict(onset_threshold=1), dict(activity_threshold=0)):
            with self.assertRaises(ValueError): reconstruct(self.model, self.piece, **kwargs)

    def test_latent_fractional_padding_statistics(self):
        # Last column has zero weight, despite deliberately extreme values.
        mean = torch.tensor([[[[1., 3., 999.]]]])
        posterior = Posterior(mean, torch.zeros_like(mean), torch.tensor([[[[1., .5, 0.]]]]))
        stats = LatentStats(); stats.update(posterior)
        report = stats.report()
        self.assertAlmostEqual(report["mean"][0], 5 / 3, places=4)
        self.assertEqual(report["posterior_std"], [1.])
        self.assertEqual(report["active_channel_fraction"], 1)

    def test_readiness_not_guaranteed_and_small_samples_flagged(self):
        metrics = dict(activity_f1=.99, notes_f1=.98, velocity_mae=2)
        latent = dict(active_channel_fraction=1.)
        self.assertEqual(readiness(metrics, latent, .98, .1, 1, 8)["status"], "insufficient_evidence")
        self.assertEqual(readiness(metrics, latent, .98, .1, 100, 800)["status"], "promising_not_proven")
        self.assertEqual(readiness(metrics, latent, .2, .98, 100, 800)["status"], "needs_review")

    def test_candidates_unique_groups_and_seeded_order(self):
        import sqlite3
        with sqlite3.connect(":memory:") as db:
            db.execute("CREATE TABLE files(id INTEGER,path TEXT,group_id TEXT)")
            db.executemany("INSERT INTO files VALUES(?,?,?)", [(1, "a.mid", "same"),
                           (2, "b.mid", "same"), (3, "c.mid", "different")])
            data = SimpleNamespace(groups={"a": np.array([1, 2, 3])}, connect=lambda: db)
            first = list(candidates(data, 42))
            self.assertEqual(len(first), 2)
            self.assertEqual(first, list(candidates(data, 42)))

    def test_single_and_batch_checkpoint_cli_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root / "midi"; data.mkdir()
            for i in range(3): source().save(data / f"piece{i}.mid")
            cfg = {k: v for k, v in vars(defaults).items() if k.isupper()}
            cfg.update(DATASETS={"classical_archives": str(data)}, ARIA_METADATA=None,
                       INDEX_PATH=str(root / "index.sqlite"), DATA_SPLIT_RATIOS={"train": 0., "validation": 0., "test": 1.})
            checkpoint = root / "model.pt"
            torch.save(dict(format="local_vae_training_v1", model_config=asdict(self.model.config),
                            model_state_dict=self.model.state_dict(), train_config=cfg, step=123,
                            decoder_calibration=dict(activity_threshold=.95, onset_threshold=.995,
                                note_decoding=vars(NoteDecoding('held', .5, 20)))), checkpoint)
            with redirect_stdout(io.StringIO()):
                build_midi_index(SimpleNamespace(**cfg))
                report = main(["--input", str(data / "piece0.mid"), "--checkpoint", str(checkpoint),
                               "--output-dir", str(root / "single"), "--device", "cpu"])
                batch = evaluate(["--checkpoint", str(checkpoint), "--output-dir", str(root / "batch"),
                                  "--num-files", "2", "--bins-per-file", "2", "--save-examples", "1", "--device", "cpu"])
            self.assertEqual(report["step"], 123)
            self.assertEqual(report['note_decoding']['mode'], 'held')
            self.assertEqual(batch["files"], 2)
            self.assertEqual(batch["bins"], 4)
            self.assertEqual(batch["readiness"]["status"], "insufficient_evidence")
            saved = json.loads((root / "batch/report.json").read_text())
            self.assertEqual(saved["files"], 2)
            self.assertLess((root / "batch/report.json").stat().st_size, 4000)
            decoded = read_performance(root / "single/reconstructed.mid")
            self.assertTrue(any(n.channel == 2 for n in decoded["notes"]))
            with self.assertRaises(FileExistsError):
                from improvnet.inference.local_vae_inference import output_directory
                output_directory(root / "single", "test-")


if __name__ == "__main__":
    unittest.main()
