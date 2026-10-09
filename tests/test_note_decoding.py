"""Musical boundary regressions for predicted held-note MIDI assembly."""
from types import SimpleNamespace
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from improvnet.tokenizer.hierarchical import roll_notes
from improvnet.utils.note_decoding import NoteDecoding, apply_note_decoding, decode_held_notes
from improvnet.inference.local_vae_inference import Scores
from improvnet.inference.local_vae_note_calibration import load_cached_confidence, release_guard, save_calibration
from improvnet.utils.latent_diffusion import checkpoint_digest


class HeldNoteTests(unittest.TestCase):
    def setUp(self):
        self.a = np.zeros((2, 128, 100), dtype=np.float32)
        self.o = np.zeros_like(self.a)
        self.v = np.full_like(self.a, 80 / 127)
        self.mask = np.ones((2, 100), dtype=bool)

    def decode(self, **kwargs):
        return decode_held_notes(self.a, self.o, self.v, self.mask,
            activity_threshold=.95, onset_threshold=.995, **kwargs)

    def spans(self, result):
        return [(n[5], n[6]) for n in roll_notes(result['velocity'], result['onset'],
                int(self.mask.sum()), 0, 0) if n[3] == 60]

    def held(self):
        self.a[0, 60, 20:] = .999
        self.a[1, 60, :40] = .999
        self.o[0, 60, 20] = .999

    def test_activity_only_cannot_start_except_initial_carry(self):
        self.a[0, 60, 20:40] = .999
        self.a[1, 60, :40] = .999  # Later bin is NOT a new window.
        self.assertEqual(self.spans(self.decode()), [])
        self.a[0, 60, :10] = .999
        self.assertEqual(self.spans(self.decode()), [(0, 10)])
        self.assertEqual(self.spans(self.decode(allow_initial_carry=False)), [])

    def test_hysteresis_and_gap_cross_bin_keep_a_single_held_note(self):
        self.held()
        self.a[0, 60, 50:55] = .7  # Below start, above continuation.
        self.a[0, 60, 99] = .1
        self.a[1, 60, 0] = .1
        self.assertEqual(self.spans(self.decode(continuation_threshold=.5, max_gap_ms=20)), [(20, 140)])

    def test_long_gap_releases_at_first_inactive_frame_without_restart(self):
        self.held()
        self.a[0, 60, 50:53] = .1
        result = self.decode(max_gap_ms=20)
        self.assertEqual(self.spans(result), [(20, 50)])
        self.assertFalse(result['velocity'][1, 60].any())

    def test_trailing_uncertainty_does_not_extend_release(self):
        self.held()
        self.a[1, 60, 38:40] = .1
        self.assertEqual(self.spans(self.decode(max_gap_ms=20)), [(20, 138)])

    def test_true_repeated_onset_preserves_a_short_release(self):
        self.held()
        self.a[0, 60, 50:52] = 0
        self.o[0, 60, 52] = .999
        self.assertEqual(self.spans(self.decode(max_gap_ms=30)), [(20, 50), (52, 140)])

    def test_one_frame_repeated_attacks_and_velocity_are_preserved(self):
        self.a[0, 60, 20:24] = .999
        self.o[0, 60, 20:24] = .999
        self.v[0, 60, 21] = 70 / 127
        result = self.decode()
        self.assertEqual(self.spans(result), [(20, 21), (21, 22), (22, 23), (23, 24)])
        self.assertEqual(int(result['velocity'][0, 60, 21]), 70)

    def test_onset_requires_activity_agreement(self):
        self.o[0, 60, 20] = .999
        self.a[0, 60, 20:40] = .7
        self.assertEqual(self.spans(self.decode()), [])

    def test_padding_is_silent_and_cannot_bridge_or_invent_a_release(self):
        self.held()
        self.mask[1, 40:] = False
        self.a[1, :, 40:] = np.nan
        self.assertEqual(self.spans(self.decode()), [(20, 140)])
        self.assertFalse(self.decode()['velocity'][1, :, 40:].any())

    def test_separate_selected_runs_do_not_borrow_reference_or_gap_state(self):
        self.a[:, 60] = .999
        self.o[0, 60, 0] = .999
        target = SimpleNamespace(velocity=np.zeros((3, 128, 100), dtype=np.uint8),
            onset=np.zeros((3, 128, 100), dtype=np.uint8), valid_mask=np.ones((3, 100), dtype=bool),
            config=SimpleNamespace(fine_ms=10))
        target.velocity[1, 61] = 99  # Unselected reference data is untouched.
        apply_note_decoding(target, np.array([0, 2]), dict(activity=self.a, onset=self.o, velocity=self.v),
            NoteDecoding('held'), activity_threshold=.95, onset_threshold=.995)
        self.assertTrue((target.velocity[1, 61] == 99).all())
        self.assertTrue((target.velocity[2, 60] == 80).all())  # Independent run's initial carry.

    def test_invalid_configuration_and_mask(self):
        for kwargs in (dict(continuation_threshold=.99), dict(max_gap_ms=-1), dict(max_gap_ms=True)):
            with self.assertRaises(ValueError): self.decode(**kwargs)
        with self.assertRaises(ValueError): NoteDecoding('unknown')
        self.mask[0, 50] = False
        with self.assertRaises(ValueError): self.decode()

    def test_release_guard_rejects_perfect_attack_scores_with_truncated_notes(self):
        self.held()
        reference = SimpleNamespace(**self.decode(), valid_mask=self.mask, config=SimpleNamespace(fine_ms=10))
        baseline = Scores(); baseline.notes(reference, reference)
        truncated = SimpleNamespace(velocity=reference.velocity.copy(), onset=reference.onset.copy(),
                                    valid_mask=self.mask, config=reference.config)
        truncated.velocity[0, 60, 80:] = 0
        truncated.velocity[1, 60] = 0
        score = Scores(); score.notes(reference, truncated)
        self.assertEqual(score.report()['notes_f1'], 1)
        self.assertEqual(score.report()['notes_with_offsets_f1'], 0)
        self.assertFalse(release_guard(score, baseline))

    def test_confidence_cache_rejects_test_split_and_wrong_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, digest in [('test', 'expected'), ('validation', 'different')]:
                (root / 'report.json').write_text(json.dumps(dict(split=split, checkpoint_sha256=digest)))
                with self.assertRaisesRegex(ValueError, 'validation data'):
                    load_cached_confidence(root, None, None, 'expected', io.StringIO())

    def test_calibration_save_preserves_weights_step_optimizer_and_original_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); checkpoint = root / 'best_model.pt'
            state = dict(format='local_vae_training_v1', step=92,
                model_state_dict={'weights': torch.tensor([1., 2.])},
                optimizer_state_dict={'momentum': torch.tensor([3., 4.])},
                decoder_calibration=dict(activity_threshold=.95, onset_threshold=.995))
            torch.save(state, checkpoint); digest = checkpoint_digest(checkpoint)
            calibration = dict(activity_threshold=.95, onset_threshold=.985,
                               note_decoding=vars(NoteDecoding('held', .7, 10)))
            update = save_calibration(checkpoint, calibration, root / 'report.json', digest)
            self.assertEqual(checkpoint_digest(update['backup']), digest)
            saved = torch.load(checkpoint, weights_only=False)
            torch.testing.assert_close(saved['model_state_dict']['weights'], state['model_state_dict']['weights'])
            torch.testing.assert_close(saved['optimizer_state_dict']['momentum'], state['optimizer_state_dict']['momentum'])
            self.assertEqual(saved['step'], 92)
            self.assertEqual(saved['decoder_calibration']['note_decoding']['mode'], 'held')
            with self.assertRaisesRegex(RuntimeError, 'changed before'):
                save_calibration(checkpoint, calibration, root / 'report.json', digest)


if __name__ == '__main__':
    unittest.main()
