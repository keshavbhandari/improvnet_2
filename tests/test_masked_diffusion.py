"""Focused CPU checks for the new tokenizer-based diffusion path."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import pickle
import random
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.multiprocessing as mp

from improvnet.model import masked_diffusion_config as defaults
from improvnet.model.masked_diffusion_model import MaskedDiffusionConfig, MaskedDiffusionModel
from improvnet.model.masked_diffusion_train import train, resume_plan
from improvnet.inference.masked_diffusion_inference import load_model, transfer, main as inference_main
from improvnet.utils.ar_utils import ProcessData
from improvnet.utils.masked_diffusion import corrupt, masked_loss, prepare_canvas, denoise, batch_objective
from improvnet.utils.masked_diffusion_data import (
    note_units, rhythm_features, split_windows, expand_slots, blank_budget,
    MaskedDiffusionDataset, GenreSampler,
)


def source():
    return ["<S>"] + [token for onset in (0, 200, 300, 600) for token in
        (("instrument", "Acoustic Piano"), ("note", 60, 80), ("onset", onset), ("dur", 100))] + ["<E>"]


def tiny_config(root):
    values = {k: v for k, v in vars(defaults).items() if k.isupper()}
    values.update(SAVE_DIR=str(root / "run"), JSONL_FILES=[str(root / "data.jsonl")],
        MIDI_FILEPATH_EXCLUDE_SUBSTRINGS={}, DATA_SPLIT_RATIOS={"train": .75, "validation": .25, "test": 0.},
        EMBED_DIM=32, N_HEADS=4, N_KV_HEADS=2, N_LAYERS=2, SEQ_LEN=64,
        BATCH_SIZE=2, ACCUM_STEPS=2, NUM_WORKERS=0, N_STEPS=4, WARMUP_STEPS=1,
        DECAY_STEPS=1, LOG_EVERY=1, VAL_EVERY=2, CHECKPOINT_EVERY=1)
    path, offsets = root / "data.jsonl", []
    with path.open("wb") as handle:
        for index in range(24):
            offsets.append(handle.tell())
            handle.write((json.dumps(dict(midi_filepath=f"{index}.mid", genre=("classical", "jazz", "blues", "unknown")[index % 4], tokens=source())) + "\n").encode())
    with (root / "data_index.pkl").open("wb") as handle:
        pickle.dump(offsets, handle)
    return SimpleNamespace(**values)


def distributed_worker(rank, config, port):
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    with redirect_stdout(io.StringIO()):
        train(config, force_cpu=True)


class ScriptedModel(torch.nn.Module):
    """Always proposes a valid note, except for requested all-blank slots."""
    def __init__(self, processor, blank_starts=(), seq_len=64):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.processor, self.blank_starts = processor, set(blank_starts)
        self.config = SimpleNamespace(seq_len=seq_len)

    def forward(self, tokens, roles, *args):
        vocab = self.processor.tokenizer
        note = next(t for t in vocab.note_tokens if t[1] == 64 and t[2] >= 80)
        values = {1: ("instrument", "Acoustic Piano"), 2: note, 3: ("onset", 100), 4: ("dur", 200)}
        output = torch.full((*tokens.shape, vocab.vocab_size), -10., device=tokens.device)
        for b in range(len(tokens)):
            for i in range(tokens.shape[1]):
                role = int(roles[b, i])
                token = "<BLANK>" if i in self.blank_starts else values.get(role, "<S>")
                output[b, i, vocab.tok_to_id[token]] = 10.
        return output


class MaskedDiffusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.processor = ProcessData()

    def test_blank_budget_groups_and_window_bounds(self):
        units = note_units(source())
        self.assertEqual(blank_budget(units, 0), 0)
        self.assertEqual(blank_budget(units, .5), 2)
        tokens, roles, slots, added = expand_slots(units, 1., random.Random(42))
        self.assertEqual(len(added), 4)
        for start in added:
            self.assertEqual(tokens[start:start + 4], ["<BLANK>"] * 4)
            self.assertEqual(roles[start:start + 4], [1, 2, 3, 4])
        for window in split_windows(units, 17, 1):
            expanded, _, _, _ = expand_slots(window, 1, random.Random(42))
            self.assertLessEqual(len(expanded), 17)
        with self.assertRaises(ValueError):
            note_units(source()[:-3])

    def test_corruption_masks_structure_excludes_padding_and_keeps_blanks_consecutive(self):
        sequence = source()[:-1] + ["<T>", "<D>", "<E>"]
        tokens, roles, _, added = expand_slots(note_units(sequence), 1., random.Random(1))
        tokens.extend(["<P>", "<P>"]); roles.extend([0, 0])
        target = self.processor.tokens_to_tensor(tokens)[None]
        roles = torch.tensor(roles)[None]; valid = torch.ones_like(target, dtype=torch.bool)
        valid[0, -2] = False
        noisy, masked, levels = corrupt(target, roles, valid, levels=torch.tensor([.5]), generator=torch.Generator().manual_seed(4))
        self.assertFalse(masked[0, -2:].any())
        for start in added:
            self.assertTrue(masked[0, start:start + 4].all() or not masked[0, start:start + 4].any())
        noisy, masked, _ = corrupt(target, roles, valid, levels=torch.ones(1))
        self.assertTrue(masked[valid & (target != 2)].all())
        self.assertFalse(masked[0, -2:].any())
        for token in ("<S>", "<T>", "<D>", "<E>"):
            self.assertTrue(masked[target == self.processor.tokenizer.tok_to_id[token]].all())
        self.assertTrue((noisy[masked] == 5).all())
        _, grouped, _ = corrupt(target, roles, valid, levels=torch.tensor([.5]),
            whole_note=torch.ones(1, dtype=torch.bool), generator=torch.Generator().manual_seed(2))
        for start in (roles[0] == 1).nonzero().flatten():
            self.assertTrue(grouped[0, start:start + 4].all() or not grouped[0, start:start + 4].any())

    def test_loss_uses_inverse_rate_and_all_eligible_denominator(self):
        logits = torch.zeros(1, 4, 2, requires_grad=True)
        target = torch.zeros(1, 4, dtype=torch.long)
        masked = torch.tensor([[True, True, False, False]])
        loss = masked_loss(logits, target, masked, torch.ones_like(masked), torch.ones_like(target), torch.tensor([.25]))
        self.assertAlmostEqual(float(loss.detach()), 2 * math_log2(), places=6)
        loss.backward()
        self.assertFalse(logits.grad[:, 2:].any())

    def test_structural_tokens_contribute_to_loss_but_padding_does_not(self):
        target = self.processor.tokens_to_tensor(["<T>", "<D>", "<E>", "<P>"])[None]
        logits = torch.zeros(1, 4, self.processor.tokenizer.vocab_size, requires_grad=True)
        valid = torch.ones_like(target, dtype=torch.bool)
        roles = torch.zeros_like(target)
        loss = masked_loss(logits, target, valid, valid, roles, torch.ones(1))
        import math
        self.assertAlmostEqual(float(loss.detach()), math.log(self.processor.tokenizer.vocab_size), places=5)
        loss.backward()
        self.assertTrue((logits.grad[:, :3].abs().sum(-1) > 0).all())
        self.assertFalse(logits.grad[:, 3:].any())

    def test_rhythm_is_tempo_pitch_velocity_and_order_invariant(self):
        original = source()
        changed = [t if not isinstance(t, tuple) else
            ("note", 72, 40) if t[0] == "note" else
            (t[0], t[1] * 3) if t[0] in ("onset", "dur") else t for t in original]
        torch.testing.assert_close(rhythm_features(original, self.processor.tokenizer), rhythm_features(changed, self.processor.tokenizer))
        units = note_units(original)
        reordered = [token for unit in units[:1] + list(reversed(units[1:-1])) + units[-1:] for token in unit]
        torch.testing.assert_close(rhythm_features(original, self.processor.tokenizer), rhythm_features(reordered, self.processor.tokenizer))
        different = [t if not isinstance(t, tuple) or t[0] != "dur" else ("dur", 400) for t in original]
        self.assertFalse(torch.equal(rhythm_features(original, self.processor.tokenizer), rhythm_features(different, self.processor.tokenizer)))

    def test_timing_and_pitch_velocity_edits_preserve_other_attributes(self):
        model = ScriptedModel(self.processor)
        for strategy, changed in (("timing", {2, 3}), ("pitch-velocity", {1})):
            output, report = transfer(model, self.processor, source(), strategy=strategy,
                                      percentage=.5, elasticity=0, temperature=0, steps=2)
            before, after = note_units(source()), note_units(output)
            for old, new in zip(before, after):
                for index, token in enumerate(old):
                    if len(old) != 4 or index not in changed:
                        self.assertEqual(token, new[index])
            self.assertEqual(report["source_notes"], report["output_notes"])
            self.assertEqual(sum(old != new for old, new in zip(before, after)), 2)

    def test_extra_slots_can_be_blank_or_valid_notes(self):
        canvas = prepare_canvas(note_units(source()), self.processor, strategy="timing", percentage=0, elasticity=.5, seed=2)
        model = ScriptedModel(self.processor, blank_starts=canvas["added"][:1])
        output, report = denoise(model, canvas, self.processor, genre=1,
            rhythm=torch.zeros(14), multi_hot=self.processor.get_instrument_multihot(source()),
            elasticity=.5, temperature=0, steps=2)
        self.assertEqual(report["added_notes"], 1)
        self.assertEqual(report["removed_notes"], 0)
        self.assertNotIn("<BLANK>", output)
        self.assertEqual(sum(len(u) == 4 for u in note_units(output)), 5)
        midi = self.processor.tokens_to_midi(output)
        with tempfile.TemporaryDirectory() as tmp:
            midi.save(str(Path(tmp) / "result.mid"))

    def test_complete_note_deletion_is_bounded_by_elasticity(self):
        canvas = prepare_canvas(note_units(source()), self.processor, strategy="notes", percentage=1, elasticity=.5, seed=3)
        model = ScriptedModel(self.processor, blank_starts=canvas["allow_blank"])
        output, report = denoise(model, canvas, self.processor, genre=1, rhythm=torch.zeros(14),
            multi_hot=self.processor.get_instrument_multihot(source()), elasticity=.5, temperature=0, steps=3)
        self.assertEqual(report["removed_notes"], 2)
        self.assertEqual(sum(len(u) == 4 for u in note_units(output)), 2)

    def test_model_bidirectional_padding_gradients_and_conditions(self):
        c = MaskedDiffusionConfig(vocab_size=self.processor.tokenizer.vocab_size, embed_dim=32,
            n_heads=4, n_kv_heads=2, n_layers=2, seq_len=64)
        model = MaskedDiffusionModel(c)
        target = self.processor.tokens_to_tensor(source())[None]
        roles = torch.tensor([[0] + [1, 2, 3, 4] * 4 + [0]])
        batch = dict(target=target, roles=roles, valid=torch.ones_like(target, dtype=torch.bool),
            genre=torch.ones(1, dtype=torch.long), multi_hot=torch.zeros(1, 41), rhythm=torch.zeros(1, 14), elasticity=torch.tensor([.5]))
        loss, stats = batch_objective(model, batch, generator=torch.Generator().manual_seed(1))
        loss.backward()
        self.assertTrue(torch.isfinite(stats).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        model.eval()
        def forward(b):
            return model(b["target"], b["roles"], b["valid"], b["genre"], b["multi_hot"], b["rhythm"], b["elasticity"])
        before = forward(batch)
        other = deepcopy(batch); other["target"][0, -3] = self.processor.tokenizer.tok_to_id[("onset", 100)]
        self.assertFalse(torch.equal(before[0, 1], forward(other)[0, 1]))
        for key in ("rhythm", "elasticity"):
            other = deepcopy(batch); other[key] = torch.ones_like(other[key])
            self.assertFalse(torch.equal(before, forward(other)))
        batch["valid"][0, -4:] = False
        before = forward(batch)
        other = deepcopy(batch); other["target"][0, -4:] = c.vocab_size + 100
        torch.testing.assert_close(before[:, :-4], forward(other)[:, :-4])

    def test_jazz_blues_weights_and_distributed_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = tiny_config(Path(tmp)); data = MaskedDiffusionDataset(c, "train")
            for i, (path, offset) in enumerate(data.indices):
                with open(path, "rb") as handle:
                    handle.seek(offset); entry = json.loads(handle.readline())
                self.assertEqual(float(data.weights[i]), 4 if entry["genre"] in ("jazz", "blues") else 1)
            validation = MaskedDiffusionDataset(c, "validation")
            self.assertIsNone(validation.weights)
            a, b = GenreSampler(data.weights, 2, 42, 0, 2), GenreSampler(data.weights, 2, 42, 1, 2)
            expected = torch.multinomial(data.weights, a.size * 2, replacement=True, generator=torch.Generator().manual_seed(42)).tolist()
            self.assertEqual(list(a), list(zip(expected[::2], range(0, len(expected), 2))))
            self.assertEqual(list(b), list(zip(expected[1::2], range(1, len(expected), 2))))
            self.assertLessEqual(len(data[0]["target"]), c.SEQ_LEN)
            sampler = GenreSampler(torch.tensor([1., 4.], dtype=torch.double).repeat(10000), 1, 42)
            values = list(sampler)
            ratio = sum(v[0] % 2 == 1 for v in values) / sum(v[0] % 2 == 0 for v in values)
            self.assertAlmostEqual(ratio, 4, delta=.25)

    def test_synthetic_checkpoint_resume_and_inference_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); c = tiny_config(root)
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
            model, actual = load_model(Path(c.SAVE_DIR) / "latest_checkpoint.pt", "cpu")
            self.assertEqual(actual["step"], 4)
            for key in expected["model_state_dict"]:
                torch.testing.assert_close(actual["model_state_dict"][key], expected["model_state_dict"][key], rtol=0, atol=0)
            input_path, donor_path, output_path = root / "source.mid", root / "donor.mid", root / "output.mid"
            self.processor.tokens_to_midi(source()).save(str(input_path))
            donor = [t if not isinstance(t, tuple) or t[0] != "dur" else ("dur", 400) for t in source()]
            self.processor.tokens_to_midi(donor).save(str(donor_path))
            with redirect_stdout(io.StringIO()):
                inference_main(["--model", str(Path(c.SAVE_DIR) / "latest_checkpoint.pt"),
                    "--input", str(input_path), "--output", str(output_path), "--rhythm-reference", str(donor_path),
                    "--strategy", "timing", "--mask-percentage", ".5", "--elasticity", ".5", "--steps", "2", "--device", "cpu"])
            self.assertTrue(output_path.exists())
            report = json.loads(output_path.with_suffix('.json').read_text())
            self.assertEqual(report["rhythm_source"], "reference")
            donor_tokens = self.processor.midi_to_tokens(self.processor.read_midi(str(donor_path)))
            self.assertEqual(report["rhythm_features"], rhythm_features(donor_tokens, self.processor.tokenizer).tolist())
            c.ELASTICITY_MAX = .5
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                with redirect_stdout(io.StringIO()):
                    train(c, force_cpu=True)

    def test_distributed_checkpoint_and_uneven_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); c = tiny_config(root)
            c.N_STEPS = 2
            c.DATA_SPLIT_RATIOS = {"train": .79, "validation": .21, "test": 0.}
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
            mp.spawn(distributed_worker, args=(c, port), nprocs=2, join=True)
            state = torch.load(Path(c.SAVE_DIR) / "latest_checkpoint.pt", weights_only=False)
            self.assertEqual(state["world_size"], 2)
            self.assertEqual(len(state["rng"]), 2)
            self.assertEqual(state["step"], 2)

    def test_resume_same_folder_from_one_rank_to_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); c = tiny_config(root)
            request = root / "request"; request.touch()
            with patch.dict(os.environ, CHECKPOINT_REQUEST_FILE=str(request)), redirect_stdout(io.StringIO()):
                train(c, force_cpu=True)
            request.unlink()
            path = Path(c.SAVE_DIR) / "latest_checkpoint.pt"
            before = torch.load(path, weights_only=False)
            self.assertEqual(before["step"], 1)
            self.assertEqual(before["world_size"], 1)
            plan = resume_plan(before, c, before["model_config"], 2)
            self.assertTrue(plan["changed"])
            self.assertEqual(plan["next_batch"] * c.BATCH_SIZE * 2,
                             before["next_batch"] * c.BATCH_SIZE)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
            mp.spawn(distributed_worker, args=(c, port), nprocs=2, join=True)
            after = torch.load(path, weights_only=False)
            self.assertEqual(after["step"], c.N_STEPS)
            self.assertEqual(after["world_size"], 2)
            self.assertEqual(len(after["rng"]), 2)
            for values in after["optimizer_state_dict"]["state"].values():
                self.assertEqual(int(values["step"]), c.N_STEPS)
            changed = deepcopy(c); changed.BATCH_SIZE = 3; changed.ACCUM_STEPS = 1
            batch_plan = resume_plan(before, changed, before["model_config"], 2)
            self.assertTrue(batch_plan["changed"])
            self.assertLess(batch_plan["repeated_examples"], 6)
            changed.ELASTICITY_MAX = .25
            with self.assertRaisesRegex(ValueError, "ELASTICITY_MAX"):
                resume_plan(before, changed, before["model_config"], 2)


def math_log2():
    import math
    return math.log(2)


if __name__ == "__main__":
    unittest.main()
