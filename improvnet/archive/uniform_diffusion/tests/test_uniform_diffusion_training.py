"""CPU-only synthetic checks; never open the production checkpoints/datasets."""

from dataclasses import replace
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from improvnet.archive.uniform_diffusion.model import uniform_diffusion_train_config as defaults
from improvnet.archive.uniform_diffusion.model.uniform_diffusion_config import UniformDiffusionConfig
from improvnet.archive.uniform_diffusion.model.uniform_diffusion_model import UniformDiffusionModel
from improvnet.archive.uniform_diffusion.model.uniform_diffusion_train import (
    batch_objective, evaluate, ResumableBatchSampler, summarize, train, validate_config,
)
from improvnet.utils.ar_utils import ProcessData
from improvnet.archive.uniform_diffusion.utils.uniform_diffusion_corruption import reconstruction_losses, uniform_corrupt
from improvnet.archive.uniform_diffusion.utils.uniform_diffusion_data import (
    sample_target_window, UniformDiffusionCollator, UniformDiffusionDataset,
)


def tiny_config(**overrides):
    values = {k: v for k, v in vars(defaults).items() if k.isupper()}
    values.update(
        MODEL=UniformDiffusionConfig(vocab_size=32, embed_dim=32, n_heads=4, n_kv_heads=2,
                                     n_layers=2, num_instruments=3, canvas_size=8, dropout=0.1),
        BATCH_SIZE=2, ACCUM_STEPS=3, VALIDATION_BATCH_SIZE=2,
        NUM_WORKERS=0, N_STEPS=4, WARMUP_STEPS=1, CE_CHUNK_SIZE=3,
        LOG_EVERY=1, SAVE_EVERY=1, VALIDATE_EVERY=2, MAX_VALIDATION_BATCHES=10,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TinyDataset:
    def __init__(self, config, split, processor=None):
        self.split, self.epoch = split, 0
        self.processor = SimpleNamespace(
            tokenizer=SimpleNamespace(vocab_size=32, tok_to_id={"<P>": 2}),
            genres=list(range(4)), INSTRUMENT_CLASSES=list(range(3)),
        )

    def __len__(self):
        return 10 if self.split == "train" else 3

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        offset = self.epoch if self.split == "train" else 0
        return dict(prefix=(torch.arange(index % 4) + offset + 8) % 32,
                    target=(torch.arange(8 - index % 3) + offset + 9) % 32,
                    genre=index % 4, multi_hot=torch.tensor([1., 0., 1.]))


def _distributed_smoke(rank, rendezvous):
    torch.set_num_threads(1)
    torch.manual_seed(11 + rank)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        config = tiny_config()
        model = DDP(UniformDiffusionModel(config.MODEL))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        data = TinyDataset(config, "train")
        batch = UniformDiffusionCollator(8, 2)([data[rank], data[rank + 2]])
        # Shared modules are visited by encoder, detached bootstrap and final
        # decoder. Check both zero-SC and active-SC steps, with accumulation.
        for probability in (0., 1.):
            config.SELF_CONDITIONING_PROB = probability
            optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                with model.no_sync() if micro == 0 else nullcontext():
                    loss, _ = batch_objective(model, batch, config)
                    (loss / 2).backward()
            optimizer.step()
        parameters = torch.cat([p.detach().flatten() for p in model.parameters()])
        reference = parameters.clone()
        dist.broadcast(reference, 0)
        torch.testing.assert_close(parameters, reference, rtol=0, atol=0)
        # Unequal validation shards (2 rows on rank 0, 1 on rank 1).
        val_data = TinyDataset(config, "validation")
        loader = DataLoader(val_data, batch_size=1, sampler=range(rank, len(val_data), 2),
                            collate_fn=UniformDiffusionCollator(8, 2),
                            generator=torch.Generator().manual_seed(8))
        result = evaluate(model, loader, config, torch.device("cpu"), rank)
        assert math.isfinite(result["loss"])
    finally:
        dist.destroy_process_group()


class UniformDiffusionTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.config = tiny_config()
        self.model = UniformDiffusionModel(self.config.MODEL)
        self.collator = UniformDiffusionCollator(8, 2)
        self.batch = self.collator([TinyDataset(self.config, "train")[i] for i in (0, 3)])

    def test_uniform_noise_endpoints_and_full_vocab(self):
        clean = torch.tensor([[1, 2, 3, 2], [1, 0, 1, 0]])
        valid = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
        zero = uniform_corrupt(clean, valid, 4, timesteps=torch.zeros(2))
        self.assertTrue(torch.equal(zero.tokens, clean))
        self.assertFalse(zero.replaced.any())
        one = uniform_corrupt(clean, valid, 4, timesteps=torch.ones(2))
        self.assertTrue(torch.equal(one.replaced, valid))
        self.assertTrue(torch.equal(one.tokens[~valid], clean[~valid]))
        # A vocabulary of one proves random replacement CAN equal the source.
        identity = uniform_corrupt(torch.zeros(2, 4, dtype=torch.long), valid, 1, timesteps=torch.ones(2))
        self.assertTrue(identity.replaced.any())
        self.assertFalse(identity.changed.any())
        many = uniform_corrupt(torch.zeros(1000, 100, dtype=torch.long), torch.ones(1000, 100, dtype=torch.bool), 4,
                               timesteps=torch.ones(1000), generator=torch.Generator().manual_seed(7))
        frequencies = torch.bincount(many.tokens.flatten(), minlength=4) / many.tokens.numel()
        torch.testing.assert_close(frequencies, torch.full((4,), .25), atol=.01, rtol=0)
        mid = uniform_corrupt(torch.zeros(1000, 100, dtype=torch.long), torch.ones(1000, 100, dtype=torch.bool), 4,
                              timesteps=torch.full((1000,), .3))
        self.assertAlmostEqual(mid.replaced.float().mean().item(), .3, delta=.01)
        with self.assertRaises(ValueError):
            uniform_corrupt(clean, valid, 4, timesteps=torch.tensor([float("nan"), 0]))

    def test_target_first_promptless_and_end_anchoring(self):
        kwargs = dict(canvas_size=8, prompt_max=5, promptless_prob=0, end_anchored_prob=1, rng=random.Random(7))
        prefix, target = sample_target_window(list(range(30)), **kwargs)
        self.assertEqual(prefix, list(range(17, 22)))
        self.assertEqual(target, list(range(22, 30)))
        prefix, target = sample_target_window([1, 2, 3], **kwargs)
        self.assertEqual(prefix, [])
        self.assertEqual(target, [1, 2, 3])
        prefix, target = sample_target_window(list(range(30)), **{**kwargs, "promptless_prob": 1})
        self.assertEqual(prefix, [])
        self.assertEqual(len(target), 8)
        starts = {sample_target_window(list(range(30)), **{**kwargs, "end_anchored_prob": 0})[1][0] for _ in range(100)}
        self.assertTrue(any(start % 8 for start in starts))

    def test_collator_contiguous_ar_and_partial_canvas_masks(self):
        self.assertEqual(self.batch["prefix_lengths"].tolist(), [0, 3])
        self.assertEqual(self.batch["clean_mask"].sum(-1).tolist(), [8, 11])
        partial = self.collator([TinyDataset(self.config, "train")[2]])
        self.assertEqual(partial["target_mask"].sum().item(), 6)
        self.assertEqual(partial["target_ids"][0, -2:].tolist(), [2, 2])
        with self.assertRaises(ValueError):
            self.collator([dict(prefix=torch.tensor([]), target=torch.tensor([]), genre=0, multi_hot=torch.zeros(3))])

    def test_dataset_drums_genres_end_and_deterministic_validation(self):
        processor = ProcessData()
        tokens = ["<S>", ["drum", 36], ["onset", 0], ["dur", 100], "<D>", "<E>"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.jsonl"
            path.write_text(json.dumps(dict(tokens=tokens, genre="jazz")) + "\n")
            config = tiny_config(JSONL_FILES=[str(path)], MODEL=replace(self.config.MODEL, canvas_size=256),
                                 END_ANCHORED_TARGET_PROB=1., PROMPTLESS_PROB=0.)
            with patch("improvnet.archive.uniform_diffusion.utils.uniform_diffusion_data.load_split_byte_index", return_value=[0]):
                data = UniformDiffusionDataset(config, "validation", processor)
            example = data[0]
            data.set_epoch(5)
            self.assertTrue(torch.equal(example["target"], data[0]["target"]))
            self.assertEqual(example["genre"], processor.get_genre_id("jazz"))
            self.assertEqual(example["multi_hot"][-1].item(), 1)
            expected = processor.tokens_to_tensor(processor.serialized_tokens_to_tokens(tokens))
            self.assertTrue(torch.equal(example["target"], expected))
            inherited_handle = data.file_handles[str(path)]
            with patch("improvnet.archive.uniform_diffusion.utils.uniform_diffusion_data.os.getpid", return_value=os.getpid() + 1):
                reread = data[0]
            self.assertTrue(inherited_handle.closed)
            self.assertTrue(torch.equal(reread["target"], example["target"]))
            data.close()

    def test_explicit_training_self_conditioning_temperature(self):
        self.model.eval()
        config = tiny_config(SELF_CONDITIONING_PROB=1., SELF_CONDITIONING_TEMPERATURE=.6)
        with patch.object(self.model, "denoise", wraps=self.model.denoise) as denoise:
            batch_objective(self.model, self.batch, config)
        self.assertEqual(denoise.call_args.kwargs["self_conditioning_temperature"], .6)
        for invalid in (0., -1., float("nan"), float("inf")):
            with self.subTest(temperature=invalid), self.assertRaises(ValueError):
                validate_config(tiny_config(SELF_CONDITIONING_TEMPERATURE=invalid))

    def test_loss_all_valid_tokens_chunking_and_padding_gradients(self):
        logits = torch.randn(2, 8, 32, requires_grad=True)
        targets, mask = self.batch["target_ids"], self.batch["target_mask"].clone()
        mask[1, -2:] = False
        actual = reconstruction_losses(logits, targets, mask, chunk_size=3)
        expected = torch.stack([torch.nn.functional.cross_entropy(logits[i][mask[i]], targets[i][mask[i]]) for i in range(2)])
        torch.testing.assert_close(actual, expected)
        grad = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected.sum(), logits)[0]
        torch.testing.assert_close(grad, expected_grad)
        self.assertEqual(grad[~mask].abs().sum().item(), 0)

    def test_training_prefix_view_hides_clean_target_and_sc_bootstrap(self):
        self.model.eval()
        batch = self.batch
        noisy = torch.full_like(batch["target_ids"], 2)  # PAD ID is valid NOISE.
        kwargs = dict(genre=batch["genre"], multi_hot=batch["multi_hot"], elasticity=batch["elasticity"],
                      context_attention_mask=batch["clean_mask"], canvas_attention_mask=batch["target_mask"],
                      prefix_lengths=batch["prefix_lengths"], return_context_logits=True,
                      bootstrap_self_conditioning=True, self_conditioning_mask=torch.tensor([True, False]))
        original = self.model(batch["clean_ids"], noisy, **kwargs)
        altered = batch["clean_ids"].clone()
        for i, length in enumerate(batch["prefix_lengths"]):
            altered[i, length:] = (altered[i, length:] + 7) % 32
        other = self.model(altered, noisy, **kwargs)
        torch.testing.assert_close(original.logits, other.logits)
        self.assertGreater((original.context.logits - other.context.logits).abs().sum().item(), 0)
        with torch.no_grad():
            context = self.model.encode_context(batch["clean_ids"], genre=batch["genre"], multi_hot=batch["multi_hot"],
                                                 elasticity=batch["elasticity"], attention_mask=batch["clean_mask"])
            cache = context.cache.prefix(batch["prefix_lengths"])
            previous = self.model.denoise(noisy, cache=cache, attention_mask=batch["target_mask"])
            manual = self.model.denoise(noisy, cache=cache, attention_mask=batch["target_mask"],
                                       self_conditioning_logits=previous, self_conditioning_mask=torch.tensor([True, False]))
        torch.testing.assert_close(original.logits, manual)

    def test_combined_objective_backward_checkpointed_and_no_sc(self):
        for probability in (0., 1.):
            with self.subTest(probability=probability):
                self.model.zero_grad(set_to_none=True)
                config = tiny_config(SELF_CONDITIONING_PROB=probability)
                loss, stats = batch_objective(self.model, self.batch, config)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertAlmostEqual(summarize(stats)["loss"], summarize(stats)["denoising"] + summarize(stats)["ar"], places=5)
                for name, parameter in self.model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(self.model.genre_emb.weight.grad.abs().sum().item(), 0)

    def test_validation_repeats_without_changing_training_rng(self):
        loader = DataLoader(TinyDataset(self.config, "validation"), batch_size=2, collate_fn=self.collator,
                            generator=torch.Generator().manual_seed(7))
        before = torch.get_rng_state().clone()
        first = evaluate(self.model, loader, self.config, torch.device("cpu"))
        second = evaluate(self.model, loader, self.config, torch.device("cpu"))
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(self.model.training)

    def test_resume_sampler(self):
        sampler = ResumableBatchSampler(list(range(11)), 2, start_batch=3)
        self.assertEqual(list(sampler), [[6, 7], [8, 9]])
        self.assertEqual(len(sampler), 2)

    def test_synthetic_trainer_resume_matches_uninterrupted_across_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ar_path = root / "ar" / "source.pt"
            ar_path.parent.mkdir()
            torch.save({k: v for k, v in self.model.state_dict().items()
                        if not k.startswith(("self_conditioning.", "elasticity_emb."))}, ar_path)
            cfg = tiny_config(AR_MODEL_PATH=str(ar_path), SAVE_DIR=str(root / "uninterrupted"))
            with patch("improvnet.archive.uniform_diffusion.model.uniform_diffusion_train.UniformDiffusionDataset", TinyDataset), \
                 patch("torch.cuda.is_available", return_value=False), \
                 patch.dict("os.environ", {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0", "CHECKPOINT_REQUEST_FILE": ""}):
                train(cfg)
                baseline = torch.load(root / "uninterrupted" / "latest_checkpoint.pt", weights_only=False)
                cfg.SAVE_DIR = str(root / "resumed")
                request = root / "stop-request"
                request.touch()
                with patch.dict("os.environ", {"CHECKPOINT_REQUEST_FILE": str(request)}):
                    train(cfg)
                partial = torch.load(root / "resumed" / "latest_checkpoint.pt", weights_only=False)
                self.assertEqual(partial["step"], 1)
                train(cfg)
                resumed = torch.load(root / "resumed" / "latest_checkpoint.pt", weights_only=False)
                self.assertEqual(resumed["step"], 4)
                self.assertEqual(resumed["epoch"], 2)
                self.assertEqual(resumed["next_batch"], 2)
                for key, value in baseline["model_state_dict"].items():
                    torch.testing.assert_close(resumed["model_state_dict"][key], value, rtol=0, atol=0)
                self.assertEqual(resumed["scheduler_state_dict"], baseline["scheduler_state_dict"])
                self.assertEqual(resumed["best_val"], baseline["best_val"])

    def test_config_protects_source_checkpoint(self):
        config = tiny_config(AR_MODEL_PATH="/tmp/source/source.pt", SAVE_DIR="/tmp/source")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            validate_config(config)

    def test_two_process_cpu_ddp_shared_backward_and_unequal_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_distributed_smoke, args=(str(Path(directory) / "rendezvous"),), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
