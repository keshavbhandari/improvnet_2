"""CPU synthetic VAE/data/index/resume tests; no production data/checkpoints."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import socket
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mido
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from improvnet.model import local_vae_config as defaults
from improvnet.model.local_vae_model import LocalVAE, LocalVAEConfig, Posterior, reconstruction_losses
from improvnet.model.local_vae_train import (
    CosineLRSchedule, batch_objective, configure_run_directory, lr_factor, main,
    model_config, summarize, train, validate_config,
)
from improvnet.tokenizer.hierarchical import GridConfig, HierarchicalTokenizer, read_performance
from improvnet.utils.local_vae_data import (
    LocalVAEDataset, bin_tensors, build_midi_index, collate_bins, genres_from, transpose_bin,
)


def midi(notes, programs=None, pedal=()):
    result = mido.MidiFile(ticks_per_beat=1000)
    events = [(0, mido.Message("program_change", channel=c, program=p)) for c, p in (programs or {0: 0}).items()]
    for channel, pitch, velocity, start, end in notes:
        events.extend(((round(start * 2000), mido.Message("note_on", channel=channel, note=pitch, velocity=velocity)),
                       (round(end * 2000), mido.Message("note_off", channel=channel, note=pitch))))
    events.extend((round(time * 2000), mido.Message("control_change", control=64, value=value)) for time, value in pedal)
    track, previous = mido.MidiTrack(), 0
    for tick, message in sorted(events, key=lambda e: e[0]):
        track.append(message.copy(time=tick - previous))
        previous = tick
    track.append(mido.MetaMessage("end_of_track"))
    result.tracks.append(track)
    return result


def config(**values):
    c = {k: v for k, v in vars(defaults).items() if k.isupper()}
    c.update(BASE_CHANNELS=4, CHANNEL_MULTIPLIERS=(1, 2), LATENT_CHANNELS=2, RES_BLOCKS=1,
             NUM_WORKERS=0, SAMPLES_PER_EPOCH=4, VALIDATION_SAMPLES=4, BATCH_SIZE=2,
             VALIDATION_BATCH_SIZE=2, ACCUM_STEPS=2, N_STEPS=4, LR=1e-3, MIN_LR=1e-4,
             WARMUP_STEPS=1, KL_WARMUP_STEPS=2, LOG_EVERY=1, VALIDATE_EVERY=2,
             SAVE_EVERY=1, MAX_VALIDATION_BATCHES=2, ARIA_METADATA=None, GIGAMIDI_METADATA=None,
             DATA_SPLIT_RATIOS={"train": .7, "validation": .3, "test": 0.}, LOG_DATA_ERRORS=True)
    c.update(values)
    return SimpleNamespace(**c)


def fixture(root):
    directory = root / "classical"
    directory.mkdir()
    for i in range(35):
        midi([(0, 60 + i % 5, 60 + i % 40, .1, .3), (0, 64, 95, .99, 1.08)], pedal=[(.1, 90), (1.1, 0)]).save(directory / f"piece{i:02d}.mid")
    c = config(DATASETS={"classical_archives": str(directory)}, INDEX_PATH=str(root / "index.sqlite"),
               SAVE_DIR=str(root / "run"), RUN_NAME=None, RUNS_DIR=str(root / "experiments"),
               TENSORBOARD_ROOT=str(root / "tensorboard"))
    with redirect_stdout(io.StringIO()):
        build_midi_index(c)
    return c


def sample_batch():
    tokenizer = HierarchicalTokenizer()
    piece = tokenizer.encode(midi([(0, 60, 85, .99, 1.08), (2, 72, 95, .95, 1.1)], programs={0: 0, 2: 40},
                                  pedal=[(.5, 73), (1.1, 0)]))
    items = [bin_tensors(piece, i) for i in (0, 1)]
    for item in items:
        item["retries"] = torch.tensor(0)
    return piece, collate_bins(items)


def ddp_check(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, world_size=2, rank=rank)
    try:
        c = config()
        model = DDP(LocalVAE(model_config(c)))
        _, batch = sample_batch()
        # Even ranks/batches with no pedal must use every trainable parameter.
        batch["pedal_enabled"][:] = False
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for _ in range(2):
            loss, _ = batch_objective(model, batch, c, 0)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        vector = torch.cat([p.detach().flatten() for p in model.parameters()])
        other = vector.clone()
        dist.broadcast(other, 0)
        torch.testing.assert_close(vector, other)
    finally:
        dist.destroy_process_group()


def ddp_train_check(rank, c, port):
    torch.set_num_threads(1)
    with patch.dict(os.environ, RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank),
                    MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port)), redirect_stdout(io.StringIO()):
        train(c, force_cpu=True)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_fine_patch_and_spatial_latent_shapes(self):
        _, batch = sample_batch()
        cfg = LocalVAEConfig(base_channels=4, res_blocks=1)
        model = LocalVAE(cfg)
        output = model(batch["roll"], batch["pedal"], batch["valid_mask"])
        self.assertEqual(output.latent.shape, (2, 8, 32, 25))
        self.assertEqual(output.reconstruction.velocity.shape, (2, 128, 100))
        self.assertEqual(output.reconstruction.pedal.shape, (2, 1, 100))
        losses = reconstruction_losses(output, batch)
        sum(v.mean() for v in losses.values()).backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))

    def test_odd_length_padding_latent_weights_and_crop(self):
        model = LocalVAE(LocalVAEConfig(frames_per_bin=101, base_channels=4, res_blocks=1))
        mask = torch.arange(101)[None] < 99
        output = model(torch.zeros(1, 2, 128, 101), torch.zeros(1, 1, 101), mask)
        self.assertEqual(output.latent.shape, (1, 8, 32, 26))
        self.assertEqual(output.reconstruction.velocity.shape[-1], 101)
        self.assertAlmostEqual(output.posterior.weights[0, 0, 0, 24].item(), .75)
        self.assertFalse(output.latent[..., 25].any())

    def test_deterministic_mode_and_seeded_posterior(self):
        _, batch = sample_batch()
        model = LocalVAE(model_config(config()))
        a = model(batch["roll"], batch["pedal"], batch["valid_mask"], sample=False)
        b = model(batch["roll"], batch["pedal"], batch["valid_mask"], sample=False)
        torch.testing.assert_close(a.latent, b.latent)
        torch.testing.assert_close(a.posterior.sample(torch.Generator().manual_seed(1)), a.posterior.sample(torch.Generator().manual_seed(1)))
        self.assertFalse(torch.equal(a.posterior.sample(), a.posterior.mean))

    def test_padding_values_do_not_affect_encoding_or_losses(self):
        _, batch = sample_batch()
        model = LocalVAE(model_config(config()))
        output = model(batch["roll"], batch["pedal"], batch["valid_mask"], sample=False)
        broken = deepcopy(batch)
        broken["roll"].masked_fill_(~broken["valid_mask"][:, None, None], float("nan"))
        broken["pedal"].masked_fill_(~broken["valid_mask"][:, None], float("nan"))
        recovered = model(broken["roll"], broken["pedal"], broken["valid_mask"], sample=False)
        torch.testing.assert_close(output.latent, recovered.latent)
        reference = reconstruction_losses(output, batch)
        r = replace(output.reconstruction, **{k: v.detach().clone() for k, v in vars(output.reconstruction).items()})
        for tensor in (r.activity_logits, r.onset_logits, r.velocity):
            tensor.masked_fill_(~batch["valid_mask"][:, None], float("nan"))
        r.pedal.masked_fill_(~batch["valid_mask"][:, None], float("nan"))
        edited = reconstruction_losses(replace(output, reconstruction=r), broken)
        for key in reference:
            torch.testing.assert_close(reference[key], edited[key])

    def test_pedal_loss_disabled_and_no_pad_reveals(self):
        _, batch = sample_batch()
        model = LocalVAE(model_config(config()))
        batch["pedal_enabled"][:] = False
        output = model(batch["roll"], batch["pedal"], batch["valid_mask"])
        self.assertFalse(reconstruction_losses(output, batch)["pedal"].any())
        arrays = output.reconstruction.tensors(batch["valid_mask"], batch["pedal_enabled"])
        self.assertFalse(arrays["pedal"].any())
        self.assertEqual(arrays["velocity"].dtype, torch.uint8)
        self.assertFalse(arrays["velocity"][1, :, 11:].any())
        self.assertFalse(arrays["onset"][1, :, 11:].any())

    def test_kl_standard_normal_and_empty_latents(self):
        mean, logvar = torch.zeros(2, 3, 4, 5), torch.zeros(2, 3, 4, 5)
        p = Posterior(mean, logvar, torch.ones(2, 1, 1, 5))
        self.assertFalse(p.kl().any())
        p.mean[:] = 1
        torch.testing.assert_close(p.kl(), torch.full((2,), .5))

    def test_silent_patch_has_finite_gradients_and_invalid_mask_rejected(self):
        _, batch = sample_batch()
        batch["roll"][:] = 0
        model = LocalVAE(model_config(config()))
        loss, stats = batch_objective(model, batch, config(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(np.isfinite(v) for v in summarize(stats).values()))
        batch["valid_mask"][0, 1] = False
        with self.assertRaisesRegex(ValueError, "trailing"):
            model(batch["roll"], batch["pedal"], batch["valid_mask"])

    def test_latent_only_decode_and_model_config_validation(self):
        model = LocalVAE(model_config(config()))
        reconstruction = model.decode(torch.zeros(1, *model.config.latent_shape))
        self.assertEqual(reconstruction.velocity.shape, (1, 128, 100))
        for args in (dict(base_channels=0), dict(frames_per_bin=0), dict(channel_multipliers=(1,))):
            with self.assertRaises(ValueError):
                LocalVAEConfig(**args)
        with self.assertRaises(ValueError):
            model.decode(torch.zeros(1, 2, 128, 100))

    def test_distributed_backward_with_disabled_pedal(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(ddp_check, args=(str(Path(root) / "rendezvous"),), nprocs=2, join=True)


class DataTests(unittest.TestCase):
    def test_bin_tensors_keep_onsets_carries_masks_and_no_pedal_broadcast(self):
        piece, batch = sample_batch()
        self.assertEqual(batch["roll"].shape, (2, 2, 128, 100))
        self.assertEqual(batch["pedal"].shape, (2, 1, 100))
        self.assertEqual(batch["event_positions"][1, 0].tolist(), [-5, 10])
        self.assertTrue(batch["event_mask"].all())
        self.assertEqual(batch["roll"][1, 1, 60, 0], 0)  # Carry, no fabricated attack.
        self.assertAlmostEqual(batch["previous_velocity"][1, 60].item(), 85 / 127)
        self.assertFalse(batch["valid_mask"][1, 11:].any())
        torch.testing.assert_close(batch["coarse_velocity"], batch["roll"][:, 0].amax(-1))
        self.assertEqual(bin_tensors(piece, 0)["events"].shape, (1, 11))

    def test_pitch_augmentation_transposes_context_not_drums_and_no_wrap(self):
        piece, _ = sample_batch()
        item = bin_tensors(piece, 1)
        transpose_bin(item, 2)
        self.assertTrue(item["roll"][0, 62].any())
        self.assertEqual(item["events"][0, 4], 74)
        self.assertEqual(item["coarse_onsets"][62], 0)
        self.assertAlmostEqual(item["previous_velocity"][62].item(), 85 / 127)
        drum = HierarchicalTokenizer().encode(midi([(9, 36, 90, 0, .1)]))
        item = bin_tensors(drum, 0)
        before = item["roll"].clone()
        transpose_bin(item, 5)
        torch.testing.assert_close(before, item["roll"])
        with self.assertRaises(ValueError):
            transpose_bin(bin_tensors(piece, 1), 100)

    def test_index_genres_grouped_segments_variants_and_contemporary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = {name: str(root / name) for name in ("gigamidi", "aria", "pijama", "doug", "classical_archives", "maestro", "symphonynet")}
            for p in datasets.values():
                Path(p).mkdir()
            source = midi([(0, 60, 80, 0, .1)])
            for name in datasets:
                source.save(Path(datasets[name]) / "example.mid")
            for segment in (0, 1):
                source.save(Path(datasets["aria"]) / f"123_{segment}.mid")
            for subset in ("drums-only", "all-instruments"):
                folder = Path(datasets["gigamidi"]) / subset
                folder.mkdir()
                source.save(folder / "abcdef.mid")
            for folder in ("classical", "contemporary"):
                path = Path(datasets["symphonynet"]) / folder
                path.mkdir()
                source.save(path / "piece.mid")
            # These are archive bookkeeping, not music, even with a MIDI suffix.
            macos = Path(datasets["gigamidi"]) / "__MACOSX"
            macos.mkdir()
            (macos / "._piece.mid").write_bytes(b"resource fork")
            (Path(datasets["gigamidi"]) / "._other.mid").write_bytes(b"resource fork")
            hidden = Path(datasets["gigamidi"]) / ".metadata"
            hidden.mkdir()
            (hidden / "sidecar.mid").write_bytes(b"resource fork")
            aria = root / "aria_metadata.json"
            aria.write_text(json.dumps({"123": {"metadata": {"genre": "blues"}}}))
            c = config(DATASETS=datasets, ARIA_METADATA=str(aria), INDEX_PATH=str(root / "index.sqlite"))
            with redirect_stdout(io.StringIO()):
                report = build_midi_index(c)
            self.assertFalse(report["gigamidi_genres_available"])
            with sqlite3.connect(c.INDEX_PATH) as db:
                rows = list(db.execute("SELECT path,genre,split FROM files"))
            self.assertEqual(len(rows), 13)
            aria_rows = [r for r in rows if "123_" in r[0]]
            self.assertEqual({r[1] for r in aria_rows}, {"blues"})
            self.assertEqual(len({r[2] for r in aria_rows}), 1)
            self.assertEqual(len({r[2] for r in rows if "abcdef" in r[0]}), 1)
            self.assertEqual([r[1] for r in rows if "/contemporary/" in r[0]], ["unknown"])
            self.assertEqual([r[1] for r in rows if "/symphonynet/classical/" in r[0]], ["classical"])
            self.assertEqual([r[1] for r in rows if "/pijama/" in r[0]], ["jazz"])
            with self.assertRaises(FileExistsError):
                build_midi_index(c)

    def test_genre_parsing_and_optional_giga_csv(self):
        self.assertEqual(genres_from("['jazz', 'blues']"), ["blues", "jazz"])
        self.assertEqual(genres_from('{"genre": ["Classical"]}'), ["classical"])
        self.assertEqual(genres_from("unknown"), [])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "midi").mkdir()
            midi([(0, 60, 80, 0, .1)]).save(root / "midi" / "abc.mid")
            csv_path = root / "genres.csv"
            csv_path.write_text('md5,genre\nabc,jazz\n')
            c = config(DATASETS={"gigamidi": str(root / "midi")}, GIGAMIDI_METADATA=str(csv_path), INDEX_PATH=str(root / "index.sqlite"))
            with redirect_stdout(io.StringIO()):
                report = build_midi_index(c)
            self.assertEqual(report["genres"]["gigamidi"], {"jazz": 1})

    def test_dataset_sampling_reproducible_across_workers_and_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = fixture(Path(tmp))
            data = LocalVAEDataset(c)
            a, b = data[0], data[0]
            torch.testing.assert_close(a["roll"], b["roll"])
            self.assertEqual(a["metadata"], b["metadata"])
            val = LocalVAEDataset(c, "validation")
            a = val[0]
            val.set_epoch(100)
            torch.testing.assert_close(a["roll"], val[0]["roll"])
            data.set_epoch(1)
            self.assertTrue(any(not torch.equal(data[i]["roll"], LocalVAEDataset(c)[i]["roll"]) for i in range(4)))
            loader = torch.utils.data.DataLoader(val, batch_size=2, num_workers=2, collate_fn=collate_bins)
            batch = next(iter(loader))
            torch.testing.assert_close(batch["roll"][0], val[0]["roll"])
            self.assertLessEqual(data.cache_bytes, c.CACHE_MB_PER_WORKER * 1024**2)

    def test_bad_midi_retries_bounded_and_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "bad"
            folder.mkdir()
            for i in range(30):
                (folder / f"bad{i}.mid").write_bytes(b"not a MIDI")
            c = config(DATASETS={"classical_archives": str(folder)}, INDEX_PATH=str(root / "index.sqlite"), MAX_DATA_RETRIES=2)
            with redirect_stdout(io.StringIO()):
                build_midi_index(c)
            data = LocalVAEDataset(c, error_dir=root / "errors")
            with self.assertRaisesRegex(RuntimeError, "2 tries"):
                data[0]
            self.assertTrue(list((root / "errors").glob("*.jsonl")))

    def test_malformed_key_signature_recovers_in_dataloader_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = fixture(root)
            # Corrupt the first deterministic sample with the exact invalid
            # (-1 flats, mode 255) metadata from the reported failure.
            bad_path = Path(LocalVAEDataset(c)[0]["metadata"]["path"])
            source = midi([(0, 60, 80, 0, .1)])
            source.tracks[0].insert(0, mido.UnknownMetaMessage(type_byte=0x59, data=(255, 255), time=0))
            source.save(bad_path)
            with self.assertRaises(mido.KeySignatureError):
                mido.MidiFile(bad_path)
            data = LocalVAEDataset(c, error_dir=root / "errors")
            loader = torch.utils.data.DataLoader(data, batch_size=2, num_workers=1, collate_fn=collate_bins)
            batch = next(iter(loader))
            self.assertEqual(batch["retries"][0].item(), 0)
            self.assertEqual(batch["metadata"][0]["path"], str(bad_path))
            self.assertTrue(batch["roll"][0].any())
            self.assertFalse((root / "errors").exists())

    def test_invalid_pitch_retries_remain_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = fixture(root)
            c.MAX_DATA_RETRIES = 2
            source = midi([(0, 60, 80, 0, .1)])
            buffer = io.BytesIO()
            source.save(file=buffer)
            raw = buffer.getvalue().replace(b"\x90\x3c\x50", b"\x90\xfb\x50")
            for path in (root / "classical").glob("*.mid"):
                path.write_bytes(raw)
            data = LocalVAEDataset(c, error_dir=root / "errors")
            with self.assertRaisesRegex(RuntimeError, "2 tries"):
                data[0]
            records = [json.loads(line) for log in (root / "errors").glob("*.jsonl") for line in log.read_text().splitlines()]
            self.assertTrue(records)
            self.assertTrue(all("data byte" in r["error"] for r in records))

    def test_training_recovers_dangling_note_without_retry_or_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = fixture(root)
            c.LOG_DATA_ERRORS = False
            path = Path(LocalVAEDataset(c)[0]["metadata"]["path"])
            source = mido.MidiFile(path)
            source.tracks[0].insert(-1, mido.Message("note_on", note=100, velocity=80, time=100))
            source.save(path)
            with self.assertRaisesRegex(ValueError, "dangling"):
                read_performance(path)
            data = LocalVAEDataset(c, error_dir=root / "errors")
            batch = next(iter(torch.utils.data.DataLoader(data, batch_size=2, num_workers=1, collate_fn=collate_bins)))
            self.assertEqual(batch["retries"][0].item(), 0)
            self.assertEqual(batch["metadata"][0]["path"], str(path))
            self.assertFalse((root / "errors").exists())

    def test_error_logging_can_be_disabled_even_for_unrecoverable_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = fixture(root)
            c.LOG_DATA_ERRORS = False
            c.MAX_DATA_RETRIES = 2
            for path in (root / "classical").glob("*.mid"):
                path.write_bytes(b"not a MIDI")
            data = LocalVAEDataset(c, error_dir=root / "errors")
            with self.assertRaisesRegex(RuntimeError, "2 tries"):
                data[0]
            self.assertFalse((root / "errors").exists())

    def test_gigamidi_drum_subset_is_fixed_per_split_and_keeps_other_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "giga"
            source = midi([(0, 60, 80, 0, .1)])
            for folder, count in (("drums-only", 200), ("no-drums", 30), ("all-instruments-with-drums", 20)):
                target = directory / folder
                target.mkdir(parents=True)
                for i in range(count):
                    source.save(target / f"{folder}_{i}.mid")
            c = config(DATASETS={"gigamidi": str(directory)}, INDEX_PATH=str(root / "index.sqlite"),
                       DATA_SPLIT_RATIOS={"train": .6, "validation": .2, "test": .2},
                       GIGAMIDI_DRUM_ONLY_FRACTION=.05)
            with redirect_stdout(io.StringIO()):
                build_midi_index(c)
            for split in ("train", "validation", "test"):
                with self.subTest(split=split), sqlite3.connect(c.INDEX_PATH) as db:
                    rows = list(db.execute("SELECT id,path FROM files WHERE split=?", (split,)))
                    drums = {i for i, path in rows if "/drums-only/" in path}
                    other = {i for i, path in rows if "/drums-only/" not in path}
                    data = LocalVAEDataset(c, split)
                    retained = set(data.groups["gigamidi"].tolist())
                    self.assertTrue(other.issubset(retained))
                    self.assertEqual(len(retained & drums), int(len(drums) * .05))
                    self.assertEqual(data.drum_subset["total"], len(drums))
                    data.set_epoch(100)
                    np.testing.assert_array_equal(data.groups["gigamidi"], LocalVAEDataset(c, split).groups["gigamidi"])
                    c.GIGAMIDI_DRUM_ONLY_FRACTION = 1.
                    self.assertEqual(set(LocalVAEDataset(c, split).groups["gigamidi"].tolist()), other | drums)
                    c.GIGAMIDI_DRUM_ONLY_FRACTION = 0.
                    self.assertEqual(set(LocalVAEDataset(c, split).groups["gigamidi"].tolist()), other)
                    c.GIGAMIDI_DRUM_ONLY_FRACTION = .05


class TrainingTests(unittest.TestCase):
    def test_extended_cosine_schedule_preserves_lr_and_round_trips(self):
        c = config(N_STEPS=50000, WARMUP_STEPS=1000, LR=1e-4, MIN_LR=1e-5)
        curve = CosineLRSchedule(c)
        for step in (0, 999, 1000, 14000, 50000):
            self.assertEqual(curve(step), lr_factor(step, c))
        restored_factor = curve(14000)
        curve.extend(100000, 14000, restored_factor)
        self.assertEqual(curve(14000), restored_factor)
        self.assertLess(curve(14001), restored_factor)
        self.assertGreater(curve(50000), c.MIN_LR / c.LR)
        self.assertEqual(curve(100000), c.MIN_LR / c.LR)
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=c.LR)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, curve)
        saved = deepcopy(scheduler.state_dict())
        other_curve = CosineLRSchedule(config(N_STEPS=100000))
        other = torch.optim.lr_scheduler.LambdaLR(optimizer, other_curve)
        other.load_state_dict(saved)
        for step in (14000, 50000, 100000):
            self.assertEqual(other_curve(step), curve(step))
        # A later extension starts at the current curve, not the original peak.
        factor = other_curve(50000)
        other_curve.extend(150000, 50000, factor)
        self.assertEqual(other_curve(50000), factor)
        self.assertEqual(other_curve(150000), c.MIN_LR / c.LR)

    def test_extension_during_warmup_does_not_restart_it(self):
        c = config(N_STEPS=50000, WARMUP_STEPS=1000)
        curve = CosineLRSchedule(c)
        curve.extend(100000, 300, curve(300))
        self.assertEqual(curve(301), 302 / 1000)
        self.assertEqual(curve(1000), 1.)
        self.assertEqual(curve(100000), c.MIN_LR / c.LR)

    def test_longer_run_resumes_legacy_checkpoint_without_resetting_data_or_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = fixture(root)
            c.SAMPLES_PER_EPOCH, c.ACCUM_STEPS, c.VALIDATE_EVERY = 16, 1, 100
            request = root / "checkpoint_request"
            request.touch()
            with patch.dict(os.environ, CHECKPOINT_REQUEST_FILE=str(request)), redirect_stdout(io.StringIO()):
                train(c, force_cpu=True)
            path = Path(c.SAVE_DIR) / "latest_checkpoint.pt"
            saved = torch.load(path, weights_only=False)
            self.assertEqual((saved["step"], saved["epoch"], saved["next_batch"]), (1, 0, 1))
            # Old trainers saved a function lambda, without callable state.
            saved["scheduler_state_dict"]["lr_lambdas"] = [None]
            saved["best_val"] = .12345
            torch.save(saved, path)
            old_lr = saved["optimizer_state_dict"]["param_groups"][0]["lr"]
            c.N_STEPS = 8
            output = io.StringIO()
            with patch.dict(os.environ, CHECKPOINT_REQUEST_FILE=str(request)), redirect_stdout(output):
                train(c, force_cpu=True)
            self.assertIn("Extended N_STEPS", output.getvalue())
            self.assertNotIn("non-exact continuation", output.getvalue())
            actual = torch.load(path, weights_only=False)
            self.assertEqual((actual["step"], actual["epoch"], actual["next_batch"]), (2, 0, 2))
            self.assertEqual(actual["best_val"], .12345)
            self.assertEqual(actual["train_config"]["N_STEPS"], 8)
            self.assertTrue(all(int(v["step"]) == 2 for v in actual["optimizer_state_dict"]["state"].values()))
            curve = actual["scheduler_state_dict"]["lr_lambdas"][0]
            self.assertEqual(curve["anchor_factor"], old_lr / c.LR)
            self.assertEqual((curve["anchor_step"], curve["total_steps"]), (1, 8))
            # The extension is checkpointed and survives another ordinary resume.
            with patch.dict(os.environ, CHECKPOINT_REQUEST_FILE=str(request)), redirect_stdout(io.StringIO()):
                train(c, force_cpu=True)
            again = torch.load(path, weights_only=False)
            self.assertEqual((again["step"], again["epoch"], again["next_batch"]), (3, 0, 3))
            self.assertEqual(again["scheduler_state_dict"]["lr_lambdas"][0], curve)
            before = path.read_bytes()
            c.N_STEPS = 7
            with self.assertRaisesRegex(ValueError, "not reducing"):
                train(c, force_cpu=True, allow_resume_changes=True)
            self.assertEqual(path.read_bytes(), before)

    def test_named_run_paths_are_isolated_and_selection_has_no_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = config(SAVE_DIR=str(root / "baseline"), RUNS_DIR=str(root / "experiments"), RUN_NAME=None,
                       TENSORBOARD_ROOT=str(root / "tensorboard"))
            configure_run_directory(c)
            self.assertEqual(c.SAVE_DIR, str(root / "baseline"))
            c.RUN_NAME = "vae176"
            configure_run_directory(c)
            self.assertEqual(c.SAVE_DIR, str(root / "experiments" / "vae176"))
            self.assertEqual(c.TENSORBOARD_DIR, str(root / "tensorboard" / "vae176"))
            self.assertFalse((root / "experiments").exists())
            for name in ("", ".", "..", "../baseline", "/absolute", "nested/name", "space name"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "single name"):
                    configure_run_directory(config(RUN_NAME=name))
            with self.assertRaisesRegex(ValueError, "reserved"):
                configure_run_directory(config(RUN_NAME="tensorboard"))

    def test_cli_named_run_width_and_explicit_path(self):
        for args, name in ((["--run-name", "vae176", "--base-channels", "176"], "vae176"),
                           (["--save-dir", "/tmp/explicit-vae"], None)):
            with self.subTest(args=args), patch("sys.argv", ["local_vae_train", *args]), \
                    patch("improvnet.model.local_vae_train.train") as mocked:
                main()
                c = mocked.call_args.args[0]
                self.assertEqual(c.RUN_NAME, name)
                if name:
                    self.assertEqual(c.BASE_CHANNELS, 176)
                    self.assertEqual(c.SAVE_DIR, str(Path(defaults.RUNS_DIR) / name))
                else:
                    self.assertEqual(c.SAVE_DIR, "/tmp/explicit-vae")
        with patch("sys.argv", ["local_vae_train", "--run-name", "x", "--save-dir", "/tmp/y"]), \
                redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            main()

    def test_new_named_run_preserves_baseline_and_has_separate_checkpoints_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = fixture(root)
            c.N_STEPS = 1
            with redirect_stdout(io.StringIO()):
                train(c, force_cpu=True)
            baseline = Path(c.SAVE_DIR)
            before = {p.name: p.read_bytes() for p in baseline.glob("*.pt")}
            c.RUNS_DIR, c.RUN_NAME = str(root / "experiments"), "comparison"
            configure_run_directory(c)
            with redirect_stdout(io.StringIO()):
                train(c, force_cpu=True)
            for name, content in before.items():
                self.assertEqual((baseline / name).read_bytes(), content)
                self.assertTrue((Path(c.SAVE_DIR) / name).is_file())
            self.assertTrue((baseline / "tensorboard").is_dir())
            self.assertTrue((Path(c.TENSORBOARD_DIR)).is_dir())
            self.assertFalse((Path(c.SAVE_DIR) / "tensorboard").exists())
            checkpoint = Path(c.SAVE_DIR) / "latest_checkpoint.pt"
            saved = torch.load(checkpoint, weights_only=False)
            self.assertIn("decoder_calibration", saved)
            from improvnet.inference.local_vae_inference import load_model, resolve_thresholds
            loaded, _, _, _ = load_model(checkpoint, torch.device("cpu"))
            args = SimpleNamespace(activity_threshold=None, onset_threshold=None)
            resolve_thresholds(args, loaded)
            self.assertEqual(args.activity_threshold, saved["decoder_calibration"]["activity_threshold"])
            self.assertEqual(args.onset_threshold, saved["decoder_calibration"]["onset_threshold"])

    def test_distributed_training_with_uneven_validation_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = fixture(Path(tmp))
            c.N_STEPS, c.SAMPLES_PER_EPOCH, c.VALIDATION_SAMPLES = 2, 8, 5
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            mp.spawn(ddp_train_check, args=(c, port), nprocs=2, join=True)
            state = torch.load(Path(c.SAVE_DIR) / "latest_checkpoint.pt", weights_only=False)
            self.assertEqual(state["step"], 2)
            self.assertEqual(state["world_size"], 2)
            self.assertEqual(len(state["rng"]), 2)
            self.assertTrue(np.isfinite(state["best_val"]))
            self.assertEqual(state["decoder_calibration"]["branches"], ["clean", "jitter"])

    def test_exact_resume_matches_uninterrupted_training_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = fixture(root)
            with redirect_stdout(io.StringIO()):
                train(c, force_cpu=True)
            expected = torch.load(Path(c.SAVE_DIR) / "latest_checkpoint.pt", weights_only=False)
            c.SAVE_DIR = str(root / "resumed")
            request = root / "checkpoint_request"
            request.touch()
            with patch.dict(os.environ, CHECKPOINT_REQUEST_FILE=str(request)), redirect_stdout(io.StringIO()):
                train(c, force_cpu=True)
            request.unlink()
            # Existing checkpoints predate the optional error-log switch.
            path = Path(c.SAVE_DIR) / "latest_checkpoint.pt"
            legacy_state = torch.load(path, weights_only=False)
            legacy_state["train_config"].pop("LOG_DATA_ERRORS")
            legacy_state["train_config"].pop("RUNS_DIR")
            legacy_state["train_config"].pop("RUN_NAME")
            torch.save(legacy_state, path)
            c.LOG_DATA_ERRORS = False
            with redirect_stdout(io.StringIO()):
                train(c, force_cpu=True)
            actual = torch.load(Path(c.SAVE_DIR) / "latest_checkpoint.pt", weights_only=False)
            self.assertEqual(actual["step"], 4)
            self.assertEqual(actual["epoch"], expected["epoch"])
            self.assertEqual(actual["next_batch"], expected["next_batch"])
            self.assertEqual(actual["scheduler_state_dict"], expected["scheduler_state_dict"])
            for key in actual["model_state_dict"]:
                torch.testing.assert_close(actual["model_state_dict"][key], expected["model_state_dict"][key], rtol=0, atol=0)
            self.assertTrue((Path(c.SAVE_DIR) / "tensorboard").is_dir())

    def test_training_wont_overwrite_unrelated_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = fixture(Path(tmp))
            directory = Path(c.SAVE_DIR)
            directory.mkdir()
            path = directory / "latest_checkpoint.pt"
            torch.save({"format": "autoregressive", "sentinel": 42}, path)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "unrelated"):
                train(c, force_cpu=True)
            self.assertEqual(before, path.read_bytes())

    def test_opt_in_resume_keeps_optimizer_and_step_after_batch_and_pool_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = fixture(root)
            c.N_STEPS, c.SAMPLES_PER_EPOCH, c.ACCUM_STEPS = 2, 8, 1
            request = root / "checkpoint_request"
            request.touch()
            with patch.dict(os.environ, CHECKPOINT_REQUEST_FILE=str(request)), redirect_stdout(io.StringIO()):
                train(c, force_cpu=True)
            request.unlink()
            checkpoint = Path(c.SAVE_DIR) / "latest_checkpoint.pt"
            before = checkpoint.read_bytes()
            saved = torch.load(checkpoint, weights_only=False)
            self.assertEqual((saved["step"], saved["epoch"], saved["next_batch"]), (1, 0, 1))
            c.BATCH_SIZE, c.GIGAMIDI_DRUM_ONLY_FRACTION = 8, .1
            with self.assertRaisesRegex(ValueError, "allow-resume-changes"):
                train(c, force_cpu=True)
            self.assertEqual(before, checkpoint.read_bytes())
            output = io.StringIO()
            with redirect_stdout(output):
                train(c, force_cpu=True, allow_resume_changes=True)
            self.assertIn("non-exact continuation", output.getvalue())
            actual = torch.load(checkpoint, weights_only=False)
            self.assertEqual((actual["step"], actual["epoch"], actual["next_batch"]), (2, 2, 0))
            self.assertEqual(actual["scheduler_state_dict"]["last_epoch"], 2)
            self.assertTrue(all(int(v["step"]) == 2 for v in actual["optimizer_state_dict"]["state"].values()))
            self.assertEqual(actual["train_config"]["BATCH_SIZE"], 8)
            self.assertEqual(actual["train_config"]["GIGAMIDI_DRUM_ONLY_FRACTION"], .1)
            # The new checkpoint can subsequently resume without the opt-in.
            with redirect_stdout(io.StringIO()):
                train(c, force_cpu=True)

    def test_resume_metric_thresholds_allowed_without_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = fixture(Path(tmp))
            with redirect_stdout(io.StringIO()):
                train(c, force_cpu=True)
            checkpoint = Path(c.SAVE_DIR) / "latest_checkpoint.pt"
            before = checkpoint.read_bytes()
            c.ONSET_THRESHOLD, c.ACTIVITY_THRESHOLD = .6, .7
            output = io.StringIO()
            with redirect_stdout(output):
                train(c, force_cpu=True)
            self.assertNotIn("non-exact continuation", output.getvalue())
            # Finished run: no extra optimization or checkpoint overwrite.
            self.assertEqual(before, checkpoint.read_bytes())

    def test_resume_opt_in_accepts_loss_weights_and_records_new_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = fixture(root)
            c.N_STEPS, c.SAMPLES_PER_EPOCH, c.ACCUM_STEPS = 2, 8, 1
            request = root / "checkpoint_request"
            request.touch()
            with patch.dict(os.environ, CHECKPOINT_REQUEST_FILE=str(request)), redirect_stdout(io.StringIO()):
                train(c, force_cpu=True)
            request.unlink()
            checkpoint = Path(c.SAVE_DIR) / "latest_checkpoint.pt"
            before = checkpoint.read_bytes()
            c.ONSET_WEIGHT, c.VELOCITY_WEIGHT, c.KL_WEIGHT = 2., 5., .0002
            with self.assertRaisesRegex(ValueError, "allow-resume-changes"):
                train(c, force_cpu=True)
            self.assertEqual(before, checkpoint.read_bytes())
            with redirect_stdout(io.StringIO()):
                train(c, force_cpu=True, allow_resume_changes=True)
            state = torch.load(checkpoint, weights_only=False)
            self.assertEqual(state["step"], 2)
            for key in ("ONSET_WEIGHT", "VELOCITY_WEIGHT", "KL_WEIGHT"):
                self.assertEqual(state["train_config"][key], getattr(c, key))
            self.assertTrue(all(int(v["step"]) == 2 for v in state["optimizer_state_dict"]["state"].values()))

    def test_resume_opt_in_still_rejects_architecture_and_optimizer_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = fixture(Path(tmp))
            with redirect_stdout(io.StringIO()):
                train(c, force_cpu=True)
            checkpoint = Path(c.SAVE_DIR) / "latest_checkpoint.pt"
            before = checkpoint.read_bytes()
            c.LR *= 2
            with self.assertRaisesRegex(ValueError, "LR"):
                train(c, force_cpu=True, allow_resume_changes=True)
            c.LR /= 2
            c.BASE_CHANNELS *= 2
            with self.assertRaisesRegex(ValueError, "architecture"):
                train(c, force_cpu=True, allow_resume_changes=True)
            self.assertEqual(before, checkpoint.read_bytes())

    def test_resume_rejects_replaced_index_without_overwriting_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = fixture(Path(tmp))
            with redirect_stdout(io.StringIO()):
                train(c, force_cpu=True)
            checkpoint = Path(c.SAVE_DIR) / "latest_checkpoint.pt"
            before = checkpoint.read_bytes()
            with sqlite3.connect(c.INDEX_PATH) as db:
                db.execute("UPDATE files SET genre='unknown' WHERE id=1")
            with self.assertRaisesRegex(ValueError, "MIDI index changed"):
                train(c, force_cpu=True)
            self.assertEqual(checkpoint.read_bytes(), before)

    def test_config_is_values_only_and_validated(self):
        self.assertFalse(hasattr(defaults, "MODEL"))
        for args in (dict(ACCUM_STEPS=0), dict(FINE_MS=0), dict(KL_WEIGHT=-1), dict(ACTIVITY_THRESHOLD=1),
                     dict(GIGAMIDI_DRUM_ONLY_FRACTION=-.1), dict(GIGAMIDI_DRUM_ONLY_FRACTION=1.1),
                     dict(GIGAMIDI_DRUM_ONLY_FRACTION=float("nan"))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_config(config(**args))


if __name__ == "__main__":
    unittest.main()
