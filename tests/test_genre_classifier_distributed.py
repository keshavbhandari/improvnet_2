"""Tiny two-rank CPU training/calibration; no GPU or production jobs."""

from pathlib import Path
import os
import socket
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.multiprocessing as mp

from improvnet.metrics.genre_classifier.audit import audit_dataset
from test_genre_classifier_audit import tiny_records, write_records


def _worker(rank, port, directory, ready):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    from improvnet.metrics.genre_classifier import genre_classifier_config as defaults
    from improvnet.metrics.genre_classifier.genre_classifier_train import train
    torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
    ready[rank].set()
    config = SimpleNamespace(**{key: value for key, value in vars(defaults).items() if key.isupper()})
    config.EMBED_DIM, config.N_HEADS, config.N_KV_HEADS, config.N_LAYERS = 16, 2, 1, 1
    config.MAX_LENGTH, config.DROPOUT = 32, 0.
    config.MANIFEST, config.SAVE_DIR = str(Path(directory) / "manifest.json"), str(Path(directory) / "run")
    config.DEVICE, config.NUM_WORKERS, config.TENSORBOARD = "cpu", 0, False
    config.MIN_LENGTH, config.EVAL_LENGTH = 8, 16
    config.BATCH_SIZE, config.ACCUM_STEPS, config.SAMPLES_PER_EPOCH = 2, 2, 16
    config.VALIDATION_BATCH_SIZE = 3  # Unequal shards: rank 0 does two forwards, rank 1 only one.
    config.N_STEPS, config.WARMUP_STEPS, config.LOG_EVERY, config.VALIDATE_EVERY, config.SAVE_EVERY = 2, 0, 1, 1, 1
    train(config)


class DistributedGenreTests(unittest.TestCase):
    def test_two_rank_accumulation_unequal_validation_and_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = tiny_records()
            from improvnet.metrics.genre_classifier.audit import piece_key, piece_split
            val_row = next(row for row in rows if row["genre"] is not None and piece_split(piece_key(row)) == "validation")
            rows.append(val_row)  # Same piece remains in validation, seven rows total.
            source = Path(directory) / "data.jsonl"
            write_records(source, rows)
            audit_dataset([source], Path(directory) / "manifest", progress_every=0)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            # Forking the unittest parent after earlier backward tests inherits
            # autograd/thread-pool state. A fresh forkserver avoids that, while
            # preloading Torch once instead of separately in every worker.
            mp.set_forkserver_preload(["torch", "improvnet.metrics.genre_classifier.genre_classifier_train"])
            ready = [mp.get_context("forkserver").Event() for _ in range(2)]
            context = mp.start_processes(_worker, args=(port, directory, ready), nprocs=2,
                                         join=False, start_method="forkserver")
            try:
                # Bound the actual test so broken distributed collectives fail,
                # rather than leaving orphan CPU jobs behind.
                import time
                # Imports from the shared environment have taken >3 minutes.
                # Give startup its own budget, then enforce a short deadline
                # on the actual distributed training/validation collectives.
                deadline = time.monotonic() + 600
                dependencies_ready = False
                while time.monotonic() < deadline:
                    if not dependencies_ready and all(event.is_set() for event in ready):
                        dependencies_ready = True
                        deadline = time.monotonic() + 180
                    if context.join(timeout=1):
                        break
                else:
                    self.fail("Distributed classifier smoke test timed out")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=10)
            latest = torch.load(Path(directory) / "run/latest_checkpoint.pt", weights_only=True)
            self.assertEqual(latest["world_size"], 2)
            self.assertEqual(len(latest["rng_states"]), 2)
            self.assertEqual(latest["trained_steps"], 2)
            self.assertEqual(latest["validation"]["count"], 7)
            calibrated = torch.load(Path(directory) / "run/calibrated_checkpoint.pt", weights_only=True)
            self.assertEqual(calibrated["calibration"]["count"], 7)
            self.assertGreater(calibrated["temperature"], 0)


if __name__ == "__main__":
    unittest.main()
