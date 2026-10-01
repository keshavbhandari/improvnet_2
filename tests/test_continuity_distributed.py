"""Global contrastive-gradient equivalence and tiny two-rank CPU training."""

import os
from pathlib import Path
import socket
import tempfile
import time
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from test_continuity import make_manifest, tiny_config


class PairNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 3, bias=False)

    def forward(self, a, b):
        return self.linear(a), self.linear(b)


def worker(rank, port, directory, ready):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    from improvnet.metrics.continuity.objective import contrastive_loss
    from improvnet.metrics.continuity.continuity_train import train
    torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
    ready[rank].set()
    torch.manual_seed(12)
    dist.init_process_group("gloo")
    try:
        reference = PairNetwork()
        model = DDP(PairNetwork())
        model.module.load_state_dict(reference.state_dict())
        inputs = torch.eye(5)[:4]
        targets = inputs + .15 * inputs.roll(1, dims=1)
        a, b = reference(inputs, targets)
        reference_loss, _ = contrastive_loss(a, b, torch.arange(4), temperature=.3)
        reference_loss.backward()
        start = rank * 2
        a, b = model(inputs[start:start+2], targets[start:start+2])
        loss, _ = contrastive_loss(a, b, torch.arange(start, start+2), temperature=.3, distributed=True)
        loss.backward()
        torch.testing.assert_close(model.module.linear.weight.grad, reference.linear.weight.grad, atol=1e-6, rtol=1e-5)
    finally:
        dist.destroy_process_group()
    config = tiny_config(directory)
    config.SAMPLES_PER_EPOCH = 16
    config.VALIDATION_BATCH_SIZE = 2  # 5 rows -> one shard has a singleton tail.
    train(config)


class DistributedContinuityTests(unittest.TestCase):
    def test_global_gradient_and_two_rank_training(self):
        with tempfile.TemporaryDirectory() as directory:
            make_manifest(directory)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            mp.set_forkserver_preload(["torch", "improvnet.metrics.continuity.continuity_train"])
            ready = [mp.get_context("forkserver").Event() for _ in range(2)]
            context = mp.start_processes(worker, args=(port, directory, ready), nprocs=2, join=False, start_method="forkserver")
            try:
                deadline = time.monotonic() + 600
                started = False
                while time.monotonic() < deadline:
                    if not started and all(event.is_set() for event in ready):
                        started, deadline = True, time.monotonic() + 180
                    if context.join(timeout=1):
                        break
                else:
                    self.fail("Distributed continuity test timed out")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=10)
            state = torch.load(Path(directory) / "run/latest_checkpoint.pt", weights_only=True)
            self.assertEqual(state["world_size"], 2)
            self.assertEqual(state["trained_steps"], 2)
            self.assertEqual(len(state["rng_states"]), 2)
            self.assertEqual(state["validation"]["pairs"], 4)


if __name__ == "__main__":
    unittest.main()
