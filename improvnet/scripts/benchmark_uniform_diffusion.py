"""Bounded GPU memory smoke test; no checkpoint writes or dataset mutation.

Optimizer updates affect only this disposable, AR-initialized model. Synthetic
full-length examples stress the configured maximum context, not average data.
"""

import argparse
import gc
import json
import math
import time
from types import SimpleNamespace

import torch

from improvnet.model import uniform_diffusion_train_config as defaults
from improvnet.model.uniform_diffusion_model import UniformDiffusionModel
from improvnet.model.uniform_diffusion_train import amp_context, batch_objective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--memory-limit-gib", type=float, default=48.)
    parser.add_argument("--headroom-gib", type=float, default=12.)
    args = parser.parse_args()
    if args.steps < 2 or min(args.batch_sizes) < 1:
        parser.error("Use positive batches and at least two steps to include optimizer state")
    if not math.isfinite(args.memory_limit_gib) or args.memory_limit_gib <= 0 or not math.isfinite(args.headroom_gib) or args.headroom_gib < 12:
        parser.error("Use a finite positive memory limit and at least 12 GiB shared-GPU headroom")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    gib = 1024 ** 3
    budget = min(args.memory_limit_gib * gib, free - args.headroom_gib * gib)
    if budget < 20 * gib:
        raise RuntimeError("Insufficient shared-GPU headroom for this full-model test")
    torch.cuda.set_per_process_memory_fraction(budget / total, device)
    print(json.dumps(dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                          free_gib=free / gib, allocator_budget_gib=budget / gib)), flush=True)
    config = SimpleNamespace(**{k: v for k, v in vars(defaults).items() if k.isupper()})
    config.SELF_CONDITIONING_PROB = 1.  # Stress the extra forward on every example.
    torch.manual_seed(7)
    model = UniformDiffusionModel(config.MODEL)
    state = torch.load(config.AR_MODEL_PATH, map_location="cpu", weights_only=False, mmap=True)
    model.initialize_from_ar(state)
    del state
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, betas=config.BETAS, weight_decay=config.WEIGHT_DECAY)
    for batch_size in args.batch_sizes:
        for mode in ("full", "padded"):
            free_now, _ = torch.cuda.mem_get_info()
            if free_now < args.headroom_gib * gib:
                print("Stopping: shared GPU headroom changed", flush=True)
                return
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            length = config.PROMPT_MAX + config.MODEL.canvas_size
            clean = torch.randint(config.MODEL.vocab_size, (batch_size, length), device=device)
            clean_mask = torch.ones_like(clean, dtype=torch.bool)
            prefix_lengths = torch.full((batch_size,), config.PROMPT_MAX, device=device, dtype=torch.long)
            if mode == "padded":
                clean_mask[0, -1] = False
                prefix_lengths[0] -= 1
            batch = dict(clean_ids=clean, clean_mask=clean_mask, prefix_lengths=prefix_lengths,
                         target_ids=clean[:, -config.MODEL.canvas_size:].clone(),
                         target_mask=torch.ones(batch_size, config.MODEL.canvas_size, device=device, dtype=torch.bool),
                         genre=torch.ones(batch_size, device=device, dtype=torch.long),
                         multi_hot=torch.ones(batch_size, config.MODEL.num_instruments, device=device),
                         elasticity=torch.zeros(batch_size, device=device))
            # Targets in the shorter row precede padding, just like the collator.
            for row in range(batch_size):
                start = int(prefix_lengths[row])
                batch["target_ids"][row] = clean[row, start:start + config.MODEL.canvas_size]
            started = time.monotonic()
            try:
                model.train()
                for step in range(args.steps):
                    optimizer.zero_grad(set_to_none=True)
                    with amp_context(device):
                        loss, _ = batch_objective(model, batch, config)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite benchmark loss")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("Non-finite benchmark gradient")
                    optimizer.step()
                    del loss
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                result = dict(batch=batch_size, mode=mode, seconds_per_step=(time.monotonic() - started) / args.steps,
                              train_peak_allocated_gib=torch.cuda.max_memory_allocated() / gib,
                              train_peak_reserved_gib=torch.cuda.max_memory_reserved() / gib)
                torch.cuda.reset_peak_memory_stats()
                model.eval()
                with torch.inference_mode(), amp_context(device):
                    val_loss, _ = batch_objective(model, batch, config)
                if not torch.isfinite(val_loss):
                    raise FloatingPointError("Non-finite validation loss")
                torch.cuda.synchronize()
                result["validation_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / gib
                print(json.dumps(result), flush=True)
                del val_loss, batch, clean, clean_mask, prefix_lengths
            except torch.OutOfMemoryError as error:
                print(json.dumps(dict(batch=batch_size, mode=mode, status="allocator_budget_exceeded", detail=str(error))), flush=True)
                return


if __name__ == "__main__":
    main()
