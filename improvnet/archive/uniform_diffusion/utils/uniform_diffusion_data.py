"""Target-first music windows for shared-backbone uniform diffusion SFT.

Only the existing tokenizer, byte indexes and label mappings are reused. This
does not import or change the masked TwoTower corruption pipeline.
"""

import json
import os
import random
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from improvnet.utils.ar_utils import ProcessData, load_split_byte_index


def sample_target_window(tokens, *, canvas_size, prompt_max, promptless_prob,
                         end_anchored_prob, rng):
    if not tokens or canvas_size < 1 or prompt_max < 0:
        raise ValueError("Need nonempty tokens, a positive canvas and nonnegative prompt limit")
    if not 0 <= promptless_prob <= 1 or not 0 <= end_anchored_prob <= 1:
        raise ValueError("Sampling probabilities must be in [0, 1]")
    target_length = min(canvas_size, len(tokens))
    max_start = len(tokens) - target_length
    end_anchored = rng.random() < end_anchored_prob
    start = (max_start if end_anchored else rng.randint(1, max_start)) if max_start else 0
    prefix = tokens[max(0, start - prompt_max):start]
    if rng.random() < promptless_prob:
        prefix = []
    return prefix, tokens[start:start + target_length]


class UniformDiffusionDataset(Dataset):
    def __init__(self, config, split, processor=None):
        self.config = SimpleNamespace(**{k: v for k, v in vars(config).items() if k.isupper()})
        self.split = split
        self.processor = processor or ProcessData()
        self.epoch = 0
        self.file_handles = {}
        self._handle_pid = os.getpid()
        self.indices = [
            (path, offset) for path in config.JSONL_FILES
            for offset in load_split_byte_index(path, split, config=config)
        ]
        if not self.indices:
            raise ValueError(f"No {split} examples found")

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getstate__(self):
        # Worker-local handles, including when multiprocessing uses spawn.
        return {**self.__dict__, "file_handles": {}}

    def close(self):
        for handle in self.file_handles.values():
            handle.close()
        self.file_handles.clear()

    def __getitem__(self, index):
        # A parent may inspect an example before DataLoader forks workers.
        # Inherited handles otherwise share file offsets across processes.
        if self._handle_pid != os.getpid():
            self.close()
            self._handle_pid = os.getpid()
        path, offset = self.indices[index]
        if path not in self.file_handles:
            self.file_handles[path] = open(path, "rb")
        handle = self.file_handles[path]
        handle.seek(offset)
        entry = json.loads(handle.readline())
        tokens = self.processor.serialized_tokens_to_tokens(entry.get("tokens", []))
        # Stateless per-item randomness makes worker count and resume offsets
        # irrelevant to data augmentation; validation is fixed across epochs.
        epoch = self.epoch if self.split == "train" else 0
        seed = self.config.SEED + 1000003 * epoch + 9176 * index
        rng = random.Random(seed)
        if self.split == "train" and self.config.PITCH_AUGMENTATION:
            saved_rng = random.getstate()
            try:
                random.seed(seed)
                tokens = self.processor.pitch_augmentation(tokens)
            finally:
                random.setstate(saved_rng)
        while tokens and tokens[-1] == "<P>":
            tokens.pop()
        if not tokens:
            tokens = ["<S>", "<E>"]
        elif "<E>" in tokens:
            tokens = tokens[:tokens.index("<E>") + 1]
        else:
            tokens.append("<E>")
        prefix, target = sample_target_window(
            tokens, canvas_size=self.config.MODEL.canvas_size,
            prompt_max=self.config.PROMPT_MAX,
            promptless_prob=self.config.PROMPTLESS_PROB,
            end_anchored_prob=self.config.END_ANCHORED_TARGET_PROB, rng=rng,
        )
        return {
            "prefix": self.processor.tokens_to_tensor(prefix),
            "target": self.processor.tokens_to_tensor(target),
            "genre": self.processor.get_genre_id(entry.get("genre", "unknown")),
            "multi_hot": self.processor.get_instrument_multihot(prefix + target),
        }


class UniformDiffusionCollator:
    def __init__(self, canvas_size, pad_id):
        self.canvas_size = canvas_size
        self.pad_id = pad_id

    def __call__(self, examples):
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        prefix_lengths = torch.tensor([len(e["prefix"]) for e in examples])
        target_lengths = torch.tensor([len(e["target"]) for e in examples])
        if bool(((target_lengths < 1) | (target_lengths > self.canvas_size)).any()):
            raise ValueError("Each target must contain 1..canvas_size tokens")
        clean_lengths = prefix_lengths + target_lengths
        clean = torch.full((len(examples), int(clean_lengths.max())), self.pad_id, dtype=torch.long)
        targets = torch.full((len(examples), self.canvas_size), self.pad_id, dtype=torch.long)
        for i, example in enumerate(examples):
            # No padding hole between prompt and target: AR next-token labels
            # stay contiguous, while decoder visibility ends at prefix_lengths.
            clean[i, :clean_lengths[i]] = torch.cat((example["prefix"], example["target"]))
            targets[i, :target_lengths[i]] = example["target"]
        return {
            "clean_ids": clean,
            "clean_mask": torch.arange(clean.shape[1])[None, :] < clean_lengths[:, None],
            "prefix_lengths": prefix_lengths,
            "target_ids": targets,
            "target_mask": torch.arange(self.canvas_size)[None, :] < target_lengths[:, None],
            "genre": torch.tensor([e["genre"] for e in examples]),
            "multi_hot": torch.stack([e["multi_hot"] for e in examples]),
            # Rigid-length baseline. BLANK/stem augmentation is deliberately
            # not inherited from the previous masked-diffusion formulation.
            "elasticity": torch.zeros(len(examples)),
        }
