"""On-the-fly task construction over indexed performance JSONL records."""
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from improvnet.model import melody_config as cfg
from improvnet.utils.ar_utils import ProcessData
from improvnet.utils.melody import (Note, MelodyTokenizer, TASKS, corrupt_melody,
                                   make_prompt, mask_timing, melody_voice, notes_to_tokens,
                                   sample_strength, window)


def prepare_example(record, rng, config=cfg, task=None, strength=None):
    tok = MelodyTokenizer()
    # Older prepared files inherited the tokenizer's noncanonical piano fallback.
    notes = [Note('Acoustic Piano' if n[0] == 'piano' else n[0], *n[1:])
             for n in record['notes']]
    melody = melody_voice([notes[i] for i in record['melody_indices']])
    melody_ids = {id(n) for n in melody}
    melody_indices = {i for i, n in enumerate(notes) if id(n) in melody_ids}
    task = task or rng.choices(TASKS, weights=config.TASK_WEIGHTS)[0]
    if task == 'harmonize' and not record['harmony_windows']:
        raise ValueError('No aligned accompaniment is available for this record')
    candidates = record['harmony_windows'] if task == 'harmonize' else record['windows']
    start = rng.choice(candidates) * config.WINDOW_MS
    end = start + config.WINDOW_MS
    # Paired transposition changes context, input and target together.
    if record.get('_augment'):
        shift = rng.randint(-5, 5)
        shift = max(-min(n.pitch for n in notes), min(127 - max(n.pitch for n in notes), shift))
        by_id = {id(n): replace(n, pitch=n.pitch + shift) for n in notes}
        melody = [by_id[id(n)] for n in melody]
        notes = [by_id[id(n)] for n in notes]
    core = window(melody, start, end)
    strength = sample_strength(rng) if strength is None else strength
    target_notes = core
    if task == 'refine':
        source = corrupt_melody(core, strength, rng, tok, config.MAX_CORRUPTIONS)
    elif task == 'rhythm':
        source = mask_timing(core, strength, rng, tok)
    elif task == 'harmonize':
        target_notes = window(notes, start, end)
        harmony = window([n for i, n in enumerate(notes) if i not in melody_indices], start, end)
        retained = rng.sample(harmony, len(harmony) - round(strength * len(harmony)))
        source = notes_to_tokens(core + retained, tok)
    else:
        raise ValueError(f'Unknown task {task}')
    span = config.CONTEXT_WINDOWS * config.WINDOW_MS
    left_context = notes if task == 'harmonize' and rng.random() < .5 else melody
    right_context = notes if task == 'harmonize' and rng.random() < .5 else melody
    left = notes_to_tokens(window(left_context, max(0, start - span), start), tok) if start else []
    right = notes_to_tokens(window(right_context, end, end + span), tok)
    target = notes_to_tokens(target_notes, tok)[1:]  # <S> is supplied in the prompt.
    prompt = make_prompt(task, source, left, right)
    # Reserve the AR model's two continuous conditioning positions.
    while len(prompt) + len(target) + 2 > config.SEQ_LEN and (left or right):
        if len(left) >= len(right):
            left = []
        else:
            right = []
        prompt = make_prompt(task, source, left, right)
    if len(prompt) + len(target) + 2 > config.SEQ_LEN:
        raise ValueError('Example exceeds context length; shorten WINDOW_MS and prepare data again')
    ids = [tok.tok_to_id[t] for t in prompt + target]
    labels = [config.IGNORE_INDEX] * len(prompt) + [tok.tok_to_id[t] for t in target]
    return {'input': torch.tensor(ids[:-1]), 'target': torch.tensor(labels[1:]),
            'genre': torch.tensor(config.GENRES.index(record['genre'])),
            'multi_hot': ProcessData().get_instrument_multihot(notes_to_tokens(target_notes, tok)),
            'strength': torch.tensor(strength, dtype=torch.float32), 'task': TASKS.index(task)}


class MelodyDataset(Dataset):
    def __init__(self, data_file, split, config=cfg):
        self.path, self.split, self.config = Path(data_file), split, config
        self.epoch = 0
        self.entries = json.loads(self.path.with_suffix('.index.json').read_text())
        if self.entries['window_ms'] != config.WINDOW_MS:
            raise ValueError('Prepared WINDOW_MS differs from config; prepare data again')
        if self.entries.get('skyline_tolerance_ms', 50) != config.SKYLINE_TOLERANCE_MS:
            raise ValueError('Prepared skyline tolerance differs from config; prepare data again')
        if self.entries['tokenizer'] != MelodyTokenizer().signature():
            raise ValueError('Prepared tokenizer differs from current tokenizer')
        self.rows = [r for r in self.entries['rows'] if r['split'] == split]
        if not self.rows:
            raise ValueError(f'No {split} records in {self.path}; run melody_prepare first')
        self.groups = {task: defaultdict(lambda: defaultdict(list)) for task in TASKS}
        for row in self.rows:
            for task in TASKS:
                if task == 'rhythm' and row['source'] in ('wikifonia', 'curated'):
                    continue  # Notated timing is not performed swing supervision.
                if task != 'harmonize' or row['has_harmony']:
                    self.groups[task][row['genre']][row['source']].append(row['offset'])

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, idx):
        # Stable across worker scheduling and resume; a new epoch resamples edits.
        rng = random.Random(self.config.DATA_SPLIT_SEED + idx * 1000003 + (
            self.epoch * 9176 if self.split == 'train' else 0))
        tasks = [t for t in TASKS if self.groups[t]]
        task = rng.choices(tasks, [self.config.TASK_WEIGHTS[TASKS.index(t)] for t in tasks])[0]
        genres = self.groups[task]
        sources = genres[rng.choice(sorted(genres))]
        offsets = sources[rng.choice(sorted(sources))]
        with self.path.open('rb') as handle:
            handle.seek(rng.choice(offsets))
            record = json.loads(handle.readline())
        record['_augment'] = self.split == 'train'
        return prepare_example(record, rng, self.config, task=task)


def collate_examples(batch):
    size = max(len(x['input']) for x in batch)
    inputs = torch.full((len(batch), size), cfg.PAD_ID, dtype=torch.long)
    targets = torch.full_like(inputs, cfg.IGNORE_INDEX)
    for i, x in enumerate(batch):
        inputs[i, :len(x['input'])] = x['input']
        targets[i, :len(x['target'])] = x['target']
    return {'input': inputs, 'target': targets,
            **{key: torch.stack([x[key] for x in batch]) for key in ('genre', 'multi_hot', 'strength')},
            'task': torch.tensor([x['task'] for x in batch])}


def build_dataloader(split, batch_size, distributed=False, config=cfg):
    dataset = MelodyDataset(config.DATA_FILE, split, config)
    # DistributedSampler also in single-process mode makes resume order explicit.
    sampler = DistributedSampler(dataset, num_replicas=None if distributed else 1,
                                 rank=None if distributed else 0,
                                 shuffle=split == 'train', seed=config.DATA_SPLIT_SEED)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available(),
                      drop_last=False, collate_fn=collate_examples,
                      generator=torch.Generator().manual_seed(config.DATA_SPLIT_SEED))
