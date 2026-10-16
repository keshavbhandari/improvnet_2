"""On-the-fly task construction over indexed performance JSONL records."""
from collections import defaultdict
from dataclasses import replace
from bisect import bisect_left
import json
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from improvnet.model import melody_config as cfg
from improvnet.utils.ar_utils import ProcessData
from improvnet.utils.melody import (Note, MelodyTokenizer, TASKS, corrupt_melody,
                                   make_prompt, mask_timing, melody_voice, notes_to_tokens,
                                   sample_strength, window, ordered)
from improvnet.utils.melody_chunks import (chunk_bounds, chunk_end, chunk_starts,
                                          history_tokens, shorten_chunk, validate_lengths)

from improvnet.utils.melody_style import performance_statistics, style_tensors


def prepare_example(record, rng, config=cfg, task=None, strength=None):
    validate_lengths(config)
    tok = MelodyTokenizer()
    # Older prepared files inherited the tokenizer's noncanonical piano fallback.
    notes = [Note('Acoustic Piano' if n[0] == 'piano' else n[0], *n[1:])
             for n in record['notes']]
    melody = melody_voice([notes[i] for i in record['melody_indices']])
    melody_ids = {id(n) for n in melody}
    melody_indices = {i for i, n in enumerate(notes) if id(n) in melody_ids}
    task = task or rng.choices(TASKS, weights=config.TASK_WEIGHTS)[0]
    has_harmony = record.get('has_harmony', bool(record.get('harmony_windows', [])))
    if task == 'harmonize' and not has_harmony:
        raise ValueError('No aligned accompaniment is available for this record')
    # Paired transposition changes context, input and target together.
    if record.get('_augment'):
        shift = rng.randint(-5, 5)
        shift = max(-min(n.pitch for n in notes), min(127 - max(n.pitch for n in notes), shift))
        by_id = {id(n): replace(n, pitch=n.pitch + shift) for n in notes}
        melody = [by_id[id(n)] for n in melody]
        notes = [by_id[id(n)] for n in notes]
    drums = [Note(*n) for n in record.get('drums', [])]
    accompaniment = [n for i, n in enumerate(notes) if i not in melody_indices]
    if task == 'drums' and not drums:
        raise ValueError('No aligned drum performance is available for this record')
    input_mono = rng.random() < config.INPUT_MONO_PROB
    # This describes the pitched view; drum history also contains preceding drums.
    history_mono = input_mono and task != 'harmonize'
    strength = sample_strength(rng) if strength is None else strength
    if task == 'drums' or task == 'harmonize' and input_mono:
        strength = 1.0  # Drums are absent; mono harmony sources have no accompaniment.
    domain = notes if task == 'harmonize' or not input_mono else melody
    starts = chunk_starts(domain, config.CHUNK_TOKENS)
    if task == 'harmonize':
        onsets = [n.onset for n in notes]
        melody_onsets = [n.onset for n in melody]
        starts = sorted({bisect_left(onsets, melody_onsets[min(
            bisect_left(melody_onsets, domain[i].onset), len(melody_onsets) - 1)]) for i in starts})
    use_history = rng.random() < config.HISTORY_PROB
    if use_history:
        noninitial = [i for i in starts if i > 0]
        if not noninitial:
            # A short piece can still supply a prefix by starting at a later attack.
            noninitial = [i for i in range(1, len(domain)) if domain[i].onset > domain[i - 1].onset]
        starts = noninitial or starts
    begin = rng.choice(starts)
    start = domain[begin].onset
    # The preceding clean target uses the same view as this target, never a
    # separately sampled view or one inferred from the corrupted note count.
    history_notes = ordered(domain + (drums if task == 'drums' else []))
    left = history_tokens([n for n in history_notes if n.onset < start], config.HISTORY_TOKENS, tok) if use_history else []
    stop = chunk_end(domain, begin, config.CHUNK_TOKENS)
    corruption_rng = rng.getstate()
    while True:
        rng.setstate(corruption_rng)
        start, end = chunk_bounds(domain, begin, stop)
        core = window(domain, start, end)
        core_melody = window(melody, start, end)
        target_notes = core
        if task == 'refine':
            source = corrupt_melody(core, strength, rng, tok, config.MAX_CORRUPTIONS)
        elif task == 'rhythm':
            source = mask_timing(core, strength, rng, tok)
        elif task == 'harmonize':
            harmony = window(accompaniment, start, end)
            retained = [] if input_mono else rng.sample(harmony, len(harmony) - round(strength * len(harmony)))
            source = notes_to_tokens(core_melody + retained, tok)
        elif task == 'drums':
            source = notes_to_tokens(core, tok)
            target_notes = ordered(core + window(drums, start, end))
        else:
            raise ValueError(f'Unknown task {task}')
        target = notes_to_tokens(target_notes, tok)[1:]
        prompt = make_prompt(task, source, left)
        if len(source) <= config.CHUNK_TOKENS and (task == 'drums' or len(target) + 1 <= config.CHUNK_TOKENS) and len(prompt) + len(target) + 2 <= config.SEQ_LEN:
            break
        available = config.SEQ_LEN - len(left) - 6
        fraction = min(config.CHUNK_TOKENS / len(source), available / (len(source) + len(target)))
        if task != 'drums':
            fraction = min(fraction, config.CHUNK_TOKENS / (len(target) + 1))
        stop = shorten_chunk(domain, begin, stop, fraction * .98)
    probability = config.HARMONY_STYLE_PROB if task == 'harmonize' else config.RHYTHM_STYLE_PROB if task == 'rhythm' else 0
    visible = rng.random() < probability
    held_harmony = ([replace(n, onset=n.onset - start) for n in accompaniment
                     if n.onset < start < n.onset + n.duration] if task == 'harmonize' and visible else [])
    statistics = performance_statistics(target_notes, core_melody, end - start,
                                       config.SKYLINE_TOLERANCE_MS, held_harmony) if visible else {}
    style_values, style_mask = style_tensors(statistics, task, visible)
    ids = [tok.tok_to_id[t] for t in prompt + target]
    labels = [config.IGNORE_INDEX] * len(prompt) + [tok.tok_to_id[t] for t in target]
    return {'input': torch.tensor(ids[:-1]), 'target': torch.tensor(labels[1:]),
            'genre': torch.tensor(config.GENRES.index(record['genre'])),
            'multi_hot': ProcessData().get_instrument_multihot(notes_to_tokens(target_notes, tok)),
            'strength': torch.tensor(strength, dtype=torch.float32), 'task': TASKS.index(task),
            'style_values': style_values, 'style_mask': style_mask,
            'history_length': len(left), 'source_length': len(source), 'response_length': len(target),
            'input_mono': input_mono, 'history_mono': history_mono,
            'start_ms': start, 'end_ms': end}


class MelodyDataset(Dataset):
    def __init__(self, data_file, split, config=cfg):
        self.path, self.split, self.config = Path(data_file), split, config
        validate_lengths(config)
        self.epoch = 0
        self.entries = json.loads(self.path.with_suffix('.index.json').read_text())
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
                if (task != 'harmonize' or row['has_harmony']) and (task != 'drums' or row.get('has_drums', False)):
                    self.groups[task][row['genre']][row['source']].append(row['offset'])

        missing = [t for i, t in enumerate(TASKS) if config.TASK_WEIGHTS[i] > 0 and not self.groups[t]]
        if missing:
            raise ValueError(f'No {split} supervision for tasks {missing}; prepare drum-preserving v3 data or explicitly set their task weights to zero')

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
            **{key: torch.stack([x[key] for x in batch]) for key in ('genre', 'multi_hot', 'strength', 'style_values', 'style_mask')},
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
