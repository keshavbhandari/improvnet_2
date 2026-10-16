"""On-the-fly task construction over indexed performance JSONL records."""
from collections import defaultdict
from dataclasses import replace
from bisect import bisect_left
import json
import math
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


class NoStemWindow(ValueError):
    """No window contains both a removable stem and a retained instrument."""


def stem_source(notes, rng, config, target_instrument=None):
    instruments = sorted({n.instrument for n in notes})
    if len(instruments) < 2:
        raise NoStemWindow('Stem restoration requires a multi-instrument mixture')
    if target_instrument is not None:
        instruments = [i for i in instruments if i == target_instrument]
    while instruments:
        instrument = rng.choices(instruments, [config.STEM_DRUM_WEIGHT if i == 'drum' else 1 for i in instruments])[0]
        source = [n for n in notes if n.instrument != instrument]
        attacks = [n.onset for n in notes if n.instrument == instrument]
        try:
            starts = chunk_starts(source, config.CHUNK_TOKENS)
        except ValueError:
            starts = []
        valid = []
        for begin in starts:
            stop = chunk_end(source, begin, config.CHUNK_TOKENS)
            start, end = chunk_bounds(source, begin, stop)
            if bisect_left(attacks, end) > bisect_left(attacks, start):
                valid.append(begin)
        if valid:
            return instrument, source, valid
        instruments.remove(instrument)
    raise NoStemWindow('No aligned missing-stem window is available')


def prepare_example(record, rng, config=cfg, task=None, strength=None, target_instrument=None):
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
    input_mono = rng.random() < config.INPUT_MONO_PROB
    if task == 'stem':
        input_mono = False  # Keep every retained instrument; no skyline extraction.
    history_mono = input_mono and task not in ('harmonize', 'stem')
    strength = sample_strength(rng) if strength is None else strength
    if task == 'stem' or task == 'harmonize' and input_mono:
        strength = 1.0
    domain = notes if task == 'harmonize' or not input_mono else melody
    if task != 'stem':
        target_instrument = None
    full_mix = ordered(notes + drums)
    if task == 'stem':
        target_instrument, domain, starts = stem_source(full_mix, rng, config, target_instrument)
    else:
        starts = chunk_starts(domain, config.CHUNK_TOKENS)
    if task == 'harmonize':
        onsets = [n.onset for n in notes]
        melody_onsets = [n.onset for n in melody]
        starts = sorted({bisect_left(onsets, melody_onsets[min(
            bisect_left(melody_onsets, domain[i].onset), len(melody_onsets) - 1)]) for i in starts})
    use_history = rng.random() < config.HISTORY_PROB
    if use_history:
        noninitial = [i for i in starts if i > 0]
        if not noninitial and task != 'stem':
            # A short piece can still supply a prefix by starting at a later attack.
            noninitial = [i for i in range(1, len(domain)) if domain[i].onset > domain[i - 1].onset]
        starts = noninitial or starts
    begin = rng.choice(starts)
    start = domain[begin].onset
    # The preceding clean target uses the same view as this target, never a
    # separately sampled view or one inferred from the corrupted note count.
    history_notes = full_mix if task == 'stem' else domain
    left = history_tokens([n for n in history_notes if n.onset < start], config.HISTORY_TOKENS, tok) if use_history else []
    stop = chunk_end(domain, begin, config.CHUNK_TOKENS)
    corruption_rng = rng.getstate()
    bounded_corruption = False
    while True:
        rng.setstate(corruption_rng)
        start, end = chunk_bounds(domain, begin, stop)
        core = window(domain, start, end)
        core_melody = window(melody, start, end)
        target_notes = core
        if task == 'refine':
            options = {'names': ['whole_mask']} if bounded_corruption else {}
            source = corrupt_melody(core, strength, rng, tok, config.MAX_CORRUPTIONS, **options)
        elif task == 'rhythm':
            source = mask_timing(core, strength, rng, tok)
        elif task == 'harmonize':
            harmony = window(accompaniment, start, end)
            retained = [] if input_mono else rng.sample(harmony, len(harmony) - round(strength * len(harmony)))
            source = notes_to_tokens(core_melody + retained, tok)
        elif task == 'stem':
            source = notes_to_tokens(core, tok)
            target_notes = window(full_mix, start, end)
            if not any(n.instrument == target_instrument for n in target_notes):
                raise NoStemWindow('Shrunk window has no target-stem attacks')
        else:
            raise ValueError(f'Unknown task {task}')
        target = notes_to_tokens(target_notes, tok)[1:]
        prompt = make_prompt(task, source, left, target_instrument)
        if len(source) <= config.CHUNK_TOKENS and (task == 'stem' or len(target) + 1 <= config.CHUNK_TOKENS) and len(prompt) + len(target) + 2 <= config.SEQ_LEN:
            break
        available = config.SEQ_LEN - len(left) - 6 - int(task == 'stem')
        fraction = min(config.CHUNK_TOKENS / len(source), available / (len(source) + len(target)))
        if task != 'stem':
            fraction = min(fraction, config.CHUNK_TOKENS / (len(target) + 1))
        try:
            stop = shorten_chunk(domain, begin, stop, fraction * .98)
        except ValueError:
            if task == 'stem':
                raise NoStemWindow('Missing-stem onset group cannot fit the full context')
            if task != 'refine' or bounded_corruption or len(target) + 1 > config.CHUNK_TOKENS:
                raise
            # A dense chord can fit cleanly while a note-adding edit cannot.
            # Keep the complete target and use attribute masking at the same strength.
            bounded_corruption = True
    probability = config.HARMONY_STYLE_PROB if task == 'harmonize' else config.RHYTHM_STYLE_PROB if task == 'rhythm' else 0
    visible = rng.random() < probability
    held_harmony = ([replace(n, onset=n.onset - start) for n in accompaniment
                     if n.onset < start < n.onset + n.duration] if task == 'harmonize' and visible else [])
    statistics = performance_statistics(target_notes, core_melody, end - start,
                                       config.SKYLINE_TOLERANCE_MS, held_harmony) if visible else {}
    style_values, style_mask = style_tensors(statistics, task, visible)
    ids = [tok.tok_to_id[t] for t in prompt + target]
    labels = [config.IGNORE_INDEX] * len(prompt) + [tok.tok_to_id[t] for t in target]
    stem_mask, current = [False] * len(prompt), None
    for token in target:
        if isinstance(token, tuple) and token[0] == 'instrument':
            current = token[1]
        stem_mask.append(isinstance(token, tuple) and target_instrument is not None and current == target_instrument)
    return {'input': torch.tensor(ids[:-1]), 'target': torch.tensor(labels[1:]),
            'genre': torch.tensor(config.GENRES.index(record['genre'])),
            'multi_hot': ProcessData().get_instrument_multihot(notes_to_tokens(target_notes, tok)),
            'strength': torch.tensor(strength, dtype=torch.float32), 'task': TASKS.index(task),
            'style_values': style_values, 'style_mask': style_mask,
            'stem_mask': torch.tensor(stem_mask[1:], dtype=torch.bool),
            'stem_instrument': torch.tensor(ProcessData().INSTRUMENT_CLASSES.index(target_instrument) if target_instrument else -1),
            'stem_name': target_instrument,
            'history_length': len(left), 'source_length': len(source), 'response_length': len(target),
            'input_mono': input_mono, 'history_mono': history_mono,
            'start_ms': start, 'end_ms': end}


class MelodyDataset(Dataset):
    def __init__(self, data_file, split, config=cfg):
        self.path, self.split, self.config = Path(data_file), split, config
        validate_lengths(config)
        if not 0 <= config.CLASSICAL_PROB <= 1:
            raise ValueError('CLASSICAL_PROB must be in [0,1]')
        if not 0 <= config.MULTI_INSTRUMENT_PROB <= 1 or not math.isfinite(config.STEM_DRUM_WEIGHT) or config.STEM_DRUM_WEIGHT <= 0:
            raise ValueError('Invalid mixture probability or drum stem weight')
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
        self.mixtures = {task: defaultdict(lambda: defaultdict(lambda: defaultdict(list))) for task in TASKS}
        for row in self.rows:
            for task in TASKS:
                if task == 'rhythm' and row['source'] in ('wikifonia', 'curated'):
                    continue  # Notated timing is not performed swing supervision.
                multi = row.get('has_multi_instrument', len(row.get('instruments', [])) >= 2)
                if (task != 'harmonize' or row['has_harmony']) and (task != 'stem' or multi):
                    self.groups[task][row['genre']][row['source']].append(row['offset'])
                    self.mixtures[task][row['genre']][multi][row['source']].append(row['offset'])

        missing = [t for i, t in enumerate(TASKS) if config.TASK_WEIGHTS[i] > 0 and not self.groups[t]]
        if missing:
            raise ValueError(f'No {split} supervision for tasks {missing}; prepare mixture-preserving v4 data or set their task weights to zero')

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
        names = sorted(genres)
        if 'classical' in genres and len(names) > 1:
            weights = [self.config.CLASSICAL_PROB if g == 'classical' else
                       (1 - self.config.CLASSICAL_PROB) / (len(names) - 1) for g in names]
            genre = rng.choices(names, weights)[0]
        else:
            genre = rng.choice(names)
        sources = genres[genre]
        views = self.mixtures[task][genre]
        if genre == 'classical' and task != 'stem' and len(views) > 1:
            sources = views[rng.random() < self.config.MULTI_INSTRUMENT_PROB]
        for _ in range(32):
            offsets = sources[rng.choice(sorted(sources))]
            with self.path.open('rb') as handle:
                handle.seek(rng.choice(offsets))
                record = json.loads(handle.readline())
            record['_augment'] = self.split == 'train'
            try:
                return prepare_example(record, rng, self.config, task=task)
            except NoStemWindow:
                if task != 'stem':
                    raise
        raise NoStemWindow('Could not sample an aligned stem window after 32 attempts')


def collate_examples(batch):
    size = max(len(x['input']) for x in batch)
    inputs = torch.full((len(batch), size), cfg.PAD_ID, dtype=torch.long)
    targets = torch.full_like(inputs, cfg.IGNORE_INDEX)
    stem_mask = torch.zeros_like(inputs, dtype=torch.bool)
    for i, x in enumerate(batch):
        inputs[i, :len(x['input'])] = x['input']
        targets[i, :len(x['target'])] = x['target']
        stem_mask[i, :len(x['stem_mask'])] = x['stem_mask']
    return {'input': inputs, 'target': targets,
            **{key: torch.stack([x[key] for x in batch]) for key in ('genre', 'multi_hot', 'strength', 'style_values', 'style_mask')},
            'task': torch.tensor([x['task'] for x in batch]), 'stem_mask': stem_mask,
            'stem_instrument': torch.stack([x['stem_instrument'] for x in batch])}


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
