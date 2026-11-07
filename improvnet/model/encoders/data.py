"""Shared piece sampling, masking, transposition and style-positive view construction."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset
from improvnet.model import melody_config
from improvnet.utils.melody import MelodyTokenizer, Note, ordered, notes_to_tokens
from improvnet.utils.melody_data import MelodyDataset
from improvnet.utils.melody_reference import PIANOS, crop_notes
from improvnet.utils.melody_stems import kit_only_notes


def transpose_view(notes, rng, semitones=7):
    pitched = [n.pitch for n in notes if n.instrument != 'drum']
    if not pitched:
        return list(notes), 0
    low, high = max(-semitones, -min(pitched)), min(semitones, 127-max(pitched))
    shift = rng.randint(low, high)
    return [replace(n, pitch=n.pitch+shift) if n.instrument != 'drum' else n for n in notes], shift


def mask_view(ids, tokenizer, rng, rate=None, *, probability=.9, rate_range=(.15,.45)):
    """Independent contiguous masking, including instrument/timing/clock tokens."""
    ids = list(ids)
    if rate is None and rng.random() >= probability:
        return ids
    rate = rng.uniform(*rate_range) if rate is None else rate
    eligible = list(range(1, len(ids)-1))  # Preserve S/E; no padding in these crops.
    count = min(len(eligible)-1, round(len(eligible)*rate))
    masked = set()
    while len(masked) < count:
        start = rng.choice(eligible)
        for pos in range(start, min(len(ids)-1, start+rng.randint(4, 16))):
            masked.add(pos)
            if len(masked) == count:
                break
    for pos in masked:
        ids[pos] = tokenizer.tok_to_id['<MASK>']
    return ids


def instrument_views(notes, rng):
    instruments = sorted({n.instrument for n in notes if n.instrument != 'drum'})
    if not instruments:
        raise ValueError('No pitched notes in encoder pair')
    piano = [i for i in instruments if i in PIANOS]
    solo = rng.choice(piano if piano and rng.random() < .9 else instruments)
    a = [n for n in notes if n.instrument == solo]
    if piano and any(n.instrument == 'drum' for n in notes) and rng.random() < .85:
        mix = {solo, 'drum'} if solo in PIANOS else {rng.choice(piano), 'drum'}
    else:
        rest = sorted({n.instrument for n in notes}-{solo})
        mix = {solo, *rng.sample(rest, min(len(rest), rng.randint(1, 3)))}
    return a, [n for n in notes if n.instrument in mix]


def positive_pair(notes, rng, tokenizer, *, max_tokens=512, augment=True,
                  max_semitones=7, aligned=None, mask_prob=.9, mask_rate_range=(.15,.45)):
    """Mixture instruments always share one original clock, even in shifted views."""
    a, b = instrument_views(ordered(notes), rng)
    aligned = rng.random() < .5 if aligned is None else aligned
    def mixture_crop():
        best, score = None, (-1, -1)
        wanted = min(2, len({n.instrument for n in b}))
        for _ in range(16):
            candidate = crop_notes(b, rng, max_tokens)
            nc = candidate[0]
            quality = (len({n.instrument for n in nc}), len(nc))
            if quality > score:
                best, score = candidate, quality
            if quality[0] >= wanted and 4*len(nc)+2 >= min(64, 4*len(b)+2):
                break
        return best
    if aligned:
        # A mixture crop defines one interval for all its instruments.
        # Only use an aligned view if its solo actually plays in this interval.
        for _ in range(8):
            bc, start, end = mixture_crop()
            ac = [replace(n, onset=n.onset-start) for n in a if start <= n.onset < end]
            if ac and len(notes_to_tokens(ac, tokenizer)) <= max_tokens:
                break
        else:
            aligned = False
    if not aligned:
        ac, a_start, a_end = crop_notes(a, rng, max_tokens)
        bc, start, end = mixture_crop()
    else:
        a_start, a_end = start, end
    # Independent shifts intentionally allow a different key in each positive.
    shifts = (0, 0)
    if augment:
        ac, sa = transpose_view(ac, rng, max_semitones)
        bc, sb = transpose_view(bc, rng, max_semitones)
        shifts = sa, sb
    encode = lambda view: [tokenizer.tok_to_id[t] for t in notes_to_tokens(view, tokenizer)]
    ai, bi = encode(ac), encode(bc)
    if augment:
        ai, bi = mask_view(ai, tokenizer, rng, probability=mask_prob, rate_range=mask_rate_range), mask_view(bi, tokenizer, rng, probability=mask_prob, rate_range=mask_rate_range)
    return dict(a=torch.tensor(ai), b=torch.tensor(bi), aligned=aligned, shifts=shifts,
                a_bounds=(a_start, a_end), b_bounds=(start, end),
                a_instruments=sorted({n.instrument for n in ac}), b_instruments=sorted({n.instrument for n in bc}))


class PiecePairDataset(Dataset):
    def __init__(self, manifest, split, tokenizer=None, min_length=64, max_length=512,
                 eval_length=512, seed=42, pitch_augmentation=True, max_semitone_shift=7,
                 aligned_pair_prob=.5, piano_drum_draw_prob=.5, jazz_piano_drum_prob=.5,
                 mask_prob=.9, mask_rate_range=(.15,.45)):
        # Encoder cohort/sampling identity must not follow later AR ratio changes.
        base_settings = SimpleNamespace(**{k:v for k,v in vars(melody_config).items() if k.isupper()})
        base_settings.CLASSICAL_RATIO = json.loads(Path(manifest).read_text())['ratio']
        self.base = MelodyDataset(manifest, split, base_settings)
        self.split, self.epoch, self.seed = split, 0, seed
        self.aligned_pair_prob, self.piano_drum_draw_prob = aligned_pair_prob, piano_drum_draw_prob
        self.jazz_piano_drum_prob = jazz_piano_drum_prob
        self.mask_prob, self.mask_rate_range = mask_prob, mask_rate_range
        self.max_tokens = max_length if split == 'train' else eval_length
        self.augment, self.pitch_range = split == 'train' and pitch_augmentation, max_semitone_shift
        self.tokenizer = tokenizer or MelodyTokenizer()
        self.rows = (self.base.jazz_blues+self.base.other+self.base.ar_mixtures
                     if split == 'train' else self.base.eval_rows)
        self.piano_drums = [i for i, row in enumerate(self.rows)
                            if 'drum' in row.get('instruments', []) and PIANOS.intersection(row.get('instruments', []))]
        self.jazz_piano_drums = [i for i in self.piano_drums if self.rows[i]['genre'] in ('jazz','blues')]
        self.vocab_digest = hashlib.sha256(json.dumps(self.tokenizer.signature()).encode()).hexdigest()
        self.fingerprint = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
        self.tokenizer.config = dict(self.tokenizer.config, reference_protocol=6, vocabulary_digest=self.vocab_digest)

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def close(self):
        self.base.cache.clear()

    def __getitem__(self, key):
        index, draw = key if isinstance(key, tuple) else (key, key)
        rng = random.Random(self.seed + self.epoch*1000000007 + draw*9176 + index)
        if self.split == 'train' and self.piano_drums and rng.random() < self.piano_drum_draw_prob:
            pool = self.jazz_piano_drums if self.jazz_piano_drums and rng.random() < self.jazz_piano_drum_prob else self.piano_drums
            index = rng.choice(pool)
        failures = []
        for attempt in range(64):
            row = self.rows[index]
            try:
                record = self.base.load_record(row)
                notes = [Note('Acoustic Piano' if n[0]=='piano' else n[0], *n[1:]) for n in record['notes']]
                notes += kit_only_notes([Note(*n) for n in record.get('drums', [])])
                result = positive_pair(notes, rng, self.tokenizer, max_tokens=self.max_tokens,
                                       augment=self.augment, max_semitones=self.pitch_range,
                                       aligned=rng.random()<self.aligned_pair_prob,
                                       mask_prob=self.mask_prob, mask_rate_range=self.mask_rate_range)
                group = row.get('group') or row.get('source_group') or row.get('sha256') or row.get('path') or str(row['offset'])
                result['piece_id'] = int.from_bytes(hashlib.sha256(str(group).encode()).digest()[:8], 'little') % (2**63-1)
                return result
            except (ValueError, OSError, KeyError) as error:
                failures.append(str(error)[:120])
                index = rng.randrange(len(self.rows))
        raise RuntimeError(f'Encoder pair could not be sampled in {self.split}: {failures[-3:]}')
