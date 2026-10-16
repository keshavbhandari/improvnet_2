"""Shared note representation, skyline and ImprovNet-style corruptions.

Times are milliseconds; no beat grid is estimated. Corruption names are never
serialized into a model prompt.
"""
from dataclasses import dataclass, replace
import random

from improvnet.tokenizer.absolute import AbsTokenizer
from improvnet.tokenizer.midi import MidiDict

TASKS = ("refine", "rhythm", "harmonize", "stem")
TASK_TOKENS = dict(zip(TASKS, ("<REFINE>", "<RHYTHM>", "<HARMONIZE>", "<STEM>")))


class MelodyTokenizer(AbsTokenizer):
    def __init__(self):
        super().__init__()
        self.base_vocab_size = self.vocab_size
        # Keep every previous vocabulary ID, including the retired drum task.
        self.add_tokens_to_vocab(["<REFINE>", "<RHYTHM>", "<HARMONIZE>", "<DRUMS>", "<STEM>"])

    def signature(self):
        # JSON-safe and exact: detects even a same-sized reordered vocabulary.
        return [list(t) if isinstance(t, tuple) else t for t in self.vocab]


@dataclass(frozen=True)
class Note:
    instrument: str
    pitch: int
    velocity: int
    onset: int
    duration: int


def ordered(notes):
    return sorted(notes, key=lambda n: (n.onset, -n.pitch, n.instrument, n.duration))


def melody_voice(notes):
    """Resolve simultaneous attacks, including ties introduced by 10 ms rounding."""
    selected = {}
    for n in notes:
        previous = selected.get(n.onset)
        if previous is None or (n.pitch, n.duration, n.velocity) > (previous.pitch, previous.duration, previous.velocity):
            selected[n.onset] = n
    return ordered(selected.values())


def notes_to_tokens(notes, tokenizer=None):
    tok = tokenizer or MelodyTokenizer()
    tokens, segment = ["<S>"], 0
    for n in ordered(notes):
        if n.onset < 0 or n.duration <= 0 or not 0 <= n.pitch <= 127:
            raise ValueError(f"Invalid note: {n}")
        onset = round(n.onset / tok.time_step_ms) * tok.time_step_ms
        new_segment, local = divmod(onset, tok.abs_time_step_ms)
        tokens.extend(["<T>"] * (new_segment - segment))
        segment = new_segment
        tokens.extend([("instrument", n.instrument),
                       ("note", n.pitch, tok._quantize_velocity(n.velocity)),
                       ("onset", local), ("dur", tok._quantize_dur(n.duration))])
    return tokens + ["<E>"]


def tokens_to_notes(tokens, tokenizer=None):
    tok = tokenizer or MelodyTokenizer()
    notes, segment, i = [], 0, 0
    while i < len(tokens):
        t = tokens[i]
        if t == "<T>":
            segment += tok.abs_time_step_ms
            i += 1
        elif t in ("<S>", "<E>", "<D>"):
            i += 1
        else:
            group = tokens[i:i + 4]
            if len(group) != 4 or any(not isinstance(x, tuple) for x in group):
                raise ValueError(f"Incomplete note at token {i}: {group}")
            if [x[0] for x in group] != ["instrument", "note", "onset", "dur"]:
                raise ValueError(f"Invalid note at token {i}: {group}")
            inst, pv, onset, dur = group
            if dur[1] <= 0 or onset[1] < 0 or onset[1] >= tok.abs_time_step_ms:
                raise ValueError(f"Invalid note timing: {group}")
            notes.append(Note(inst[1], pv[1], pv[2], segment + onset[1], dur[1]))
            i += 4
    return ordered(notes)


def read_midi_notes(path, tokenizer=None, include_drums=False):
    tok = tokenizer or MelodyTokenizer()
    midi = MidiDict.from_midi(str(path))
    tokens = tok.tokenize(midi, remove_preceding_silence=False, add_dim_tok=False)
    # Drum examples are not melody/harmony supervision for the piano pilot.
    return [n for n in tokens_to_notes(tokens, tok) if include_drums or n.instrument != "drum"]


def write_midi_notes(notes, path, tokenizer=None):
    tok = tokenizer or MelodyTokenizer()
    # A late note-off from a previous strike must not cut off its retrigger.
    notes = ordered(notes)
    next_attack, released = {}, []
    for n in reversed(notes):
        key = (n.instrument, n.pitch)
        following = next_attack.get(key)
        if following is not None and following > n.onset:
            n = replace(n, duration=min(n.duration, following - n.onset))
        next_attack[key] = n.onset
        released.append(n)
    tok.detokenize(notes_to_tokens(released, tok)).to_midi().save(str(path))


def attack_groups(notes, tolerance_ms=50):
    """Anchored performed-chord groups; preserve fast runs and retriggers."""
    notes = ordered(notes)
    groups, i = [], 0
    while i < len(notes):
        j = i + 1
        while (j < len(notes) and notes[j].onset - notes[i].onset <= tolerance_ms
               and notes[j].pitch not in {n.pitch for n in notes[i:j]}
               and max(n.onset + n.duration for n in notes[i:j]) > notes[j].onset):
            j += 1
        groups.append(notes[i:j])
        i = j
    return groups


def skyline(notes, chord_tolerance_ms=50):
    """Highest original attack per rolled chord; an approximate voice separator."""
    notes = ordered(notes)
    selected = {id(max(g, key=lambda n: (n.pitch, -n.onset)))
                for g in attack_groups(notes, chord_tolerance_ms)}
    return ([n for n in notes if id(n) in selected],
            [n for n in notes if id(n) not in selected])


def window(notes, start, end):
    # Ownership is by onset. A held note crosses a boundary with no new attack.
    return [replace(n, onset=n.onset - start) for n in notes if start <= n.onset < end]


def sample_strength(rng):
    # Include exact endpoints, which a continuous uniform draw almost never hits.
    u = rng.random()
    return 0.0 if u < .05 else 1.0 if u > .95 else rng.random()


CORRUPTIONS = ("pitch_velocity_mask", "onset_duration_mask", "whole_mask",
               "permute_pitch", "permute_pitch_velocity", "fragmentation",
               "incorrect_transposition", "note_modification")


def corrupt_melody(notes, strength, rng, tokenizer=None, max_corruptions=3, names=None):
    """Compose 1–3 paper-inspired edits, scaled by a shared severity in [0, 1]."""
    if not 0 <= strength <= 1 or not 1 <= max_corruptions <= 3:
        raise ValueError("strength must be in [0,1]; max_corruptions in [1,3]")
    tok = tokenizer or MelodyTokenizer()
    notes = ordered(notes)
    if strength == 0 or not notes:
        return notes_to_tokens(notes, tok)
    choices = list(names) if names is not None else rng.sample(
        CORRUPTIONS, rng.randint(1, max_corruptions))
    if len(choices) > 3 or any(name not in CORRUPTIONS for name in choices):
        raise ValueError("Expected at most three known corruptions")
    # Apply slot masks last so subsequent note edits cannot invalidate them.
    choices.sort(key=lambda name: name.endswith("mask"))
    masks = set()
    for name in choices:
        k = min(len(notes), max(1, round(strength * len(notes))))
        indices = rng.sample(range(len(notes)), k)
        if name in ("pitch_velocity_mask", "onset_duration_mask", "whole_mask"):
            attrs = {"pitch_velocity_mask": (1,), "onset_duration_mask": (2, 3),
                     "whole_mask": (0, 1, 2, 3)}[name]
            # Store by note object identity until structural edits are complete.
            masks.update((id(notes[i]), attr) for i in indices for attr in attrs)
        elif name.startswith("permute"):
            original = [notes[i] for i in indices]
            shuffled = original.copy()
            rng.shuffle(shuffled)
            for i, source in zip(indices, shuffled):
                old = notes[i]
                notes[i] = replace(old, pitch=source.pitch, velocity=(
                    source.velocity if name == "permute_pitch_velocity" else old.velocity))
                masks.update((id(notes[i]), a) for a in range(4) if (id(old), a) in masks)
        elif name == "incorrect_transposition":
            for i in indices:
                old = notes[i]
                notes[i] = replace(old, pitch=max(0, min(127, old.pitch + rng.choice((-5, 5)))))
                masks.update((id(notes[i]), a) for a in range(4) if (id(old), a) in masks)
        elif name == "fragmentation":
            notes = notes[:len(notes) - k]
        else:  # note_modification: merge omissions and split long notes.
            selected = set(indices)
            modified = []
            for i, n in enumerate(notes):
                if i in selected and modified and n.onset - modified[-1].onset >= 50:
                    prev = modified[-1]
                    modified[-1] = replace(prev, duration=max(prev.duration, n.onset + n.duration - prev.onset))
                elif i in selected and n.duration > 500:
                    half = max(10, round(n.duration / 20) * 10)
                    modified.extend([replace(n, duration=half), replace(n,
                        onset=n.onset + half, duration=n.duration - half,
                        pitch=max(0, min(127, n.pitch + rng.choice((-5, -2, 2, 5)))),
                        velocity=rng.choice((45, 60, 75, 90, 105)))])
                else:
                    modified.append(n)
            notes = modified
    # Serialize masks without losing the unmasked clock or note boundaries.
    tokens, segment = ["<S>"], 0
    for n in ordered(notes):
        encoded = notes_to_tokens([n], tok)[1:-1]
        new_segment = n.onset // tok.abs_time_step_ms
        tokens.extend(["<T>"] * (new_segment - segment))
        segment = new_segment
        attrs = encoded[-4:]
        tokens.extend("<MASK>" if (id(n), a) in masks else t for a, t in enumerate(attrs))
    return tokens + ["<E>"]


def mask_timing(notes, fraction, rng, tokenizer=None):
    if not 0 <= fraction <= 1:
        raise ValueError("Mask fraction must be in [0,1]")
    tokens = notes_to_tokens(notes, tokenizer)
    # Mask simultaneous notes together, so polyphonic timing edits preserve chords.
    onsets = sorted({n.onset for n in notes})
    chosen = set(rng.sample(onsets, round(fraction * len(onsets))))
    selected = [n.onset in chosen for n in ordered(notes)]
    note_idx = -1
    for i, token in enumerate(tokens):
        if isinstance(token, tuple) and token[0] == "instrument":
            note_idx += 1
        if note_idx >= 0 and selected[note_idx] and isinstance(token, tuple) and token[0] in ("onset", "dur"):
            tokens[i] = "<MASK>"
    return tokens


def make_prompt(task, source_tokens, left=(), target_instrument=None):
    condition = []
    if task == 'stem':
        if ('instrument', target_instrument) not in MelodyTokenizer().tok_to_id:
            raise ValueError('Stem task requires a supported target instrument')
        condition = [('instrument', target_instrument)]
    return ([TASK_TOKENS[task], *condition, *left, "<SEP>", *source_tokens,
             "<SEP>", "<S>"])


def record_instruments(record):
    """Instrument stems use the tokenizer's canonical classes, including drums."""
    return sorted({'Acoustic Piano' if n[0] == 'piano' else n[0]
                   for n in record['notes'] + record.get('drums', [])})
