"""Classical performance MIDI -> melody refinement -> timing -> harmonization."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import random

import torch

from improvnet.model.melody_model import MelodyRefinementModel, validate_checkpoint
from improvnet.model.melody_train import autocast
from improvnet.utils.ar_utils import ProcessData
from improvnet.utils.melody import (MelodyTokenizer, corrupt_melody, make_prompt,
                                   mask_timing, melody_voice, notes_to_tokens, ordered,
                                   read_midi_notes, skyline, tokens_to_notes,
                                   window, write_midi_notes)


def preserve_melody(generated, melody, tolerance_ms=50):
    """Reinstate anchors exactly, replacing generated near-duplicates first."""
    remaining = list(generated)
    for anchor in melody:
        matches = [i for i, n in enumerate(remaining) if n.instrument == anchor.instrument
                   and n.pitch == anchor.pitch and (abs(n.onset - anchor.onset) <= tolerance_ms
                   or anchor.onset <= n.onset < anchor.onset + anchor.duration)]
        # All near-identical attacks are replacements, not extra strikes.
        for i in reversed(matches):
            remaining.pop(i)
    return ordered(remaining + list(melody))


class MelodyInference:
    def __init__(self, checkpoint_path, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
        self.meta = validate_checkpoint(checkpoint)
        self.tokenizer = MelodyTokenizer()
        self.model = MelodyRefinementModel(**self.meta['model_kwargs'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
        self.model.to(self.device).eval()
        self.processor = ProcessData()

    @torch.no_grad()
    def generate(self, prompt, genre, strength, instruments, window_ms,
                 temperature=0.9, top_k=50, fixed_notes=None, monophonic=False):
        """Generate every attribute, with a note grammar and optional pitch anchors.

        Fixed-note decoding is for timing edits: instrument/pitch/velocity and
        count are anchored; onsets, durations and sequence end are decoded.
        """
        tok, device = self.tokenizer, self.device
        if genre not in self.meta['genres'] or not 0 <= strength <= 1:
            raise ValueError('Unknown genre or strength outside [0,1]')
        budget = self.meta['seq_len'] - len(prompt) - 2
        if budget < 5 or (fixed_notes is not None and budget < 4 * len(fixed_notes) + 1):
            raise ValueError('Prompt leaves insufficient room for a complete response')
        conditions = self.processor.get_instrument_multihot(
            [('instrument', inst) for inst in instruments]).unsqueeze(0).to(device)
        ids = torch.tensor([[tok.tok_to_id[t] for t in prompt]], device=device)
        with autocast(device):
            logits, cache = self.model(ids, genre=torch.tensor([self.meta['genres'].index(genre)], device=device),
                                       multi_hot=conditions, strength=torch.tensor([strength], device=device),
                                       use_cache=True)
        offset = len(prompt) + 2
        output, stage, segment, last_onset, count = ['<S>'], 0, 0, -1, 0
        inst_ids = [tok.tok_to_id[('instrument', inst)] for inst in instruments if inst != 'drum']
        if not inst_ids:
            raise ValueError('No supported pitched output instrument')
        pitch_ids = [tok.tok_to_id[t] for t in tok.note_tokens if t[2] > 0]
        duration_ids = [tok.tok_to_id[t] for t in tok.dur_tokens if t[1] > 0]
        for _ in range(budget):
            if stage == 0:
                if fixed_notes is not None:
                    legal = ([tok.tok_to_id['<E>']] if count == len(fixed_notes) else
                             [tok.tok_to_id[('instrument', fixed_notes[count].instrument)]])
                    if count < len(fixed_notes) and fixed_notes[count].onset // tok.abs_time_step_ms > segment // tok.abs_time_step_ms:
                        legal = [tok.tok_to_id['<T>']]
                else:
                    legal = [tok.tok_to_id['<E>']] if count else []
                    # Reserve four attributes plus an end token for a new note.
                    if len(output) - 1 + 5 <= budget and last_onset < window_ms - 10:
                        legal += inst_ids
                    if segment + tok.abs_time_step_ms < window_ms:
                        legal.append(tok.tok_to_id['<T>'])
            elif stage == 1:
                legal = ([tok.tok_to_id[('note', fixed_notes[count].pitch,
                          tok._quantize_velocity(fixed_notes[count].velocity))]]
                         if fixed_notes is not None else pitch_ids)
            elif stage == 2:
                # Keep notes ordered, permit simultaneous harmony notes.
                minimum = max(segment, last_onset)
                if fixed_notes is not None or monophonic:
                    minimum = max(segment, last_onset + 10)
                maximum = window_ms
                if fixed_notes is not None:
                    # Reserve at least one time tick for every remaining note.
                    maximum -= 10 * (len(fixed_notes) - count - 1)
                legal = [tok.tok_to_id[t] for t in tok.onset_tokens
                         if minimum <= segment + t[1] < maximum]
            else:
                legal = duration_ids
            if not legal:
                raise RuntimeError('No valid next token; reduce timing masking or shorten the excerpt')
            scores = logits[0, -1, legal].float()
            if not torch.isfinite(scores).any():
                raise RuntimeError('No finite legal logits')
            if temperature <= 0:
                pick = scores.argmax().item()
            else:
                scores = scores / temperature
                if top_k > 0 and top_k < len(legal):
                    threshold = scores.topk(top_k).values[-1]
                    scores[scores < threshold] = -torch.inf
                pick = torch.multinomial(scores.softmax(-1), 1).item()
            token_id = legal[pick]
            token = tok.id_to_tok[token_id]
            output.append(token)
            if token == '<E>':
                return tokens_to_notes(output, tok)
            if token == '<T>':
                segment += tok.abs_time_step_ms
            else:
                if stage == 2:
                    last_onset = segment + token[1]
                if stage == 3:
                    count += 1
                stage = (stage + 1) % 4
            with autocast(device):
                logits, cache = self.model(torch.tensor([[token_id]], device=device),
                                           use_cache=True, past_key_values=cache, seq_offset=offset)
            offset += 1
        raise RuntimeError('Response exhausted its token budget before <E>')

    def transform(self, melody, task, genre, strength, rng, temperature=.9, top_k=50,
                  keep_pitches=True, output_instruments=None):
        if task != 'harmonize' and strength == 0:
            return list(melody), []
        tok = self.tokenizer
        width = self.meta['window_ms']
        span = self.meta['context_windows'] * width
        result, stats = [], []
        if not melody:
            return result, stats
        for start in range(0, max(n.onset for n in melody) + 1, width):
            core = window(melody, start, start + width)
            if not core:
                continue
            if task == 'refine':
                source = corrupt_melody(core, strength, rng, tok, self.meta['max_corruptions'])
            elif task == 'rhythm':
                source = mask_timing(core, strength, rng, tok)
            elif task == 'harmonize':
                source = notes_to_tokens(core, tok)
            else:
                raise ValueError(task)
            # Previously generated left context and unchanged future melody.
            left = notes_to_tokens(window(result, max(0, start - span), start), tok) if start else []
            right = notes_to_tokens(window(melody, start + width, start + width + span), tok)
            prompt = make_prompt(task, source, left, right)
            reserve = len(notes_to_tokens(core, tok)) + 2
            while len(prompt) + reserve + 2 > self.meta['seq_len'] and (left or right):
                if len(left) >= len(right):
                    left = []
                else:
                    right = []
                prompt = make_prompt(task, source, left, right)
            instruments = output_instruments if task == 'harmonize' and output_instruments else sorted({n.instrument for n in core})
            generated = self.generate(prompt, genre, strength, instruments, width,
                                      temperature, top_k,
                                      fixed_notes=ordered(core) if task == 'rhythm' and keep_pitches else None,
                                      monophonic=task != 'harmonize')
            if task == 'harmonize':
                generated = preserve_melody(generated, core)
            result.extend(replace(n, onset=n.onset + start) for n in generated)
            stats.append(dict(start_ms=start, input_notes=len(core), output_notes=len(generated),
                              task=task, strength=strength))
        if task == 'harmonize':
            # Also protect held anchors that started in a preceding window.
            result = preserve_melody(result, melody)
        return ordered(result), stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--genre', choices=['jazz', 'blues'], default='jazz')
    parser.add_argument('--task', choices=['pipeline', 'refine', 'rhythm', 'harmonize'], default='pipeline')
    parser.add_argument('--melody-strength', type=float, default=.2)
    parser.add_argument('--rhythm-strength', type=float, default=.5)
    parser.add_argument('--passes', type=int, default=1, help='Melody/timing passes; harmony is generated once')
    parser.add_argument('--no-harmonize', action='store_true')
    parser.add_argument('--free-timing-pitches', action='store_true', help='Also allow pitch/velocity/count changes in timing responses')
    parser.add_argument('--temperature', type=float, default=.9)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    if args.passes < 1 or any(not 0 <= x <= 1 for x in (args.melody_strength, args.rhythm_strength)):
        parser.error('Passes must be positive and strengths must be in [0,1]')
    outputs = [args.output, *(args.output.with_suffix('.' + suffix + '.mid') for suffix in ('melody', 'refine', 'rhythm'))]
    if args.input.resolve() in {p.resolve() for p in outputs}:
        parser.error('Choose output paths distinct from the original MIDI')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    inference = MelodyInference(args.model, args.device)
    original = read_midi_notes(args.input, inference.tokenizer)
    if not original:
        raise ValueError('Input has no supported pitched notes')
    melody = melody_voice(skyline(original, inference.meta['skyline_tolerance_ms'])[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stem = args.output.with_suffix('')
    write_midi_notes(melody, str(stem) + '.melody.mid', inference.tokenizer)
    rng, stats = random.Random(args.seed), []
    for _ in range(args.passes):
        for task, strength in [('refine', args.melody_strength), ('rhythm', args.rhythm_strength)]:
            if args.task not in ('pipeline', task):
                continue
            melody, log = inference.transform(melody, task, args.genre, strength, rng,
                                               args.temperature, args.top_k,
                                               keep_pitches=not args.free_timing_pitches)
            stats.extend(log)
            write_midi_notes(melody, str(stem) + f'.{task}.mid', inference.tokenizer)
    result = melody
    if args.task == 'harmonize' or args.task == 'pipeline' and not args.no_harmonize:
        result, log = inference.transform(melody, 'harmonize', args.genre, 1., rng,
                                          args.temperature, args.top_k,
                                          output_instruments=sorted({n.instrument for n in original}))
        stats.extend(log)
    write_midi_notes(result, args.output, inference.tokenizer)
    report = dict(input=str(args.input), output=str(args.output), genre=args.genre,
                  source_notes=len(original), skyline_notes=len(melody_voice(skyline(original, inference.meta['skyline_tolerance_ms'])[0])),
                  output_notes=len(result), seed=args.seed, windows=stats)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'windows'}, indent=2))


if __name__ == '__main__':
    main()
