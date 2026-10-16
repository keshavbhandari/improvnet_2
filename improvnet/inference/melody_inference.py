"""MIDI -> configurable refinement/timing/harmony order -> targeted stem addition."""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
from bisect import bisect_left, bisect_right

import torch
from tqdm import tqdm

from improvnet.model.melody_model import MelodyRefinementModel, validate_checkpoint
from improvnet.model.melody_train import autocast
from improvnet.utils.ar_utils import ProcessData
from improvnet.utils.melody import (MelodyTokenizer, corrupt_melody, make_prompt,
                                   mask_timing, melody_voice, notes_to_tokens, ordered,
                                   read_midi_notes, skyline, tokens_to_notes,
                                   window, write_midi_notes, TASKS)
from improvnet.utils.melody_chunks import (chunk_bounds, chunk_end, history_tokens,
                                          shorten_chunk, group_end)
from improvnet.utils.melody_style import choose_reference, performance_statistics, style_tensors
from improvnet.model import melody_config as cfg


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


def merge_additions(generated, anchors, tokenizer, max_tokens, target_instrument=None):
    """Keep input notes exactly; fit added whole onset groups into the response."""
    merged = ([n for n in generated if n.instrument == target_instrument] + list(anchors)
              if target_instrument is not None else preserve_melody(generated, anchors))
    pending = Counter(anchors)
    extras = []
    for n in ordered(merged):
        if pending[n]:
            pending[n] -= 1
        else:
            extras.append(n)
    result, begin = list(anchors), 0
    if len(notes_to_tokens(result, tokenizer)) > max_tokens:
        raise ValueError('Input anchors exceed the complete response token budget')
    while begin < len(extras):
        stop = group_end(extras, begin)
        candidate = result + extras[begin:stop]
        if len(notes_to_tokens(candidate, tokenizer)) <= max_tokens:
            result = candidate
        begin = stop
    return ordered(result)


class MelodyInference:
    def __init__(self, checkpoint_path, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
        self.checkpoint_step = checkpoint.get('step')
        self.meta = validate_checkpoint(checkpoint)
        self.tokenizer = MelodyTokenizer()
        self.model = MelodyRefinementModel(**self.meta['model_kwargs'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
        self.model.to(self.device).eval()
        self.processor = ProcessData()

    @torch.no_grad()
    def generate(self, prompt, genre, strength, instruments, window_ms,
                 temperature=0.9, top_k=50, fixed_notes=None, monophonic=False,
                 max_new_tokens=None, progress=False, progress_desc='Decoding tokens',
                 style_values=None, style_mask=None):
        """Generate every attribute, with a note grammar and optional pitch anchors.

        Fixed-note decoding is for timing edits: instrument/pitch/velocity and
        count are anchored; onsets, durations and sequence end are decoded.
        """
        tok, device = self.tokenizer, self.device
        if genre not in self.meta['genres'] or not 0 <= strength <= 1:
            raise ValueError('Unknown genre or strength outside [0,1]')
        budget = self.meta['seq_len'] - len(prompt) - 2
        if max_new_tokens is not None:
            budget = min(budget, max_new_tokens)
        fixed_onsets = [n.onset for n in fixed_notes] if fixed_notes is not None else []
        fixed_attacks = sorted(set(fixed_onsets))
        fixed_length = 4 * len(fixed_onsets) + (fixed_onsets[-1] // tok.abs_time_step_ms if fixed_onsets else 0) + 1
        if budget < 5 or (fixed_notes is not None and budget < fixed_length):
            raise ValueError('Prompt leaves insufficient room for a complete response')
        with tqdm(total=budget, desc=progress_desc, unit='tok', position=1,
                  leave=False, dynamic_ncols=True, disable=not progress) as token_progress:
            conditions = self.processor.get_instrument_multihot(
                [('instrument', inst) for inst in instruments]).unsqueeze(0).to(device)
            token_progress.set_postfix_str('prefill')
            ids = torch.tensor([[tok.tok_to_id[t] for t in prompt]], device=device)
            with autocast(device):
                logits, cache = self.model(ids, genre=torch.tensor([self.meta['genres'].index(genre)], device=device),
                                           multi_hot=conditions, strength=torch.tensor([strength], device=device),
                                           style_values=style_values.unsqueeze(0).to(device) if style_values is not None else None,
                                           style_mask=style_mask.unsqueeze(0).to(device) if style_mask is not None else None,
                                           use_cache=True)
            token_progress.set_postfix_str('decoding')
            offset = len(prompt) + 2
            output, stage, segment, last_onset, count = ['<S>'], 0, 0, -tok.time_step_ms, 0
            inst_ids = [tok.tok_to_id[('instrument', inst)] for inst in instruments]
            if not inst_ids:
                raise ValueError('No supported output instrument')
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
                        minimum = max(segment, last_onset + (10 if monophonic else 0))
                        if len(output) - 1 + 5 <= budget and minimum < min(segment + tok.abs_time_step_ms, window_ms):
                            legal += inst_ids
                        if segment + tok.abs_time_step_ms < window_ms and len(output) - 1 + 6 <= budget:
                            legal.append(tok.tok_to_id['<T>'])
                elif stage == 1:
                    legal = ([tok.tok_to_id[('note', fixed_notes[count].pitch,
                              tok._quantize_velocity(fixed_notes[count].velocity))]]
                             if fixed_notes is not None else pitch_ids)
                elif stage == 2:
                    # Keep notes ordered, permit simultaneous harmony notes.
                    minimum = max(segment, last_onset)
                    same_attack = fixed_notes is not None and count > 0 and fixed_onsets[count] == fixed_onsets[count - 1]
                    if fixed_notes is not None and not same_attack or monophonic:
                        minimum = max(segment, last_onset + 10)
                    maximum = min(window_ms, segment + tok.abs_time_step_ms)
                    if fixed_notes is not None:
                        # Reserve ticks for remaining notes in this five-second bin,
                        # rather than forcing a later note past its required <T>.
                        remaining_here = bisect_left(fixed_attacks, segment + tok.abs_time_step_ms) - bisect_left(fixed_attacks, fixed_onsets[count]) - 1
                        maximum -= 10 * remaining_here
                    legal = [tok.tok_to_id[t] for t in tok.onset_tokens
                             if minimum <= segment + t[1] < maximum]
                    if same_attack:
                        legal = [tok.tok_to_id[('onset', last_onset - segment)]]
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
                token_progress.update(1)
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
                  keep_pitches=True, output_instruments=None, progress=False,
                  progress_desc=None, style_reference=None, history_view='match', monophonic=None,
                  target_instrument=None):
        if task not in TASKS or history_view not in ('mono', 'poly', 'match'):
            raise ValueError('Unknown task or history view')
        if task == 'stem':
            if ('instrument', target_instrument) not in self.tokenizer.tok_to_id:
                raise ValueError('Choose a supported target instrument for stem generation')
            if any(n.instrument == target_instrument for n in melody):
                raise ValueError('Target instrument is already present; remove that stem from the input first')
            monophonic = False
        if task in ('harmonize', 'stem'):
            history_view = 'poly'  # Match the complete clean target used during training.
        if task in ('refine', 'rhythm') and strength == 0:
            return list(melody), []
        tok = self.tokenizer
        passthrough_drums = [n for n in melody if n.instrument == 'drum'] if task != 'stem' else []
        melody = ordered(melody if task == 'stem' else [n for n in melody if n.instrument != 'drum'])
        result, stats = [], []
        if not melody:
            return result + passthrough_drums, stats
        input_mono = (len(melody_voice(melody)) == len(melody)) if monophonic is None else monophonic
        if history_view == 'match':
            history_view = 'mono' if input_mono else 'poly'
        melody_onsets = [n.onset for n in melody]
        reference = style_reference or {}
        style_statistics = (reference.get('melody_statistics', reference.get('statistics', reference))
                            if task == 'rhythm' and input_mono else reference.get('statistics', reference))
        style_values, style_mask = style_tensors(style_statistics, task)
        begin = 0
        with tqdm(total=len(melody), desc=progress_desc or task.capitalize(), unit='input note',
                  position=0, dynamic_ncols=True, disable=not progress) as note_progress:
            while begin < len(melody):
                # Every stage starts without history and uses only its generated output.
                history_notes = result
                # A generated mono target already has the selected view; running
                # skyline again could discard rapid, overlapping melody attacks.
                if history_view == 'mono' and not input_mono:
                    pitched_history = melody_voice(skyline([n for n in result if n.instrument != 'drum'],
                                                           self.meta.get('skyline_tolerance_ms', 50))[0])
                    history_notes = ordered(pitched_history + [n for n in result if n.instrument == 'drum'])
                left = history_tokens(history_notes, self.meta['history_tokens'], tok)
                available = self.meta['seq_len'] - len(left) - 6 - int(task == 'stem')
                if task == 'harmonize':
                    expansion = max(self.meta['harmony_expansion'],
                                    1 + style_statistics.get('accompaniment_voicing_density', 0))
                    input_cap = min(int(self.meta['chunk_tokens'] / expansion),
                                    int(available / (1 + expansion)))
                else:
                    input_cap = min(self.meta['chunk_tokens'], available // 2)
                # A conservative expansion estimate must not split a chord
                # that still fits the real input/response limits.
                minimum_group = 4 * (group_end(melody, begin) - begin) + 2
                if input_cap < minimum_group <= self.meta['chunk_tokens'] and 2 * minimum_group <= available:
                    input_cap = minimum_group
                stop = chunk_end(melody, begin, input_cap)
                corruption_rng = rng.getstate()
                bounded_corruption = False
                while True:
                    rng.setstate(corruption_rng)
                    start, end = chunk_bounds(melody, begin, stop)
                    core = window(melody, start, end)
                    if task == 'refine':
                        options = {'names': ['whole_mask']} if bounded_corruption else {}
                        source = corrupt_melody(core, strength, rng, tok, self.meta['max_corruptions'], **options)
                    elif task == 'rhythm':
                        source = mask_timing(core, strength, rng, tok)
                    else:
                        source = notes_to_tokens(core, tok)
                    clean_reply = len(notes_to_tokens(core, tok)) - 1
                    if len(source) <= self.meta['chunk_tokens'] and len(source) + clean_reply <= available:
                        break
                    fraction = min(self.meta['chunk_tokens'] / len(source), available / (len(source) + clean_reply))
                    try:
                        stop = shorten_chunk(melody, begin, stop, fraction * .98)
                    except ValueError:
                        if task != 'refine' or bounded_corruption or clean_reply + 1 > self.meta['chunk_tokens']:
                            raise
                        bounded_corruption = True
                prompt = make_prompt(task, source, left, target_instrument)
                reply_budget = available - len(source)
                if task != 'stem':
                    reply_budget = min(self.meta['chunk_tokens'] - 1, reply_budget)
                instruments = (output_instruments if task == 'harmonize' and output_instruments
                               else sorted({n.instrument for n in core}))
                if task == 'stem':
                    instruments = sorted(set(instruments) | {target_instrument})
                note_progress.set_postfix(chunk=len(stats) + 1, source=len(source), history=len(left))
                generated = self.generate(prompt, genre, strength, instruments, end - start,
                                          temperature, top_k,
                                          fixed_notes=ordered(core) if task == 'rhythm' and keep_pitches else None,
                                          monophonic=input_mono and task in ('refine', 'rhythm'), max_new_tokens=reply_budget,
                                          progress=progress, progress_desc=f'Chunk {len(stats) + 1} tokens (limit)',
                                          style_values=style_values, style_mask=style_mask)
                consumed_stop, early_end = stop, False
                if task == 'refine':
                    if not generated:
                        raise RuntimeError('Empty refinement response; refusing to skip the input chunk')
                    last_attack = max(n.onset for n in generated)
                    ending_slack = min(tok.abs_time_step_ms, max(tok.time_step_ms, core[-1].onset * .05))
                    if last_attack + ending_slack < core[-1].onset:
                        consumed_stop = min(stop, max(begin + 1, bisect_right(melody_onsets, start + last_attack)))
                        early_end = consumed_stop < stop
                        if early_end and progress:
                            tqdm.write(f'Early refinement end; continuing input from '
                                       f'{melody[consumed_stop].onset / 1000:.2f}s', file=sys.stderr)
                if task in ('harmonize', 'stem'):
                    generated = merge_additions(generated, core, tok, reply_budget + 1,
                                                target_instrument=target_instrument if task == 'stem' else None)
                result.extend(replace(n, onset=n.onset + start) for n in generated)
                response_tokens = len(notes_to_tokens(generated, tok)) - 1
                if len(prompt) + response_tokens + 2 > self.meta['seq_len']:
                    raise RuntimeError('Complete output exceeds the context budget')
                stats.append(dict(start_ms=start, end_ms=end, input_notes=len(core), output_notes=len(generated),
                                  task=task, strength=strength, history_view=history_view,
                                  target_instrument=target_instrument,
                                  history_tokens=len(left), source_tokens=len(source),
                                  response_tokens=response_tokens, response_token_budget=reply_budget,
                                  total_token_budget=len(prompt) + reply_budget + 2,
                                  consumed_input_notes=consumed_stop - begin, early_end=early_end,
                                  input_mono=input_mono, style_mask=style_mask.tolist(),
                                  bounded_corruption=bounded_corruption,
                                  advance_to_ms=melody[consumed_stop].onset if consumed_stop < len(melody) else end))
                note_progress.update(consumed_stop - begin)
                begin = consumed_stop
        if task == 'harmonize':
            result = preserve_melody(result, melody)
        return ordered(result + passthrough_drums), stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--genre', choices=['jazz', 'blues'], default='jazz')
    parser.add_argument('--task', choices=['pipeline', *TASKS], default='pipeline')
    parser.add_argument('--pipeline-order', default=','.join(TASKS), help='Comma-separated task order; stem addition, if selected, must be last')
    parser.add_argument('--target-instrument', default='drum', choices=ProcessData().INSTRUMENT_CLASSES,
                        help='Missing instrument to add in the stem task (default: drum)')
    parser.add_argument('--input-view', choices=['auto', 'mono', 'poly'], default='auto', help='Auto extracts a melody for the pipeline and retains polyphony for individual tasks')
    parser.add_argument('--history-view', choices=['match', 'mono', 'poly'], default='match',
                        help='Default matches the target view; harmony and stem addition use the complete generated mix')
    parser.add_argument('--style-reference', type=Path, help='Use aggregate controls from this MIDI')
    parser.add_argument('--style-presets', type=Path, default=cfg.STYLE_REFERENCES)
    parser.add_argument('--no-style-conditioning', action='store_true')
    parser.add_argument('--melody-strength', type=float, default=.2)
    parser.add_argument('--rhythm-strength', type=float, default=.5)
    parser.add_argument('--passes', type=int, default=1, help='Repeat pitched tasks in order; the requested stem is added once at the end')
    parser.add_argument('--no-harmonize', action='store_true')
    parser.add_argument('--no-stem', '--no-drums', dest='no_stem', action='store_true')
    parser.add_argument('--free-timing-pitches', action='store_true', help='Also allow pitch/velocity/count changes in timing responses')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default=None)
    parser.add_argument('--no-progress', action='store_true', help='Hide progress bars and status messages')
    args = parser.parse_args()
    if args.passes < 1 or any(not 0 <= x <= 1 for x in (args.melody_strength, args.rhythm_strength)):
        parser.error('Passes must be positive and strengths must be in [0,1]')
    order = [x.strip() for x in args.pipeline_order.split(',')] if args.task == 'pipeline' else [args.task]
    if not order or len(set(order)) != len(order) or any(x not in TASKS for x in order):
        parser.error('Pipeline order must contain distinct valid task names')
    if args.no_harmonize:
        order = [x for x in order if x != 'harmonize']
    if args.no_stem:
        order = [x for x in order if x != 'stem']
    if not order or 'stem' in order and order[-1] != 'stem':
        parser.error('Choose at least one task and place stem addition last')
    outputs = [args.output, args.output.with_suffix('.json'), *(args.output.with_suffix('.' + suffix + '.mid')
                for suffix in ('melody', 'input', *TASKS))]
    if args.input.resolve() in {p.resolve() for p in outputs}:
        parser.error('Choose output paths distinct from the original MIDI')
    reference = None
    needed = tuple(t for t in ('rhythm', 'harmonize') if t in order)
    if needed and not args.no_style_conditioning:
        if args.style_reference:
            reference_notes = read_midi_notes(args.style_reference)
            if not reference_notes:
                parser.error('Style reference must contain supported pitched notes')
            reference_melody = melody_voice(skyline(reference_notes, cfg.SKYLINE_TOLERANCE_MS)[0])
            reference = dict(id=str(args.style_reference),
                             statistics=performance_statistics(reference_notes, reference_melody),
                             melody_statistics=performance_statistics(reference_melody, reference_melody))
        else:
            if not args.style_presets.is_file():
                parser.error('Style presets are missing; run melody_style_references or provide --style-reference/--no-style-conditioning')
            reference = choose_reference(args.style_presets, args.genre, random.Random(args.seed), needed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    progress = not args.no_progress
    if progress:
        tqdm.write(f'Loading checkpoint: {args.model}', file=sys.stderr)
        if reference and 'harmonize' in needed and not style_tensors(reference['statistics'], 'harmonize')[1].any():
            tqdm.write('Reference has no accompaniment; harmony controls are omitted.', file=sys.stderr)
    inference = MelodyInference(args.model, args.device)
    if progress:
        tqdm.write(f'Model ready on {inference.device}; reading {args.input}', file=sys.stderr)
    original = read_midi_notes(args.input, inference.tokenizer, include_drums=True)
    pitched = [n for n in original if n.instrument != 'drum']
    if not pitched and not (order == ['stem'] and original):
        raise ValueError('Input has no supported pitched notes')
    melody = melody_voice(skyline(pitched, inference.meta['skyline_tolerance_ms'])[0])
    input_view = ('mono' if args.task == 'pipeline' and len(order) > 1 else 'poly') if args.input_view == 'auto' else args.input_view
    current = melody if input_view == 'mono' else original
    is_mono = input_view == 'mono'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stem = args.output.with_suffix('')
    write_midi_notes(melody, str(stem) + '.melody.mid', inference.tokenizer)
    write_midi_notes(current, str(stem) + '.input.mid', inference.tokenizer)
    if progress:
        tqdm.write(f'Input view: {input_view}, {len(current)} notes; style reference: '
                   f'{reference["id"] if reference else "none"}', file=sys.stderr)
    rng, stats = random.Random(args.seed), []
    for pass_idx in range(args.passes):
        for task in (t for t in order if t != 'stem'):
            strength = args.melody_strength if task == 'refine' else args.rhythm_strength if task == 'rhythm' else 1.
            current, log = inference.transform(current, task, args.genre, strength, rng,
                                               args.temperature, args.top_k,
                                               keep_pitches=not args.free_timing_pitches,
                                               output_instruments=sorted({n.instrument for n in pitched}) if task == 'harmonize' else None,
                                               progress=progress,
                                               progress_desc=f'{task.capitalize()} pass {pass_idx + 1}/{args.passes}',
                                               style_reference=reference, history_view=args.history_view, monophonic=is_mono)
            if task == 'harmonize':
                is_mono = False
            stats.extend(log)
            write_midi_notes(current, str(stem) + f'.{task}.mid', inference.tokenizer)
            if progress:
                tqdm.write(f'Saved {stem}.{task}.mid ({len(current)} notes)', file=sys.stderr)
    result = current
    if 'stem' in order:
        result, log = inference.transform(current, 'stem', args.genre, 1., rng,
                                          args.temperature, args.top_k, progress=progress,
                                          progress_desc='Add ' + args.target_instrument,
                                          history_view='poly', monophonic=False,
                                          target_instrument=args.target_instrument)
        stats.extend(log)
        write_midi_notes(result, str(stem) + '.stem.mid', inference.tokenizer)
    write_midi_notes(result, args.output, inference.tokenizer)
    report = dict(model=str(args.model), checkpoint_step=inference.checkpoint_step,
                  input=str(args.input), output=str(args.output), genre=args.genre,
                  melody_strength=args.melody_strength, rhythm_strength=args.rhythm_strength,
                  passes=args.passes, temperature=args.temperature, top_k=args.top_k,
                  task_order=order, input_view=input_view, history_view=args.history_view,
                  target_instrument=args.target_instrument if 'stem' in order else None,
                  style_reference=reference,
                  source_notes=len(original), skyline_notes=len(melody),
                  output_notes=len(result), drum_notes=sum(n.instrument == 'drum' for n in result),
                  seed=args.seed, chunks=stats)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    if progress:
        tqdm.write(f'Saved {args.output} and {args.output.with_suffix(".json")}', file=sys.stderr)
    print(json.dumps({k: v for k, v in report.items() if k != 'chunks'}, indent=2))


if __name__ == '__main__':
    main()
