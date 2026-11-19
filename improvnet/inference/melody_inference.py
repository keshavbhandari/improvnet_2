"""Batched melody repair, rhythm/harmony and bounded stem addition."""
from bisect import bisect_right
from dataclasses import replace
from contextlib import nullcontext
import sys
from types import SimpleNamespace

import torch
from tqdm import tqdm
from improvnet.model.melody_model import MelodyRefinementModel, validate_checkpoint
from improvnet.utils.ar_utils import ProcessData
from improvnet.utils.melody import (MelodyTokenizer, TASKS, make_prompt, mask_timing,
    PREFIX_TOKENS, normalize_refinement_velocity, combine_notes, melody_voice, notes_to_tokens, ordered, skyline, window, harmonize_strength, tokens_to_notes)
from improvnet.utils.melody_chunks import (chunk_bounds, chunk_end, group_end,
                                          history_tokens, trim_history, stem_bounds, shorten_chunk)
from improvnet.utils.melody_corruptions import refinement_source, refinement_strength
from improvnet.model.encoders.runtime import FrozenConditioners
from improvnet.utils.melody_stems import percussion_mode

from improvnet.inference.melody_runtime import INFERENCE_SEED, TOP_K, inference_context
from improvnet.inference.melody_outputs import inference_output_paths


def deduplicate_attacks(notes, time_step_ms=10):
    seen, result = set(), []
    for n in notes:
        key = (n.instrument, n.pitch, round(n.onset / time_step_ms))
        if key not in seen:
            seen.add(key)
            result.append(n)
    return ordered(result)

class MelodyInference:
    def __init__(self, checkpoint_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
        self.meta = validate_checkpoint(checkpoint)
        self.checkpoint_step = checkpoint.get('step')
        self.tokenizer = MelodyTokenizer()
        self.options = SimpleNamespace(**{k.upper(): v for k, v in self.meta.items()})
        self.encoders=FrozenConditioners(self.meta['style_encoder'],self.meta['beat_encoder'],self.device,
            style_sha256=self.meta['style_encoder_sha256'],beat_sha256=self.meta['beat_encoder_sha256'],windows=self.meta['reference_windows'],
            local_bins=self.meta['reference_local_bins'])
        self.model = MelodyRefinementModel(**self.meta['model_kwargs'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
        self.model.to(self.device).eval()
        self.processor = ProcessData()

    def model_autocast(self):
        # A parallel worker only submits requests. Its coordinator owns the
        # autocast context, including the persistent inference weight cache.
        return nullcontext() if getattr(self, 'model_forward_is_proxy', False) else inference_context(self.device)

    def generate(self, *args, **kwargs):
        from improvnet.inference.melody_note_decode import generate_notes
        return generate_notes(self, *args, **kwargs)

    def transform(self, notes, task, genre, strength, rng, temperature=1.,
                  instruments=None, style_reference=None, monophonic=None, target_instrument=None, *, timing_strength=None, source_stream=None, on_chunk=None,percussion_palette='kit'):
        stage_label = ('Drums' if target_instrument == 'drum' else
                       'Bass' if target_instrument == 'Acoustic Bass' else
                       f'Stem {target_instrument}' if task == 'stem' else task.capitalize())
        if task not in self.meta.get('tasks', TASKS):
            raise ValueError(f'This checkpoint was not trained for {task}')
        if task == 'harmonize':
            strength = harmonize_strength(rng, strength, self.meta['harmonize_min_mask_ratio'])
        # Inference strength zero means an exact skyline pass-through, rather
        # than training's "ornament merges but no additional corruption" case.
        bypass_refinement = task == 'refine' and strength == 0
        refinement_timing = (harmonize_strength(rng,.95 if timing_strength is None else timing_strength,
            self.meta.get('refine_min_mask_ratio',.9)) if task=='refine' and not bypass_refinement else None)
        tok = self.tokenizer
        def prepare_source(raw):
            pitched = [n for n in raw if n.instrument != 'drum']
            source = list(raw) if task == 'stem' else pitched
            if task in ('refine', 'harmonize'):
                source = (melody_voice(pitched) if monophonic else
                          melody_voice(skyline(pitched, self.meta['skyline_tolerance_ms'])[0]))
            return source
        notes = ordered(notes)
        pitched = [n for n in notes if n.instrument != 'drum']
        source_notes = prepare_source(notes)
        if task in ('stem',) and target_instrument is not None:
            if ('instrument', target_instrument) not in tok.tok_to_id:
                raise ValueError('Choose a supported missing instrument')
            if any(n.instrument == target_instrument for n in source_notes):
                raise ValueError('Requested stem already exists; remove it before restoring it')
        if not source_notes:
            if on_chunk and source_notes:
                on_chunk(list(source_notes))
            while source_stream is not None and not source_stream.done:
                batch = source_stream.read()
                if batch is not None:
                    prepared = prepare_source(ordered(batch))
                    source_notes.extend(prepared)
                    if on_chunk and prepared:
                        on_chunk(prepared)
            return list(source_notes), []
        selected = sorted(set(instruments or [n.instrument for n in pitched]))
        if task == 'harmonize':
            selected = [i for i in selected if i != 'drum']
        if task in ('stem',):
            if task=='stem':
                selected=sorted(set(selected)|{n.instrument for n in source_notes})
            if target_instrument:
                selected=sorted(set(selected)|{target_instrument})
            if any(n.instrument=='drum' for n in source_notes):
                selected=sorted(set(selected)|{'drum'})
            if not any(i not in ('drum',target_instrument) for i in selected):
                raise ValueError('Choose at least one pitched output instrument')
        if task=='stem' and target_instrument is None:
            raise ValueError('Stem addition requires --target-instrument (pipeline defaults to missing drums)')
        reference = style_reference or {}
        condition={name:reference.get(name) for name in ('reference_features','reference_mask','reference_positions','drum_profile')}
        palette=reference.get('drum_palette') if target_instrument=='drum' else None
        if target_instrument=='drum' and reference and not palette:
            raise ValueError('Reference contains no standard-kit drum sounds')

        result, logs, begin = [], [], 0
        onsets = [n.onset for n in source_notes]
        with tqdm(total=len(source_notes), desc=stage_label, unit='input note', dynamic_ncols=True,
                  position=getattr(self, 'progress_position', None)) as progress:
            while begin < len(source_notes) or source_stream is not None and not source_stream.done:
                left = history_tokens(result, self.meta['history_tokens'], tok)
                available = self.meta['seq_len'] - PREFIX_TOKENS - len(left) - 4 - 2*int(target_instrument is not None)
                response_cap=self.meta['seq_len']
                input_cap=self.meta['chunk_tokens']
                # One complete lookahead group makes streaming chunk boundaries
                # match offline chunking, including rests and held releases.
                if source_stream is not None:
                    while not source_stream.done:
                        if begin < len(source_notes):
                            try:
                                ready_stop = chunk_end(source_notes, begin, input_cap)
                            except ValueError:
                                ready_stop = begin
                            # Pitch protection also needs the complete 50ms
                            # performed chord, even when it straddles chunks.
                            skyline_ready = (task != 'stem'
                                or ready_stop > begin and source_notes[-1].onset >
                                source_notes[ready_stop-1].onset+self.meta['skyline_tolerance_ms'])
                            if begin < ready_stop < len(source_notes) and skyline_ready:
                                break
                        batch = source_stream.read()
                        if batch is not None:
                            notes = combine_notes(notes, batch)
                            source_notes = prepare_source(notes)
                            onsets = [n.onset for n in source_notes]
                            progress.total = len(source_notes)
                    if begin >= len(source_notes):
                        break
                minimum = 4 * (group_end(source_notes, begin) - begin) + 2
                stop = chunk_end(source_notes, begin, input_cap)
                state = rng.getstate()
                while True:
                    rng.setstate(state)
                    bounds=stem_bounds if task=='stem' else chunk_bounds
                    start, end = bounds(source_notes, begin, stop)
                    core = window(source_notes, start, end)
                    chunk_strength = (refinement_strength(core, strength, self.options)
                                      if task == 'refine' and not bypass_refinement else strength)
                    if bypass_refinement:
                        source = notes_to_tokens(core, tok)
                    elif task == 'refine':
                        unmasked_source = normalize_refinement_velocity(refinement_source(core, strength, rng, tok, self.options))
                        source = mask_timing(tokens_to_notes(unmasked_source,tok),refinement_timing,rng,tok,round_up=True)
                    elif task == 'harmonize':
                        source = mask_timing(core, strength, rng, tok, round_up=True)
                    else:
                        source = notes_to_tokens(core, tok)
                    # Reserve a complete retained/source response before adding history.
                    response_floor = len(notes_to_tokens(core, tok)) - 1
                    room = self.meta['seq_len'] - PREFIX_TOKENS - len(make_prompt(task,source,(),target_instrument)) - response_floor
                    left = trim_history(left, max(0,room), tok)
                    available = self.meta['seq_len'] - PREFIX_TOKENS - len(left) - 4 - 2*int(target_instrument is not None)
                    if (len(source) <= self.meta['chunk_tokens']
                            and len(source) + response_floor <= available
                            and response_floor <= response_cap-1):
                        break
                    fraction = min(self.meta['chunk_tokens'] / len(source),
                                   available / (len(source) + response_floor),
                                   (response_cap-1) / response_floor)
                    stop = shorten_chunk(source_notes, begin, stop, fraction * .98)
                prompt = make_prompt(task, source, left, target_instrument)
                budget = min(response_cap - 1, available - len(source))
                output_instruments = selected if task in ('harmonize', 'stem') else sorted({n.instrument for n in core})
                self.last_decode_stats = {}
                protected_pitches = None
                if task == 'stem' and self.meta.get('stem_source_corruption_prob',1.)>0:
                    # Skyline uses the original performed attacks (50ms groups)
                    # across chunk edges, before rebasing the source window.
                    melody, _ = skyline([n for n in source_notes if n.instrument != 'drum'],
                                        self.meta['skyline_tolerance_ms'])
                    protected = set(melody)
                    protected_pitches = [replace(n, onset=n.onset+start) in protected for n in ordered(core)]
                progress.set_postfix(chunk=len(logs) + 1, source=len(source), history=len(left))
                if bypass_refinement:
                    generated = list(core)
                    self.last_decode_stats = dict(end_reason='zero_strength_passthrough', model_bypassed=True)
                else:
                    generated = self.generate(prompt, genre, chunk_strength, output_instruments, end - start,
                        temperature=temperature, monophonic=task == 'refine', max_new_tokens=None,
                        **condition,drum_palette=palette, stop_at_window_end=True,
                        fixed_notes=ordered(core) if task in ('stem',) else None,
                        stem_add=task=='stem', target_instrument=target_instrument,
                        protected_pitches=protected_pitches,
                        progress_desc=f'{stage_label} chunk {len(logs) + 1}',
                    percussion_palette=percussion_palette if target_instrument=='drum' else percussion_mode(core))
                if not generated:
                    raise RuntimeError(f'Empty {task} output; refusing to skip the source melody')
                consumed = stop
                if task in ('refine', 'harmonize'):
                    last = max(n.onset for n in generated)
                    slack = min(tok.abs_time_step_ms, max(tok.time_step_ms, core[-1].onset * .05))
                    if last + slack < core[-1].onset:
                        consumed = min(stop, max(begin + 1, bisect_right(onsets, start + last)))
                        tqdm.write(f'Early {task} end; continuing from {source_notes[consumed].onset / 1000:.2f}s', file=sys.stderr)
                published=[replace(n,onset=n.onset+start) for n in generated]
                result.extend(published)
                if on_chunk:
                    on_chunk(published)
                logs.append(dict(task=task, start_ms=start, end_ms=end, input_notes=len(core),
                    output_notes=len(generated), history_tokens=len(left), history_view='mono' if task == 'refine' else 'poly',
                    source_tokens=len(source), response_tokens=self.last_decode_stats.get(
                        'decoded_tokens',len(notes_to_tokens(generated, tok)) - 1),
                    response_token_budget=self.last_decode_stats.get('response_token_budget',budget),
                    history_tokens_dropped=self.last_decode_stats.get('history_tokens_dropped',0),
                    total_token_budget=len(prompt) + budget + 2,
                    consumed_input_notes=consumed - begin, strength=chunk_strength, requested_strength=strength, early_end=consumed < stop,
                    timing_mask_fraction=source.count('<MASK>')/(2*max(1,sum(isinstance(t,tuple) and t[0]=='note' for t in source))) if task in ('refine','harmonize') and not bypass_refinement else None,
                    target_instrument=target_instrument, added_stem_notes=sum(n.instrument == target_instrument for n in generated) if task in ('stem',) else None,
                    percussion_mode=percussion_palette if target_instrument=='drum' else percussion_mode(core),
                    reference_conditioned=condition['reference_features'] is not None, decoding=dict(self.last_decode_stats)))
                progress.update(consumed - begin)
                begin = consumed
        result=deduplicate_attacks(result,tok.time_step_ms)
        if task=='stem' and logs:
            added=sum(n.instrument==target_instrument for n in result)
            active=sum(x['added_stem_notes']>0 for x in logs)
            if not added:
                tqdm.write(f'No {target_instrument} notes generated; continuing with the retained mix.', file=sys.stderr)
            tqdm.write(f'Stem {target_instrument}: {added} notes in {active}/{len(logs)} chunks '
                       f'({len(logs)-active} rest chunks).',file=sys.stderr)
        return result, logs


def main(pipeline_runner=None):
    from improvnet.inference.melody_cli import run_cli
    run_cli(MelodyInference, pipeline_runner=pipeline_runner)


if __name__ == '__main__':
    main()
