"""Batched melody repair, rhythm/harmony and bounded stem addition."""
from bisect import bisect_right
from dataclasses import replace
from contextlib import nullcontext
import sys
from types import SimpleNamespace

import torch
from tqdm import tqdm
from improvnet.model.melody_model import MelodyRefinementModel, validate_checkpoint
from improvnet.model.melody_train import autocast
from improvnet.utils.ar_utils import ProcessData
from improvnet.utils.melody import (MelodyTokenizer, TASKS, make_prompt, mask_timing,
    melody_voice, notes_to_tokens, ordered, skyline, window, harmonize_strength)
from improvnet.utils.melody_chunks import (chunk_bounds, chunk_end, group_end,
                                          history_tokens, stem_bounds, shorten_chunk)
from improvnet.utils.melody_corruptions import refinement_source, refinement_strength
from improvnet.utils.melody_style import style_tensors
from improvnet.utils.melody_groove import GrooveGuide
from improvnet.utils.melody_stems import kit_only_notes

from improvnet.inference.melody_runtime import INFERENCE_SEED, TOP_K
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
        self.model = MelodyRefinementModel(**dict(self.meta['model_kwargs'], style_feature_count=len(self.meta['style_features'])))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
        self.model.to(self.device).eval()
        self.processor = ProcessData()

    def model_autocast(self):
        # A parallel worker only submits requests. Its coordinator owns the
        # autocast context, including the persistent inference weight cache.
        return nullcontext() if getattr(self, 'model_forward_is_proxy', False) else autocast(self.device)

    def generate(self, *args, **kwargs):
        from improvnet.inference.melody_note_decode import generate_notes
        return generate_notes(self, *args, **kwargs)

    def transform(self, notes, task, genre, strength, rng, temperature=1.,
                  instruments=None, style_reference=None, monophonic=None, target_instrument=None, groove_guide=None, *, source_stream=None, on_chunk=None):
        if task not in self.meta.get('tasks', TASKS):
            raise ValueError(f'This checkpoint was not trained for {task}')
        if task == 'harmonize' and self.meta.get('version', 0) >= 9:
            strength = harmonize_strength(rng, strength, self.meta.get('harmonize_min_mask_ratio', .6))
        tok = self.tokenizer
        def prepare_source(raw):
            pitched = [n for n in raw if n.instrument != 'drum']
            source = kit_only_notes(raw) if task == 'stem' else pitched
            if task in ('refine', 'harmonize'):
                source = (melody_voice(pitched) if monophonic else
                          melody_voice(skyline(pitched, self.meta['skyline_tolerance_ms'])[0]))
            return source
        notes = ordered(kit_only_notes(notes))
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
        if groove_guide is None and reference:
            groove_guide=GrooveGuide(notes,reference,rng)
        statistics = reference.get('melody_statistics', reference.get('statistics', reference)) if task in ('harmonize','stem') else {}
        values, mask = style_tensors(statistics, task,**({'feature_names':self.meta['style_features']} if 'style_features' in self.meta else {}))

        result, logs, begin = [], [], 0
        onsets = [n.onset for n in source_notes]
        with tqdm(total=len(source_notes), desc=task.capitalize(), unit='input note', dynamic_ncols=True,
                  position=getattr(self, 'progress_position', None)) as progress:
            while begin < len(source_notes) or source_stream is not None and not source_stream.done:
                left = history_tokens(result, self.meta['history_tokens'], tok)
                available = self.meta['seq_len'] - len(left) - 6 - 2*int(target_instrument is not None)
                response_cap = (self.meta['mix_response_tokens']
                                if task != 'refine' else self.meta['chunk_tokens'])
                input_cap = self.meta['chunk_tokens']
                if task == 'harmonize':
                    input_cap = min(input_cap, int(response_cap / self.meta['harmonize_expansion']))
                # One complete lookahead group makes streaming chunk boundaries
                # match offline chunking, including rests and held releases.
                if source_stream is not None:
                    while not source_stream.done:
                        if begin < len(source_notes):
                            try:
                                ready_stop = chunk_end(source_notes, begin, input_cap)
                            except ValueError:
                                ready_stop = begin
                            # V5 pitch protection also needs the complete 50ms
                            # performed chord, even when it straddles chunks.
                            skyline_ready = (task != 'stem' or self.meta.get('version', 0) < 9
                                or ready_stop > begin and source_notes[-1].onset >
                                source_notes[ready_stop-1].onset+self.meta['skyline_tolerance_ms'])
                            if begin < ready_stop < len(source_notes) and skyline_ready:
                                break
                        batch = source_stream.read()
                        if batch is not None:
                            notes = ordered(notes + batch)
                            source_notes = prepare_source(notes)
                            onsets = [n.onset for n in source_notes]
                            progress.total = len(source_notes)
                    if begin >= len(source_notes):
                        break
                minimum = 4 * (group_end(source_notes, begin) - begin) + 2
                input_cap = max(minimum, min(input_cap, available // 2))
                stop = chunk_end(source_notes, begin, input_cap)
                bounded, state = False, rng.getstate()
                while True:
                    rng.setstate(state)
                    bounds=stem_bounds if task=='stem' else chunk_bounds
                    start, end = bounds(source_notes, begin, stop)
                    core = window(source_notes, start, end)
                    chunk_strength = refinement_strength(core, strength, self.options) if task == 'refine' else strength
                    if task == 'refine':
                        source = refinement_source(core, strength, rng, tok, self.options, bounded)
                    elif task == 'harmonize':
                        source = mask_timing(core, strength, rng, tok, round_up=True)
                    else:
                        source = notes_to_tokens(core, tok)
                    if len(source) <= self.meta['chunk_tokens'] and len(source) + len(notes_to_tokens(core, tok)) - 1 <= available:
                        break
                    fraction = min(self.meta['chunk_tokens'] / len(source),
                                   available / (len(source) + len(notes_to_tokens(core, tok))))
                    try:
                        stop = shorten_chunk(source_notes, begin, stop, fraction * .98)
                    except ValueError:
                        if task != 'refine' or bounded:
                            raise
                        bounded = True
                prompt = make_prompt(task, source, left, target_instrument)
                budget = min(response_cap - 1, available - len(source))
                output_instruments = selected if task in ('harmonize', 'stem') else sorted({n.instrument for n in core})
                if groove_guide is not None and task in ('harmonize','stem'):
                    statistics=groove_guide.statistics(start,end)
                    values,mask=style_tensors(statistics,task,**({'feature_names':self.meta['style_features']} if 'style_features' in self.meta else {}))
                self.last_decode_stats = {}
                protected_pitches = None
                if task == 'stem' and self.meta.get('version', 0) >= 9:
                    # Skyline uses the original performed attacks (50ms groups)
                    # across chunk edges, before rebasing the source window.
                    melody, _ = skyline([n for n in source_notes if n.instrument != 'drum'],
                                        self.meta['skyline_tolerance_ms'])
                    protected = set(melody)
                    protected_pitches = [replace(n, onset=n.onset+start) in protected for n in ordered(core)]
                progress.set_postfix(chunk=len(logs) + 1, source=len(source), history=len(left))
                generated = self.generate(prompt, genre, chunk_strength, output_instruments, end - start,
                    temperature=temperature, monophonic=task == 'refine', max_new_tokens=budget,
                    style_values=values, style_mask=mask, stop_at_window_end=task in ('harmonize', 'stem'),
                    fixed_notes=ordered(core) if task in ('stem',) else None,
                    stem_add=task=='stem', target_instrument=target_instrument,
                    protected_pitches=protected_pitches,
                    progress_desc=f'{task.capitalize()} chunk {len(logs) + 1}')
                if not generated:
                    raise RuntimeError(f'Empty {task} output; refusing to skip the source melody')
                consumed = stop
                if task in ('refine', 'harmonize'):
                    last = max(n.onset for n in generated)
                    slack = min(tok.abs_time_step_ms, max(tok.time_step_ms, core[-1].onset * .05))
                    if last + slack < core[-1].onset:
                        consumed = min(stop, max(begin + 1, bisect_right(onsets, start + last)))
                        tqdm.write(f'Early {task} end; continuing from {source_notes[consumed].onset / 1000:.2f}s', file=sys.stderr)
                published = [replace(n, onset=n.onset + start) for n in generated]
                result.extend(published)
                if on_chunk:
                    on_chunk(published)
                logs.append(dict(task=task, start_ms=start, end_ms=end, input_notes=len(core),
                    output_notes=len(generated), history_tokens=len(left), history_view='mono' if task == 'refine' else 'poly',
                    source_tokens=len(source), response_tokens=len(notes_to_tokens(generated, tok)) - 1,
                    response_token_budget=budget, total_token_budget=len(prompt) + budget + 2,
                    consumed_input_notes=consumed - begin, strength=chunk_strength, requested_strength=strength, early_end=consumed < stop,
                    timing_mask_fraction=source.count('<MASK>')/(2*len(core)) if task == 'harmonize' else None,
                    target_instrument=target_instrument, added_stem_notes=sum(n.instrument == target_instrument for n in generated) if task in ('stem',) else None,
                    style_statistics=statistics, style_mask=mask.tolist(), decoding=dict(self.last_decode_stats)))
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
