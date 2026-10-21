"""One command interface shared by the canonical and compatibility entrypoints."""
import argparse
import json
from pathlib import Path
import random
import sys
import torch
from tqdm import tqdm
from improvnet.inference.melody_outputs import inference_output_paths
from improvnet.inference.melody_runtime import INFERENCE_SEED, TOP_K
from improvnet.model import melody_config as cfg
from improvnet.utils.ar_utils import ProcessData
from improvnet.utils.melody import TASKS, melody_voice, read_midi_notes, skyline, write_midi_notes
from improvnet.utils.melody_style import choose_reference, performance_statistics
from improvnet.utils.melody_groove import GrooveGuide, reference_windows
from improvnet.utils.melody_stems import kit_only_notes, DRUM_KIT_PITCHES, DRUM_KIT_ALIASES

def run_cli(inference_factory, pipeline_runner=None):
    parser = argparse.ArgumentParser(description='Refine melody, harmonize rhythm/harmony and add a stem with batched inference.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--genre', choices=['jazz', 'blues'], default='jazz')
    parser.add_argument('--task', choices=['pipeline', *TASKS],
                        metavar='{pipeline,refine,harmonize,stem}', default='pipeline')
    parser.add_argument('--instruments', nargs='+', choices=ProcessData().INSTRUMENT_CLASSES,
                        help='Requested output mix; missing drums are added after rhythm/harmony')
    parser.add_argument('--target-instrument', choices=ProcessData().INSTRUMENT_CLASSES,
                        help='Stem to add; pipeline defaults to drums when requested and absent')
    parser.add_argument('--melody-strength', type=float, default=.2)
    parser.add_argument('--rhythm-strength', type=float, default=.8,
                        help='Timing-mask fraction; V5 harmonize enforces at least 60%% (1 masks all timings)')
    parser.add_argument('--style-reference', type=Path, help='Rhythm reference MIDI')
    parser.add_argument('--no-rhythm-conditioning', '--no-style-conditioning', action='store_true')
    parser.add_argument('--temperature', type=float, default=1.)
    args = parser.parse_args()
    if any(not 0 <= x <= 1 for x in (args.melody_strength, args.rhythm_strength)):
        parser.error('Strengths must be in [0,1]')
    random.seed(INFERENCE_SEED)
    torch.manual_seed(INFERENCE_SEED)
    tqdm.write(f'Loading checkpoint: {args.model}', file=sys.stderr)
    infer = inference_factory(args.model)
    if pipeline_runner is None:
        from improvnet.inference.melody_pipeline import run_pipeline
        pipeline_runner = run_pipeline
    loaded_notes = read_midi_notes(args.input, infer.tokenizer, include_drums=True)
    original = kit_only_notes(loaded_notes)
    pitched = [n for n in original if n.instrument != 'drum']
    original_instruments = sorted({n.instrument for n in pitched})
    if not original or not pitched:
        raise ValueError('Input has no supported source music for this task')
    requested = sorted(set(args.instruments or original_instruments + (['drum'] if args.task == 'pipeline' else [])))
    if args.target_instrument:
        requested=sorted(set(requested)|{args.target_instrument})
    requested_pitched = [i for i in requested if i != 'drum']
    if not requested_pitched:
        parser.error('Choose at least one pitched output instrument')
    stem_targets=[]
    if args.task == 'pipeline':
        tasks = ['refine', 'harmonize']
        stem_targets = ['drum'] if 'drum' in requested and not any(n.instrument=='drum' for n in original) else []
        if args.target_instrument and args.target_instrument not in stem_targets:
            stem_targets.append(args.target_instrument)
        tasks.extend(['stem']*len(stem_targets))
    else:
        tasks = [args.task]
    reference = None
    if any(t in tasks for t in ('harmonize','stem')) and not args.no_rhythm_conditioning:
        if args.style_reference:
            ref = kit_only_notes(read_midi_notes(args.style_reference, infer.tokenizer,include_drums=True))
            ref_melody = melody_voice(skyline(ref, infer.meta['skyline_tolerance_ms'])[0])
            reference = dict(id=str(args.style_reference), melody_statistics=performance_statistics(ref_melody),
                             groove=reference_windows(ref,ref_melody))
        else:
            reference = choose_reference(cfg.STYLE_REFERENCES, args.genre, random.Random(INFERENCE_SEED),source_notes=original)
    guide=GrooveGuide(original,reference,random.Random(INFERENCE_SEED)) if reference else None
    outputs = inference_output_paths(args.output)
    written_names = {'final', 'report', 'melody', *tasks}
    if args.input.resolve() in {p.resolve() for name, p in outputs.items() if name in written_names}:
        parser.error('Output or an intermediate file would overwrite the input MIDI')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outputs['directory'].mkdir(parents=True, exist_ok=True)
    mono = melody_voice(skyline(pitched, infer.meta['skyline_tolerance_ms'])[0])
    write_midi_notes(mono, outputs['melody'], infer.tokenizer)
    current, chunks, task_order, extra_report = pipeline_runner(
        infer, original, tasks, args, requested, original_instruments,
        reference, guide, outputs['directory'], stem_targets)
    write_midi_notes(current, args.output, infer.tokenizer)
    report = dict(model=str(args.model), checkpoint_step=infer.checkpoint_step, protocol_version=infer.meta['version'],
                  input=str(args.input), output=str(args.output), genre=args.genre,
                  nonkit_input_drum_notes_filtered=sum(n.instrument=='drum' and n.pitch not in DRUM_KIT_PITCHES and n.pitch not in DRUM_KIT_ALIASES for n in loaded_notes),
                  input_drum_articulations_normalized=sum(n.instrument=='drum' and n.pitch in DRUM_KIT_ALIASES for n in loaded_notes),
                  requested_instruments=requested, instruments=sorted({n.instrument for n in current}),
                  missing_instruments=sorted(set(requested)-{n.instrument for n in current}),
                  task_order=task_order, output_notes=len(current), harmony_conditioning=False,
                  style_reference=reference, shared_clock=vars(guide.clock) if guide else None, seed=INFERENCE_SEED, top_k=TOP_K, device=str(infer.device), chunks=chunks)
    report['stem_summary'] = []
    for target in sorted({log['target_instrument'] for log in chunks if log['task'] == 'stem'}):
        added = sum(log['added_stem_notes'] for log in chunks
                    if log['task'] == 'stem' and log['target_instrument'] == target)
        report['stem_summary'].append(dict(instrument=target, generated_notes=added,
                                           status='generated' if added else 'not_generated'))
    report['files'] = {name: str(path) for name, path in outputs.items() if name != 'directory'
                       and (name in ('final', 'report', 'melody') or name in task_order)}
    report.update(extra_report)
    outputs['report'].write_text(json.dumps(report, indent=2))
    tqdm.write(f'Saved {args.output}', file=sys.stderr)
