"""One command interface shared by the canonical and compatibility entrypoints."""
import argparse
import json
from pathlib import Path
import random
import sys
import torch
from tqdm import tqdm
from improvnet.inference.melody_outputs import inference_output_paths, stage_output_names, plan_stages
from improvnet.inference.melody_runtime import INFERENCE_SEED, TOP_K
from improvnet.utils.ar_utils import ProcessData
from improvnet.utils.melody import TASKS, melody_voice, read_midi_notes, skyline, write_midi_notes
from improvnet.model.encoders.reference import choose_reference, reference_report

def run_cli(inference_factory, pipeline_runner=None):
    parser = argparse.ArgumentParser(description='Refine melody, harmonize and add drums/bass with batched inference.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='Run name/path: claire_v12.mid writes all outputs inside claire_v12/, including final.mid')
    parser.add_argument('--genre', choices=['jazz', 'blues'], default='jazz')
    parser.add_argument('--task', choices=['pipeline', *TASKS],
                        metavar='{pipeline,refine,harmonize,stem}', default='pipeline')
    parser.add_argument('--instruments', nargs='+', choices=ProcessData().INSTRUMENT_CLASSES,
                        help='Requested mix; pipeline defaults to piano/source instruments, drums and acoustic bass')
    parser.add_argument('--target-instrument', choices=ProcessData().INSTRUMENT_CLASSES,
                        help='Stem to add; pipeline adds requested drums then acoustic bass after harmonization')
    parser.add_argument('--melody-strength', type=float, default=.2,
                        help='Refinement corruption strength; 0 passes through the extracted skyline without generation')
    parser.add_argument('--rhythm-strength', type=float, default=.95,
                        help='Timing-mask fraction; Harmonize enforces at least 90%% (1 masks all timings)')
    parser.add_argument('--reference-mode', choices=['exact','trio'], default='trio',
                        help='Match each stage output exactly or use its piano/drums/acoustic-bass subset')
    parser.add_argument('--style-reference', type=Path, help='Reference MIDI; shared random window by default, or averaged reference features')
    parser.add_argument('--reference-aggregation', choices=['window','average'], default='window',
                        help='Use one random shared excerpt, or average global/local features across all eligible excerpts')
    parser.add_argument('--reference-id', help='Select a jazz/blues entry from reference_catalog.json')
    parser.add_argument('--reference-bucket', help='Sample a matching beat/style similarity bucket from the catalogue')
    parser.add_argument('--temperature', type=float, default=1.)
    args = parser.parse_args()
    if any(not 0 <= x <= 1 for x in (args.melody_strength, args.rhythm_strength)):
        parser.error('Strengths must be in [0,1]')
    if args.style_reference and (args.reference_id or args.reference_bucket):
        parser.error('Choose either a reference MIDI or a catalogue reference ID/bucket')
    random.seed(INFERENCE_SEED)
    torch.manual_seed(INFERENCE_SEED)
    tqdm.write(f'Loading checkpoint: {args.model}', file=sys.stderr)
    infer = inference_factory(args.model)
    if pipeline_runner is None:
        from improvnet.inference.melody_pipeline import run_pipeline
        pipeline_runner = run_pipeline
    loaded_notes = read_midi_notes(args.input, infer.tokenizer, include_drums=True)
    original = list(loaded_notes)
    pitched = [n for n in original if n.instrument != 'drum']
    original_instruments = sorted({n.instrument for n in pitched})
    if not original or not pitched:
        raise ValueError('Input has no supported source music for this task')
    requested = sorted(set(args.instruments or original_instruments +
                           (['drum', 'Acoustic Bass'] if args.task == 'pipeline' else [])))
    if args.target_instrument:
        requested=sorted(set(requested)|{args.target_instrument})
    requested_pitched = [i for i in requested if i != 'drum']
    if not requested_pitched:
        parser.error('Choose at least one pitched output instrument')
    stem_targets=[]
    if args.task == 'pipeline':
        tasks = ['refine', 'harmonize']
        # Harmonization rebuilds the pitched mix, so source drums never survive
        # into the next stage. Schedule requested drums against that mix.
        stem_targets = [instrument for instrument in ('drum', 'Acoustic Bass') if instrument in requested]
        if args.target_instrument and args.target_instrument not in stem_targets:
            stem_targets.append(args.target_instrument)
        tasks.extend(['stem']*len(stem_targets))
    else:
        tasks = [args.task]
    stage_names = stage_output_names(tasks,stem_targets)
    planned = plan_stages(original,tasks,requested,original_instruments,stem_targets,
                          task_mode=args.task,target_instrument=args.target_instrument)
    stage_instruments = {name:selected for name,(task,_,selected) in zip(stage_names,planned) if task != 'refine'}
    reference = None
    if any(t in tasks for t in ('harmonize','stem')):
        if args.style_reference:
            ref = read_midi_notes(args.style_reference, infer.tokenizer,include_drums=True)
            reference = infer.encoders.from_notes(ref, infer.tokenizer,mode=args.reference_mode,stage_instruments=stage_instruments,aggregation=args.reference_aggregation,rng=random.Random(INFERENCE_SEED))
            reference['id'] = str(args.style_reference)
        else:
            reference = choose_reference(infer.encoders, args.genre, random.Random(INFERENCE_SEED),
                reference_id=args.reference_id, bucket=args.reference_bucket,
                mode=args.reference_mode,stage_instruments=stage_instruments,aggregation=args.reference_aggregation)
    if reference:
        for stage,condition in reference['stage_conditions'].items():
            if 'drum' in condition['output_instruments'] and not condition['drum_palette']:
                raise ValueError(f'Reference for {stage} has no standard drum-kit sounds; choose a reference containing drums')
    outputs = inference_output_paths(args.output)
    stage_names = stage_output_names(tasks, stem_targets)
    outputs.update({name: outputs['directory'] / f'{name}.mid' for name in stage_names})
    written_names = {'final', 'report', 'melody', *stage_names}
    if args.input.resolve() in {p.resolve() for name, p in outputs.items() if name in written_names}:
        parser.error('Output or an intermediate file would overwrite the input MIDI')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outputs['directory'].mkdir(parents=True, exist_ok=True)
    mono = melody_voice(skyline(pitched, infer.meta['skyline_tolerance_ms'])[0])
    write_midi_notes(mono, outputs['melody'], infer.tokenizer)
    current, chunks, task_order, extra_report = pipeline_runner(
        infer, original, tasks, args, requested, original_instruments,
        reference, outputs['directory'], stem_targets)
    write_midi_notes(current, outputs['final'], infer.tokenizer)
    report = dict(model=str(args.model), checkpoint_step=infer.checkpoint_step, protocol_version=infer.meta['version'],
                  input=str(args.input), output=str(outputs['final']), genre=args.genre,
                  nonkit_input_drum_notes_filtered=0,percussion_mode='reference_kit',reference_mode=args.reference_mode,reference_aggregation=args.reference_aggregation,
                  input_drum_articulations_normalized=0,
                  requested_instruments=requested, instruments=sorted({n.instrument for n in current}),
                  missing_instruments=sorted(set(requested)-{n.instrument for n in current}),
                  task_order=task_order, stage_order=stage_names, output_notes=len(current),
                  style_reference=reference_report(reference) if reference else None, seed=INFERENCE_SEED, top_k=TOP_K, device=str(infer.device), chunks=chunks)
    report['stem_summary'] = []
    for target in sorted({log['target_instrument'] for log in chunks if log['task'] == 'stem'}):
        added = sum(log['added_stem_notes'] for log in chunks
                    if log['task'] == 'stem' and log['target_instrument'] == target)
        report['stem_summary'].append(dict(instrument=target, generated_notes=added,
                                           status='generated' if added else 'not_generated'))
    report['files'] = {name: str(path) for name, path in outputs.items() if name != 'directory'
                       and name in written_names}
    report.update(extra_report)
    outputs['report'].write_text(json.dumps(report, indent=2))
    tqdm.write(f'Saved {outputs["final"]}', file=sys.stderr)
