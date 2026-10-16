"""Calibrate held-note decoding on validation ONLY, with release/reattack guards."""
import argparse
import csv
from datetime import datetime, timezone
from itertools import product
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace

import mido
import numpy as np
import torch

from improvnet.inference.local_vae_evaluate import candidates
from improvnet.inference.local_vae_inference import (
    Scores, common_args, inputs, load_model, output_directory, resolve_thresholds, write_report,
)
from improvnet.utils.local_vae_data import LocalVAEDataset, seed_for
from improvnet.utils.latent_flow import checkpoint_digest
from improvnet.utils.note_decoding import NoteDecoding, confidence_arrays, decode_held_notes


@torch.inference_mode()
def collect_confidence(model, tokenizer, config, *, split, count, bins_per_file, seed, batch_size,
                       activity_threshold, onset_threshold, manifest, cache_dir=None):
    """Cache complete contiguous windows before thresholding; no reference-conditioned inference."""
    dataset = LocalVAEDataset(config, split)
    device = next(model.parameters()).device
    windows, failures = [], {}
    try:
        for attempt, (row_id, collection, path) in enumerate(candidates(dataset, seed)):
            if len(windows) >= count or attempt >= count * 5:
                break
            try:
                piece = tokenizer.encode(path, recover=True, seed=seed_for(config.SEED, 0, row_id))
                if not piece.metadata['target_present']:
                    raise ValueError('No valid target notes')
            except (ValueError, OSError, EOFError, IndexError, KeyError, mido.KeySignatureError) as error:
                key = str(error)[:100]; failures[key] = failures.get(key, 0) + 1
                continue
            length = min(bins_per_file, len(piece.velocity))
            start = random.Random(seed_for(seed, row_id, 'window')).randrange(len(piece.velocity) - length + 1)
            ids = np.arange(start, start + length)
            reference = SimpleNamespace(config=piece.config, metadata={'pedal_enabled': piece.metadata['pedal_enabled']},
                **{k: getattr(piece, k)[ids].copy() for k in ('velocity', 'onset', 'pedal', 'valid_mask')})
            confidence = {k: [] for k in ('activity', 'onset', 'velocity')}
            arrays = {k: [] for k in ('velocity', 'onset', 'pedal')}
            for offset in range(0, length, batch_size):
                sl = np.arange(offset, min(offset + batch_size, length))
                roll, pedal, mask, enabled = inputs(reference, sl, device)
                posterior = model.encode(roll, pedal, mask)
                rebuilt = model.decode(posterior.mode())
                for key, value in confidence_arrays(rebuilt).items():
                    confidence[key].append(value)
                for key, value in rebuilt.tensors(mask, enabled, activity_threshold=activity_threshold,
                                                  onset_threshold=onset_threshold).items():
                    arrays[key].append(value.cpu().numpy())
            confidence = {k: np.concatenate(v) for k, v in confidence.items()}
            legacy = SimpleNamespace(config=reference.config, valid_mask=reference.valid_mask,
                **{k: np.concatenate(v) for k, v in arrays.items()})
            item = dict(reference=reference, confidence=confidence, legacy=legacy, row_id=row_id,
                collection=collection, path=str(path), start_bin=start, bins=length,
                target_kind=piece.metadata['target_kind'])
            windows.append(item)
            manifest.writerow([split, row_id, collection, start, length])
            if cache_dir is not None:
                np.savez_compressed(cache_dir / f'{row_id}.npz',
                    pedal_enabled=np.array(piece.metadata['pedal_enabled']),
                    target_kind=np.array(piece.metadata['target_kind']),
                    **{f'confidence_{k}': v for k, v in confidence.items()},
                    **{f'reference_{k}': getattr(reference, k) for k in ('velocity', 'onset', 'pedal', 'valid_mask')},
                    **{f'legacy_{k}': getattr(legacy, k) for k in ('velocity', 'onset', 'pedal')})
            if len(windows) % 20 == 0 or len(windows) == count:
                print(f'Cached {split}: {len(windows)}/{count}', flush=True)
    finally:
        if dataset.connection is not None:
            dataset.connection.close()
    if not windows:
        raise RuntimeError(f'No usable {split} windows; failures={failures}')
    return windows, failures


def load_cached_confidence(directory, tokenizer, config, digest, manifest):
    """Reuse verified validation predictions when expanding the decoding grid."""
    report = json.loads((directory / 'report.json').read_text())
    if report.get('split') != 'validation' or report.get('checkpoint_sha256') != digest:
        raise ValueError('Confidence cache must be validation data from this exact checkpoint')
    dataset = LocalVAEDataset(config, 'validation')
    windows = []
    try:
        with (directory / 'selection.csv').open() as handle:
            for row in csv.DictReader(handle):
                row_id, start, bins = (int(row[k]) for k in ('row_id', 'start_bin', 'bins'))
                collection = row['collection']
                if row['split'] != 'validation' or row_id not in dataset.groups.get(collection, ()):
                    raise ValueError('Cached row is not in the validation split')
                path = dataset.connect().execute('SELECT path FROM files WHERE id=?', (row_id,)).fetchone()[0]
                with np.load(directory / 'confidence' / f'{row_id}.npz', allow_pickle=False) as data:
                    arrays = {k: data['reference_' + k].copy() for k in ('velocity', 'onset', 'pedal', 'valid_mask')}
                    if 'pedal_enabled' in data:
                        enabled, kind = bool(data['pedal_enabled']), str(data['target_kind'])
                    else:
                        # Older caches omitted controller metadata. Recover it
                        # from the identical selected source, without model work.
                        piece = tokenizer.encode(path, recover=True, seed=seed_for(config.SEED, 0, row_id))
                        if any(not np.array_equal(getattr(piece, k)[start:start + bins], arrays[k]) for k in arrays):
                            raise ValueError('Cached reference no longer matches its source/window')
                        enabled, kind = piece.metadata['pedal_enabled'], piece.metadata['target_kind']
                    reference = SimpleNamespace(config=tokenizer.config, metadata={'pedal_enabled': enabled}, **arrays)
                    legacy = SimpleNamespace(config=reference.config, valid_mask=reference.valid_mask,
                        **{k: data['legacy_' + k].copy() for k in ('velocity', 'onset', 'pedal')})
                    windows.append(dict(reference=reference, legacy=legacy,
                        confidence={k: data['confidence_' + k].copy() for k in ('activity', 'onset', 'velocity')},
                        row_id=row_id, collection=collection, path=path, start_bin=start, bins=bins, target_kind=kind))
                    manifest.writerow(['validation', row_id, collection, start, bins])
    finally:
        if dataset.connection is not None:
            dataset.connection.close()
    if len(windows) != report['files']:
        raise ValueError('Incomplete validation confidence cache')
    return windows, report['failures']


def prediction(window, activity_threshold, onset_threshold, options):
    reference = window['reference']
    if options.mode == 'legacy':
        return window['legacy']
    arrays = decode_held_notes(*(window['confidence'][k] for k in ('activity', 'onset', 'velocity')),
        reference.valid_mask, activity_threshold=activity_threshold, onset_threshold=onset_threshold,
        continuation_threshold=options.continuation_threshold, max_gap_ms=options.max_gap_ms,
        fine_ms=reference.config.fine_ms)
    return SimpleNamespace(**arrays, pedal=window['legacy'].pedal,
                           valid_mask=reference.valid_mask, config=reference.config)


def evaluate_windows(windows, activity_threshold, onset_threshold, options, *, frames=False):
    total, collections = Scores(), {}
    for window in windows:
        result = prediction(window, activity_threshold, onset_threshold, options)
        score = Scores(); score.notes(window['reference'], result)
        if frames:
            score.arrays(window['reference'], result)
        total.add(score)
        group = collections.setdefault(window['collection'], dict(files=0, scores=Scores()))
        group['files'] += 1; group['scores'].add(score)
    return total, {k: dict(files=v['files'], **v['scores'].report()) for k, v in collections.items()}


def ratio(score, name, metric='f1'):
    c = score.counts
    tp, fp, fn = (c[name + '_' + k] for k in ('tp', 'fp', 'fn'))
    numerator, denominator = ((2 * tp, 2 * tp + fp + fn) if metric == 'f1' else
        ((tp, tp + fn) if metric == 'recall' else (tp, tp + fp)))
    return numerator / denominator if denominator else 0.


def release_guard(score, baseline):
    """Keep recall, genuine fast reattacks and release fidelity while reducing extras."""
    c, b = score.counts, baseline.counts
    mae = c['offset_ms'] / max(1, c['notes_tp'])
    old_mae = b['offset_ms'] / max(1, b['notes_tp'])
    rapid = c['rapid_matched'] / max(1, c['rapid_notes'])
    old_rapid = b['rapid_matched'] / max(1, b['rapid_notes'])
    within = c['offset_within_20ms'] / max(1, c['notes_tp'])
    old_within = b['offset_within_20ms'] / max(1, b['notes_tp'])
    return (ratio(score, 'notes', 'recall') >= .99 and mae <= old_mae + 2
            and ratio(score, 'notes_with_offsets') >= ratio(baseline, 'notes_with_offsets')
            and (not c['rapid_notes'] or rapid >= old_rapid - .001)
            and within >= old_within - .005)


def save_calibration(checkpoint, calibration, report_path, expected_digest):
    """Atomically update decoding metadata; retain the exact old checkpoint."""
    checkpoint = Path(checkpoint)
    if checkpoint_digest(checkpoint) != expected_digest:
        raise RuntimeError('Checkpoint changed before calibration could be saved')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state.get('format') != 'local_vae_training_v1':
        raise ValueError('Expected a locally trained VAE checkpoint')
    options = NoteDecoding(**calibration['note_decoding'])
    if options.mode != 'held' or options.continuation_threshold > calibration['activity_threshold']:
        raise ValueError('Invalid held-note calibration')
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = checkpoint.with_name(f'{checkpoint.stem}.before_held_note_decoding_{timestamp}{checkpoint.suffix}')
    old = state.get('decoder_calibration') or {}
    state['decoder_calibration'] = dict(calibration, method='held_note_validation_v1',
        calibration_report=str(Path(report_path).resolve()),
        legacy_thresholds=dict(activity=old.get('activity_threshold', .5), onset=old.get('onset_threshold', .99)))
    with tempfile.NamedTemporaryFile(dir=checkpoint.parent, prefix='.held-note-', suffix='.pt', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(state, temporary)
        reloaded = torch.load(temporary, map_location='cpu', weights_only=False)
        if (reloaded['step'] != state['step'] or reloaded['model_state_dict'].keys() != state['model_state_dict'].keys()
                or any(not torch.equal(value, reloaded['model_state_dict'][key])
                       for key, value in state['model_state_dict'].items())):
            raise RuntimeError('Checkpoint weight/step verification failed')
        del reloaded
        if checkpoint_digest(checkpoint) != expected_digest:
            raise RuntimeError('Checkpoint changed while saving calibration')
        os.chmod(temporary, checkpoint.stat().st_mode & 0o777)
        os.link(checkpoint, backup)
        os.replace(temporary, checkpoint)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(backup=str(backup), previous_sha256=expected_digest,
                checkpoint_sha256=checkpoint_digest(checkpoint), model_weights_unchanged=True, training_step=state['step'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    common_args(parser)
    parser.add_argument('--num-files', type=int, default=200)
    parser.add_argument('--bins-per-file', type=int, default=8)
    parser.add_argument('--activity-grid', type=float, nargs='+', default=[.90, .95])
    parser.add_argument('--onset-grid', type=float, nargs='+', default=[.95, .97, .98, .985, .99, .995])
    parser.add_argument('--continuation-grid', type=float, nargs='+', default=[.5, .7, .8, .9])
    parser.add_argument('--gap-grid-ms', type=int, nargs='+', default=[0, 10, 20])
    parser.add_argument('--cache-confidence', action='store_true', help='Save compact windows for diagnosis/reproducibility')
    parser.add_argument('--cached-validation', type=Path, help='Reuse a validation confidence cache from this exact checkpoint')
    parser.add_argument('--save-calibration', action='store_true', help='Save selected metadata atomically, preserving an original checkpoint backup')
    parser.add_argument('--gpu-memory-fraction', type=float, default=.12)
    args = parser.parse_args(argv)
    if args.num_files < 1 or args.bins_per_file < 1 or not 0 < args.gpu_memory_fraction <= 1:
        parser.error('Invalid file/bin counts or GPU memory fraction')
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,
            device.index if device.index is not None else torch.cuda.current_device())
    digest = checkpoint_digest(args.checkpoint)
    model, tokenizer, step, saved = load_model(args.checkpoint, device)
    resolve_thresholds(args, model)
    grid = sorted(set(product(args.activity_grid, args.onset_grid, args.continuation_grid, args.gap_grid_ms)))
    for activity, onset, continuation, gap in grid:
        NoteDecoding('held', continuation, gap)
        if not 0 < activity < 1 or not 0 < onset < 1 or continuation > activity:
            parser.error('Invalid calibration grid')
    directory = output_directory(args.output_dir, 'held-note-calibration-')
    cache = directory / 'confidence' if args.cache_confidence else None
    if cache is not None:
        cache.mkdir()
    with (directory / 'selection.csv').open('x', newline='') as handle:
        manifest = csv.writer(handle); manifest.writerow(['split', 'row_id', 'collection', 'start_bin', 'bins'])
        if args.cached_validation:
            cached_report = json.loads((args.cached_validation / 'report.json').read_text())
            if (cached_report['seed'] != args.seed or cached_report['legacy_thresholds'] !=
                    dict(activity=args.activity_threshold, onset=args.onset_threshold)):
                raise ValueError('Cached seed/legacy thresholds differ from this calibration')
            windows, failures = load_cached_confidence(args.cached_validation, tokenizer, SimpleNamespace(**saved), digest, manifest)
        else:
            windows, failures = collect_confidence(model, tokenizer, SimpleNamespace(**saved), split='validation',
                count=args.num_files, bins_per_file=args.bins_per_file, seed=args.seed, batch_size=args.batch_size,
                activity_threshold=args.activity_threshold, onset_threshold=args.onset_threshold, manifest=manifest, cache_dir=cache)
    baseline, baseline_groups = evaluate_windows(windows, args.activity_threshold, args.onset_threshold,
                                                NoteDecoding(), frames=True)
    candidates_scored, eligible = [], []
    for index, (activity, onset, continuation, gap) in enumerate(grid):
        options = NoteDecoding('held', continuation, gap)
        scores, _ = evaluate_windows(windows, activity, onset, options)
        acceptable = release_guard(scores, baseline)
        row = dict(activity_threshold=activity, onset_threshold=onset, note_decoding=vars(options),
                   passes_release_guard=acceptable, metrics=scores.report())
        candidates_scored.append(row)
        if acceptable:
            key = (ratio(scores, 'notes'), ratio(scores, 'notes_with_offsets'),
                   -scores.counts['offset_ms'] / max(1, scores.counts['notes_tp']), -gap)
            eligible.append((key, row))
        if (index + 1) % 20 == 0 or index + 1 == len(grid):
            print(f'Calibrated {index+1}/{len(grid)} candidates; eligible={len(eligible)}', flush=True)
    selected = None
    for _, row in sorted(eligible, key=lambda item: item[0], reverse=True):
        scores, groups = evaluate_windows(windows, row['activity_threshold'], row['onset_threshold'],
                                         NoteDecoding(**row['note_decoding']), frames=True)
        if ratio(scores, 'activity') >= .99 and ratio(scores, 'onset') >= .99:
            selected = dict(**{k: v for k, v in row.items() if k != 'metrics'}, metrics=scores.report(), collections=groups)
            break
    if checkpoint_digest(args.checkpoint) != digest:
        raise RuntimeError('Checkpoint changed during calibration')
    report = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest, step=step, seed=args.seed,
        confidence_cache=str(args.cached_validation.resolve()) if args.cached_validation else None,
        split='validation', files=len(windows), bins=sum(w['bins'] for w in windows),
        legacy_thresholds=dict(activity=args.activity_threshold, onset=args.onset_threshold),
        legacy_metrics=baseline.report(), legacy_collections=baseline_groups, selected=selected,
        guards=dict(minimum_note_recall=.99, maximum_release_mae_increase_ms=2,
            minimum_offset_f1='legacy baseline', maximum_rapid_recall_drop=.001,
            maximum_20ms_release_fraction_drop=.005, minimum_activity_and_onset_f1=.99),
        note_matching='Same-pitch chronological one-to-one; 50-ms onset tolerance. Offsets: max(50 ms, 20% reference duration). Rapid: successive same-pitch attacks <=200 ms.',
        candidates=candidates_scored, failures=failures)
    write_report(directory / 'report.json', report)
    if selected is not None:
        calibration = {k: selected[k] for k in ('activity_threshold', 'onset_threshold', 'note_decoding')}
        write_report(directory / 'decoder_calibration.json', calibration)
        if args.save_calibration:
            update = save_calibration(args.checkpoint, calibration, directory / 'report.json', digest)
            write_report(directory / 'checkpoint_update.json', update)
    elif args.save_calibration:
        raise RuntimeError('No candidate passed validation safeguards; checkpoint was not changed')
    print(json.dumps(dict(selected=selected, legacy_metrics=baseline.report())), flush=True)
    print(f'Saved: {directory}', flush=True)
    return report


if __name__ == '__main__':
    main()
