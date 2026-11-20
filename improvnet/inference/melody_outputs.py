"""Short output paths shared by both inference entrypoints."""
from pathlib import Path
from improvnet.utils.melody import TASK_TOKENS, skyline, melody_voice


def stage_output_names(tasks, stem_targets):
    """Disambiguate repeated stem stages without changing single-stem filenames."""
    if tasks.count('stem') <= 1:
        return list(tasks)
    targets = iter(stem_targets)
    names = {'drum': 'drums', 'Acoustic Bass': 'bass'}
    result = []
    for task in tasks:
        if task == 'stem':
            target = next(targets)
            result.append(names.get(target, 'stem_' + target.lower().replace(' ', '_')))
        else:
            result.append(task)
    return result

def inference_output_paths(output):
    """Use the requested output stem as a run folder containing every artifact."""
    output = Path(output)
    directory = output.with_suffix('')
    if directory == output:
        directory = output.with_name(output.name + '_steps')
    return dict(final=directory / 'final.mid', directory=directory, report=directory / 'report.json',
                **{name: directory / f'{name}.mid' for name in ('melody', *TASK_TOKENS)})


def plan_stages(original, tasks, requested, original_instruments, stem_targets, *, task_mode='pipeline', target_instrument=None, skyline_tolerance_ms=50):
    """One instrument plan shared by reference preparation and streaming workers."""
    current = {n.instrument for n in original}
    targets = iter(stem_targets)
    planned = []
    for task in tasks:
        target = None
        if task == 'stem':
            target = next(targets) if task_mode == 'pipeline' else target_instrument
            if target is None and 'drum' in requested and 'drum' not in current:
                target = 'drum'
            if target is None:
                raise ValueError('Stem addition requires --target-instrument')
            if target in current:
                raise ValueError(f'Requested {target} stem already exists')
            current = current | {target}
            selected = sorted(current)
        else:
            selected = ([i for i in requested if i != 'drum'] if task == 'harmonize' else
                        sorted({n.instrument for n in melody_voice(skyline(
                            [n for n in original if n.instrument!='drum'],skyline_tolerance_ms)[0])}))
            if task == 'harmonize' and task_mode == 'pipeline':
                selected = [i for i in selected if i not in stem_targets] or list(original_instruments)
            current = set(selected)
        planned.append((task,target,selected))
    return planned
