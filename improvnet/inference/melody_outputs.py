"""Short output paths shared by both inference entrypoints."""
from pathlib import Path
from improvnet.utils.melody import TASK_TOKENS


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
