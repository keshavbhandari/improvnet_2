"""Short output paths shared by both inference entrypoints."""
from pathlib import Path
from improvnet.utils.melody import TASK_TOKENS

def inference_output_paths(output):
    """Keep the requested final filename and short per-stage names in its folder."""
    output = Path(output)
    directory = output.with_suffix('')
    if directory == output:
        directory = output.with_name(output.name + '_steps')
    return dict(final=output, directory=directory, report=directory / 'report.json',
                **{name: directory / f'{name}.mid' for name in ('melody', *TASK_TOKENS)})
