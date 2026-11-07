"""Stream dependent task chunks and continuously batch their token requests.

One worker owns each task's history and grammar; only the coordinator runs the
shared model. Workers never launch independent model copies or GPU forwards.
"""
from collections import Counter, deque
from copy import copy
from dataclasses import dataclass, field
import random
from pathlib import Path
import threading
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from improvnet.utils.melody_stems import kit_only_notes
from improvnet.inference.melody_runtime import INFERENCE_SEED
from improvnet.inference.melody_outputs import stage_output_names
from improvnet.model.melody_train import autocast
from improvnet.utils.melody import write_midi_notes


@dataclass
class ForwardRequest:
    owner: int
    ids: torch.Tensor
    kwargs: dict
    answer: object = field(default=None, repr=False)


@torch.no_grad()
def forward_batch(model, requests):
    """Return independent logits/caches, with no padded keys visible to attention.

    Prefills are right padded (causality isolates real prompt tokens). Cached
    steps are left padded, explicitly masked, and use each row's real RoPE
    offset. Neither padding nor another request's keys enters a returned cache.
    """
    if not requests:
        return []
    if len(requests) == 1:
        req = requests[0]
        return [model(req.ids, **req.kwargs)]
    cached = requests[0].kwargs.get('past_key_values') is not None
    if any((r.kwargs.get('past_key_values') is not None) != cached for r in requests):
        raise ValueError('Prefills and cached steps must be dispatched separately')
    if not cached:
        lengths = [r.ids.size(1) for r in requests]
        width = max(lengths)
        ids = torch.cat([F.pad(r.ids, (0, width - n)) for r, n in zip(requests, lengths)])
        kwargs = dict(use_cache=True)
        for name in ('genre', 'multi_hot', 'strength', 'style_embedding', 'style_mask', 'beat_embedding', 'beat_mask'):
            values = [r.kwargs.get(name) for r in requests]
            if any(v is not None for v in values):
                exemplar = next(v for v in values if v is not None)
                kwargs[name] = torch.cat([v if v is not None else torch.zeros_like(exemplar) for v in values])
        logits, cache = model(ids, **kwargs)
        return [(logits[row:row+1, n-1:n].clone(),
                 [(k[row:row+1, :n+2].clone(), v[row:row+1, :n+2].clone()) for k, v in cache])
                for row, n in enumerate(lengths)]
    if any(r.ids.shape != (1, 1) for r in requests):
        raise ValueError('Cached requests must contain exactly one token')
    caches = [r.kwargs['past_key_values'] for r in requests]
    lengths = [c[0][0].size(1) for c in caches]
    offsets = [r.kwargs['seq_offset'] for r in requests]
    if offsets != lengths:
        raise ValueError('Cache lengths must equal the next absolute token positions')
    width = max(lengths)
    packed = []
    for layer in range(len(caches[0])):
        packed.append(tuple(torch.cat([F.pad(c[layer][kv], (0, 0, 0, 0, width-n, 0))
                                       for c, n in zip(caches, lengths)]) for kv in (0, 1)))
    ids = torch.cat([r.ids for r in requests])
    positions = torch.tensor(lengths, device=ids.device)
    kwargs = dict(use_cache=True, past_key_values=packed)
    if min(lengths) == width:
        kwargs['seq_offset'] = width  # Equal-length rows retain FlashAttention.
    else:
        kwargs.update(seq_offset=positions, cache_lengths=positions)
    logits, cache = model(ids, **kwargs)
    return [(logits[row:row+1].clone(),
             [(k[row:row+1, width-n:].clone(), v[row:row+1, width-n:].clone()) for k, v in cache])
            for row, n in enumerate(lengths)]


class PipelineCoordinator:
    """Cooperative CPU workers hand all GPU forwards to one batching loop."""
    def __init__(self, model, device, count):
        self.model, self.device = model, device
        self.condition = threading.Condition()
        self.states = ['running'] * count
        self.pending = {}
        self.error = None
        self.batch_sizes = Counter()
        self.prefill_batch_sizes = Counter()
        self.first_forward = {}
        self.finished_at = {}

    def check(self):
        if self.error is not None:
            raise RuntimeError('Parallel inference stopped after a task failed') from self.error

    def fail(self, error):
        with self.condition:
            if self.error is None:
                self.error = error
            self.condition.notify_all()

    def proxy(self, owner):
        def request(ids, **kwargs):
            item = ForwardRequest(owner, ids, kwargs)
            with self.condition:
                self.check()
                self.pending[owner] = item
                self.states[owner] = 'model'
                self.condition.notify_all()
                while item.answer is None:
                    self.check()
                    self.condition.wait()
                return item.answer
        return request

    def finish(self, owner):
        with self.condition:
            self.states[owner] = 'done'
            self.finished_at[owner] = time.perf_counter()
            self.condition.notify_all()

    @torch.no_grad()
    def run(self):
        with autocast(self.device), tqdm(desc='Batched pipeline', unit='token request', dynamic_ncols=True,
                  position=len(self.states)*2) as progress:
            while True:
                with self.condition:
                    while True:
                        self.check()
                        if all(s == 'done' for s in self.states):
                            return
                        # Wait only for CPU grammar work, never for unavailable
                        # upstream music. This avoids fixed batching timeouts.
                        if self.pending and all(s != 'running' for s in self.states):
                            requests = list(self.pending.values())
                            self.pending.clear()
                            break
                        self.condition.wait()
                for cached in (True, False):
                    batch = [r for r in requests if (r.kwargs.get('past_key_values') is not None) == cached]
                    if not batch:
                        continue
                    answers = forward_batch(self.model, batch)
                    now = time.perf_counter()
                    for r in batch:
                        self.first_forward.setdefault(r.owner, now)
                    (self.batch_sizes if cached else self.prefill_batch_sizes)[len(batch)] += 1
                    with self.condition:
                        for r, answer in zip(batch, answers):
                            r.answer = answer
                            self.states[r.owner] = 'running'
                        self.condition.notify_all()
                    progress.update(len(batch))
                    progress.set_postfix(batch=len(batch), max_batch=max(self.batch_sizes, default=1), refresh=False)


class NoteStream:
    """A stream contains complete, ordered onset groups from its upstream task."""
    def __init__(self, coordinator, owner):
        self.coordinator, self.owner = coordinator, owner
        self.queue = deque()
        self.done = False

    def put(self, notes):
        with self.coordinator.condition:
            self.queue.append(notes)
            self.coordinator.condition.notify_all()

    def read(self):
        c = self.coordinator
        with c.condition:
            while not self.queue:
                c.check()
                c.states[self.owner] = 'source'
                c.condition.notify_all()
                c.condition.wait()
            c.check()
            c.states[self.owner] = 'running'
            notes = self.queue.popleft()
            if notes is None:
                self.done = True
            return notes


def run_pipeline(infer, original, tasks, args, requested, original_instruments,
                 reference, output_directory, stem_targets):
    original = kit_only_notes(original)
    if infer.meta.get('version', 0) != 11:
        raise ValueError('Inference requires a dual style/beat checkpoint (protocol 11)')
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    missing_tasks = set(tasks) - set(infer.meta.get('tasks', ()))
    if missing_tasks:
        raise ValueError(f'Checkpoint lacks tasks: {sorted(missing_tasks)}')
    coordinator = PipelineCoordinator(infer.model, infer.device, len(tasks))
    streams = [NoteStream(coordinator, i) for i in range(len(tasks))]
    results, chunk_logs = [None]*len(tasks), [None]*len(tasks)
    # Each task owns a fixed random stream, so thread scheduling cannot change
    # its corruptions or multinomial draws. Sequential CLI RNG stays unchanged.
    workers = []
    targets = iter(stem_targets)
    planned = []
    stage_names = stage_output_names(tasks, stem_targets)
    current_instruments = {n.instrument for n in original}
    for task in tasks:
        target = None
        if task == 'stem':
            target = next(targets) if args.task == 'pipeline' else args.target_instrument
            if target is None and 'drum' in requested and 'drum' not in current_instruments:
                target = 'drum'
            if target is None:
                raise ValueError('Stem addition requires --target-instrument')
            if target in current_instruments:
                raise ValueError(f'Requested {target} stem already exists')
            selected = sorted(current_instruments | {target})
            current_instruments.add(target)
        elif task in ('refine', 'harmonize'):
            use_requested = args.task != 'pipeline' or task == 'harmonize'
            selected = [i for i in requested if i != 'drum'] if use_requested else original_instruments
            if task == 'harmonize' and args.task == 'pipeline':
                selected = [i for i in selected if i not in stem_targets] or original_instruments
            current_instruments = set(selected)
        else:
            selected = requested
            current_instruments = set(requested)
        planned.append((task, target, selected))

    def worker(index):
        task, target, selected = planned[index]
        try:
            local = copy(infer)
            local.model = coordinator.proxy(index)
            local.model_forward_is_proxy = True
            local.progress_position = index*2
            local.sampling_generator = torch.Generator(device=infer.device).manual_seed(INFERENCE_SEED+index*1009)
            stream = streams[index]
            publish = streams[index+1].put if index+1 < len(tasks) else None
            first = stream.read()
            if first is None:
                raise ValueError(f'No source notes reached the {task} task')
            strength = args.melody_strength if task == 'refine' else args.rhythm_strength if task == 'harmonize' else 1.
            notes, logs = local.transform(first, task, args.genre, strength,
                random.Random(INFERENCE_SEED+index*1009), args.temperature,
                instruments=selected, style_reference=reference,
                monophonic=True if task == 'harmonize' and 'refine' in tasks[:index] else None,
                target_instrument=target, source_stream=stream, on_chunk=publish)
            results[index], chunk_logs[index] = notes, logs
            write_midi_notes(notes, output_directory / f'{stage_names[index]}.mid', infer.tokenizer)
            if publish:
                publish(None)
            coordinator.finish(index)
        except BaseException as error:
            failure = RuntimeError(f'{task} task failed: {error}')
            failure.__cause__ = error
            coordinator.fail(failure)

    streams[0].put(original)
    streams[0].put(None)
    started = time.perf_counter()
    for i in range(len(tasks)):
        thread = threading.Thread(target=worker, args=(i,), name=f'melody-{i}-{tasks[i]}', daemon=True)
        workers.append(thread)
        thread.start()
    try:
        coordinator.run()
    except BaseException as error:
        coordinator.fail(error)
        raise
    finally:
        for thread in workers:
            thread.join()
    coordinator.check()
    elapsed = time.perf_counter()-started
    histogram = coordinator.batch_sizes
    forwards = sum(histogram.values())
    requests = sum(size*count for size, count in histogram.items())
    stats = dict(mode='continuous_task_batching', shared_model=True,
                 elapsed_seconds=elapsed, max_batch_size=max(histogram, default=1),
                 mean_decode_batch_size=requests/max(1, forwards),
                 decode_forward_calls=forwards, decode_token_requests=requests,
                 decode_batch_sizes=dict(histogram), prefill_batch_sizes=dict(coordinator.prefill_batch_sizes),
                 per_task_seed=[INFERENCE_SEED+i*1009 for i in range(len(tasks))],
                 stage_order=stage_names,
                 stage_timing=[dict(task=task, stage=stage_names[i], target_instrument=planned[i][1],
                                    first_forward_seconds=coordinator.first_forward.get(i, started)-started,
                                    finished_seconds=coordinator.finished_at[i]-started)
                               for i, task in enumerate(tasks)])
    tqdm.write(f'Pipeline finished in {elapsed:.1f}s; mean decode batch {stats["mean_decode_batch_size"]:.2f}, '
               f'maximum {stats["max_batch_size"]}.')
    return results[-1], [log for logs in chunk_logs for log in logs], list(tasks), dict(parallel_inference=stats)

