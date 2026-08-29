import random
import copy
import os
import json
import math
import torch
from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import improvnet.model.ar_config as ar_config
from improvnet.tokenizer.midi import MidiDict
from improvnet.tokenizer.absolute import AbsTokenizer
from improvnet.model.ar_config import GENRES

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

def read_jsonl_files(data_dirs, split="train"):
    files = []
    for file in data_dirs:
        if os.path.exists(file):
            with open(file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data.get("split", "train") == split:
                        files.append(data)
        else:
            print(f"Warning: {file} does not exist. Skipping.")
    return files

class ProcessData:
    def __init__(self):
        self.tokenizer = AbsTokenizer()
        self.genres = GENRES
        
        self.INSTRUMENT_CLASSES = [
            "Acoustic Piano", "Electric Piano", "Chromatic Percussion", "Organ", 
            "Acoustic Guitar", "Clean Electric Guitar", "Distorted Electric Guitar", 
            "Acoustic Bass", "Electric Bass", "Violin", "Viola", "Cello", "Contrabass", 
            "Orchestral Harp", "Timpani", "String Ensemble", "Synth Strings", 
            "Choir and Voice", "Orchestra Hit", "Trumpet", "Trombone", "Tuba", 
            "French Horn", "Brass Section", "Soprano/Alto Sax", "Tenor Sax", 
            "Baritone Sax", "Oboe", "English Horn", "Bassoon", "Clarinet", "Piccolo", 
            "Flute", "Pipe", "Synth Lead", "Synth Pad", "Synth Effect", "Ethnic", 
            "Percussive", "Sound Effects", "drum"
        ]

    def get_genre_id(self, genre_str: str) -> int:
        if not genre_str:
            return self.genres.index("unknown")
        g = str(genre_str).lower().strip()
        if g in self.genres:
            return self.genres.index(g)
        return self.genres.index("unknown")
    
    def read_midi(self, file_path: str) -> MidiDict:
        return MidiDict.from_midi(file_path)
    
    def save_midi(self, midi_dict: MidiDict, file_path: str):
        midi_dict.save(file_path)

    def midi_to_tokens(self, midi_dict: MidiDict) -> list:
        return self.tokenizer.tokenize(midi_dict)
    
    def tokens_to_midi(self, tokens: list) -> MidiDict:
        return self.tokenizer.detokenize(tokens).to_midi()
    
    def tokens_to_tensor(self, tokens: list) -> torch.Tensor:
        ids = []
        for tok in tokens:
            if tok in self.tokenizer.tok_to_id:
                ids.append(self.tokenizer.tok_to_id[tok])
            else:
                raise KeyError(f"Token {tok} not found in vocab")
        return torch.tensor(ids, dtype=torch.long)

    def format_variable_sequence(self, tokens: list, target_length: int, pad_id: int = 2) -> torch.Tensor:
        if not tokens:
            return torch.full((target_length,), pad_id, dtype=torch.long)

        tensor_seq = self.tokens_to_tensor(tokens)
        valid_len = min(tensor_seq.shape[0], target_length)
        
        final_tensor = torch.full((target_length,), pad_id, dtype=torch.long)
        if valid_len > 0:
            final_tensor[:valid_len] = tensor_seq[:valid_len]
            
        return final_tensor

    def pitch_augmentation(self, tokens: list) -> list:
        semitone_shift = random.randint(-7, 7)
        augmented_tokens = copy.deepcopy(tokens)
        for i, event in enumerate(augmented_tokens):
            if isinstance(event, tuple) and len(event) in (2, 3) and isinstance(event[1], int):
                if event[0] in ('onset', 'dur', 'prefix'): continue
                if "Drum" in str(event[0]) or "Percuss" in str(event[0]) or event[0] == 'drum': continue
                    
                new_pitch = max(0, min(127, event[1] + semitone_shift))
                if len(event) == 3:
                    augmented_tokens[i] = (event[0], new_pitch, event[2])
                else:
                    augmented_tokens[i] = (event[0], new_pitch)
        return augmented_tokens

    def get_instrument_multihot(self, tokens: list) -> torch.Tensor:
        active_instruments = set()
        for event in tokens:
            if isinstance(event, tuple) and len(event) in (2, 3):
                inst_name = event[0]
                if inst_name in self.INSTRUMENT_CLASSES:
                    active_instruments.add(inst_name)
        
        multi_hot = torch.zeros(len(self.INSTRUMENT_CLASSES), dtype=torch.float32)
        for i, cls_name in enumerate(self.INSTRUMENT_CLASSES):
            if cls_name in active_instruments:
                multi_hot[i] = 1.0
        return multi_hot

def _cfg(name, default=None, config=None):
    return getattr(config or ar_config, name, default)

def distributed_rank_world():
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1

def _epoch_indices(loader: DataLoader, epoch: int) -> list[int]:
    sampler = loader.sampler
    dataset_len = len(loader.dataset)
    shuffle = getattr(sampler, 'shuffle', False)
    seed = getattr(sampler, 'seed', 0)

    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed + epoch)
        return torch.randperm(dataset_len, generator=generator).tolist()
    return list(range(dataset_len))

def _loader_from_indices(loader: DataLoader, indices: list[int]) -> DataLoader:
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=indices,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        collate_fn=loader.collate_fn
    )

def resume_epoch_loader(
    loader: DataLoader,
    start_sample_idx: int = 0,
    epoch: int | None = None,
    start_global_sample_idx: int | None = None
) -> DataLoader:
    if start_global_sample_idx is not None:
        if start_global_sample_idx <= 0:
            return loader
        rank, world_size = distributed_rank_world()
        remaining_indices = _epoch_indices(loader, epoch or 0)[start_global_sample_idx:]
        return _loader_from_indices(loader, remaining_indices[rank::world_size])

    if start_sample_idx <= 0:
        return loader
    if loader.sampler is None:
        return loader

    remaining_indices = list(loader.sampler)[start_sample_idx:]
    return _loader_from_indices(loader, remaining_indices)

_BNB_STATE_KEYS = {
    "state1", "state2", "qmap1", "qmap2", "absmax1", "absmax2",
    "max1", "max2", "new_max1", "new_max2", "unorm_vec",
    "__bnb_optimizer_quant_state__"
}
_TORCH_ADAMW_STATE_KEYS = {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}

def checkpoint_path(config=None):
    return os.path.join(_cfg("SAVE_DIR", config=config), "latest_checkpoint.pt")

def load_training_checkpoint(config=None):
    path = checkpoint_path(config=config)
    # path = os.path.join(_cfg("SAVE_DIR", config=config), "best_model.pt")
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')

def _move_state_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _move_state_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_state_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_state_to_cpu(v) for v in value)
    return value

def _normalize_optimizer_backend(backend):
    if backend is None:
        return "auto"
    backend = str(backend).lower().replace("-", "_")
    aliases = {
        "torch": "adamw",
        "torch_adamw": "adamw",
        "adamw32": "adamw",
        "adamw_32bit": "adamw",
        "bnb": "paged_adamw8bit",
        "bitsandbytes": "paged_adamw8bit",
        "8bit": "paged_adamw8bit",
        "adamw_8bit": "adamw8bit",
        "adamw8": "adamw8bit",
        "paged_adamw_8bit": "paged_adamw8bit",
        "paged_8bit": "paged_adamw8bit",
    }
    return aliases.get(backend, backend)

def _optimizer_state_format(optimizer_state_dict):
    if not optimizer_state_dict:
        return None

    state = optimizer_state_dict.get("state", {})
    saw_state = False
    for param_state in state.values():
        if not isinstance(param_state, dict) or len(param_state) == 0:
            continue
        saw_state = True
        keys = set(param_state.keys())
        wrapped_quant_state = param_state.get("__bnb_optimizer_quant_state__")
        if isinstance(wrapped_quant_state, dict):
            keys.update(wrapped_quant_state.keys())

        if keys & _BNB_STATE_KEYS:
            return "bitsandbytes"
        if keys & _TORCH_ADAMW_STATE_KEYS:
            return "torch_adamw"

    return "empty" if not saw_state else "unknown"

def _default_8bit_backend(config=None):
    if bnb is None:
        return None
    if _cfg("PREFER_PAGED_8BIT_OPTIMIZER", True, config=config) and hasattr(bnb.optim, "PagedAdamW8bit"):
        return "paged_adamw8bit"
    if hasattr(bnb.optim, "AdamW8bit"):
        return "adamw8bit"
    return None

def optimizer_backend_from_instance(optimizer):
    module_name = optimizer.__class__.__module__.lower()
    class_name = optimizer.__class__.__name__.lower()
    if module_name.startswith("bitsandbytes"):
        if "paged" in class_name:
            return "paged_adamw8bit"
        if "8bit" in class_name:
            return "adamw8bit"
    return "adamw"

def _checkpoint_optimizer_backend(checkpoint, config=None):
    if checkpoint is None:
        return None

    backend = checkpoint.get("optimizer_backend")
    if backend is not None:
        return _normalize_optimizer_backend(backend)

    optimizer_class = str(checkpoint.get("optimizer_class", "")).lower()
    if "pagedadamw8bit" in optimizer_class:
        return "paged_adamw8bit"
    if "adamw8bit" in optimizer_class:
        return "adamw8bit"
    if "adamw" in optimizer_class:
        return "adamw"

    state_format = _optimizer_state_format(checkpoint.get("optimizer_state_dict"))
    if state_format == "bitsandbytes":
        return _default_8bit_backend(config=config) or "adamw8bit"
    if state_format == "torch_adamw":
        return "adamw"
    return None

def _is_bitsandbytes_backend(backend):
    return backend in ("adamw8bit", "paged_adamw8bit")

def _is_bitsandbytes_optimizer(optimizer):
    return optimizer.__class__.__module__.lower().startswith("bitsandbytes")

def _choose_optimizer_backend(checkpoint, config=None):
    requested_backend = _normalize_optimizer_backend(_cfg("OPTIMIZER_BACKEND", "auto", config=config))
    valid_backends = {"auto", "adamw", "adamw8bit", "paged_adamw8bit"}
    if requested_backend not in valid_backends:
        raise ValueError(
            f"Unsupported OPTIMIZER_BACKEND={_cfg('OPTIMIZER_BACKEND', 'auto', config=config)!r}. "
            f"Use one of {sorted(valid_backends)}."
        )

    if requested_backend != "auto":
        return requested_backend

    checkpoint_backend = _checkpoint_optimizer_backend(checkpoint, config=config)
    if checkpoint_backend is not None:
        return checkpoint_backend

    return _default_8bit_backend(config=config) or "adamw"

def build_optimizer(parameters, checkpoint=None, is_main_process=True, config=None):
    params = list(parameters)
    backend = _choose_optimizer_backend(checkpoint, config=config)
    requested_backend = _normalize_optimizer_backend(_cfg("OPTIMIZER_BACKEND", "auto", config=config))
    auto_requested = requested_backend == "auto"
    checkpoint_format = _optimizer_state_format(
        checkpoint.get("optimizer_state_dict") if checkpoint is not None else None
    )

    if _is_bitsandbytes_backend(backend) and bnb is None:
        if auto_requested and checkpoint_format != "bitsandbytes":
            backend = "adamw"
        else:
            raise RuntimeError(
                "This checkpoint needs a bitsandbytes optimizer, but bitsandbytes "
                "could not be imported in this environment."
            )

    optimizer_kwargs = dict(lr=_cfg("LR", config=config), weight_decay=1e-2, betas=(0.9, 0.95))
    if backend == "adamw":
        optimizer = AdamW(params, **optimizer_kwargs)
    elif backend == "paged_adamw8bit":
        if not hasattr(bnb.optim, "PagedAdamW8bit"):
            backend = "adamw8bit"
            optimizer = bnb.optim.AdamW8bit(params, **optimizer_kwargs)
        else:
            optimizer = bnb.optim.PagedAdamW8bit(params, **optimizer_kwargs)
    elif backend == "adamw8bit":
        optimizer = bnb.optim.AdamW8bit(params, **optimizer_kwargs)
    else:
        raise ValueError(f"Unsupported optimizer backend: {backend}")

    if is_main_process:
        print(f"Using optimizer backend: {backend} ({optimizer.__class__.__name__}).")
    return optimizer, backend

def _optimizer_state_is_compatible(optimizer, optimizer_state_dict):
    state_format = _optimizer_state_format(optimizer_state_dict)
    if state_format in (None, "empty"):
        return True
    if state_format == "unknown":
        return True
    if _is_bitsandbytes_optimizer(optimizer):
        return state_format == "bitsandbytes"
    return state_format == "torch_adamw"

def _load_optimizer_state_dict(optimizer, optimizer_state_dict):
    if _is_bitsandbytes_optimizer(optimizer):
        try:
            optimizer.load_state_dict(optimizer_state_dict, move_to_device=True)
            return
        except TypeError:
            pass
    optimizer.load_state_dict(optimizer_state_dict)

def _current_rank_rng_state(device):
    return {
        "rank": dist.get_rank() if dist.is_initialized() else 0,
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state(device)
            if device.type == 'cuda' and torch.cuda.is_available()
            else None
        ),
    }

def collect_rng_state_for_checkpoint(device):
    rank_state = _current_rank_rng_state(device)
    if dist.is_initialized():
        rank_states = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(rank_states, rank_state)
    else:
        rank_states = [rank_state]
    return {"rank_states": rank_states}

def _restore_rng_state(rng_state, device):
    if not rng_state:
        return False

    rank_state = rng_state
    if isinstance(rng_state, dict) and "rank_states" in rng_state:
        rank = dist.get_rank() if dist.is_initialized() else 0
        rank_states = rng_state.get("rank_states") or []
        if rank < len(rank_states):
            rank_state = rank_states[rank]
        elif rank_states:
            rank_state = rank_states[0]
        else:
            return False

    if not isinstance(rank_state, dict):
        return False

    python_random_state = rank_state.get("python_random_state")
    torch_rng_state = rank_state.get("torch_rng_state")
    cuda_rng_state = rank_state.get("cuda_rng_state")

    if python_random_state is not None:
        random.setstate(python_random_state)
    if torch_rng_state is not None:
        torch.set_rng_state(torch_rng_state.cpu())
    if (
        cuda_rng_state is not None
        and device.type == 'cuda'
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state(cuda_rng_state.cpu(), device=device)

    return python_random_state is not None or torch_rng_state is not None or cuda_rng_state is not None

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    step,
    best_val_loss,
    epoch,
    next_batch_idx,
    next_sample_idx,
    next_global_sample_idx,
    micro_step,
    scaler=None,
    is_best=False,
    optimizer_backend=None,
    rng_state=None,
    config=None,
):
    os.makedirs(_cfg("SAVE_DIR", config=config), exist_ok=True)
    model_to_save = model.module if isinstance(model, DDP) else model
    optimizer_state_dict = _move_state_to_cpu(optimizer.state_dict())
    checkpoint = {
        'checkpoint_version': 3,
        'step': step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer_state_dict,
        'optimizer_backend': optimizer_backend or optimizer_backend_from_instance(optimizer),
        'optimizer_class': f"{optimizer.__class__.__module__}.{optimizer.__class__.__name__}",
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'epoch': epoch,
        'next_batch_idx': next_batch_idx,
        'next_sample_idx': next_sample_idx,
        'next_global_sample_idx': next_global_sample_idx,
        'batch_size': _cfg("BATCH_SIZE", config=config),
        'accum_steps': _cfg("ACCUM_STEPS", config=config),
        'world_size': dist.get_world_size() if dist.is_initialized() else 1,
        'micro_step': micro_step
    }
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    if rng_state is not None:
        checkpoint['rng_state'] = rng_state
    torch.save(checkpoint, checkpoint_path(config=config))
    if is_best:
        torch.save(checkpoint, os.path.join(_cfg("SAVE_DIR", config=config), "best_model.pt"))
    del checkpoint, optimizer_state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _checkpoint_base_lrs(checkpoint, optimizer=None, config=None):
    scheduler_state = checkpoint.get('scheduler_state_dict', {})
    if 'base_lrs' in scheduler_state:
        return scheduler_state['base_lrs']
    optimizer_state_dict = checkpoint.get('optimizer_state_dict')
    if not optimizer_state_dict:
        if optimizer is not None:
            return [
                group.get('initial_lr', group['lr'])
                for group in optimizer.param_groups
            ]
        return [_cfg("LR", config=config)]
    return [
        group.get('initial_lr', group['lr'])
        for group in optimizer_state_dict['param_groups']
    ]

def _lrs_match_config(checkpoint_lrs, config_lr: float) -> bool:
    return all(math.isclose(lr, config_lr, rel_tol=1e-12, abs_tol=1e-16) for lr in checkpoint_lrs)

def load_checkpoint(model, optimizer, scheduler, device, scaler=None, checkpoint=None, config=None):
    if checkpoint is None:
        checkpoint = load_training_checkpoint(config=config)
    if checkpoint is None:
        return 0, float('inf'), False, False, None, None

    has_resume_position = all(
        key in checkpoint for key in ('epoch', 'next_batch_idx', 'micro_step')
    )
    resume_state = {
        'has_resume_position': has_resume_position,
        'epoch': checkpoint.get('epoch'),
        'next_batch_idx': checkpoint.get('next_batch_idx'),
        'next_sample_idx': checkpoint.get('next_sample_idx'),
        'next_global_sample_idx': checkpoint.get('next_global_sample_idx'),
        'checkpoint_batch_size': checkpoint.get('batch_size', _cfg("RESUME_CHECKPOINT_BATCH_SIZE", config=config)),
        'checkpoint_accum_steps': checkpoint.get('accum_steps', _cfg("RESUME_CHECKPOINT_ACCUM_STEPS", config=config)),
        'checkpoint_world_size': checkpoint.get('world_size', _cfg("RESUME_CHECKPOINT_WORLD_SIZE", config=config)),
        'micro_step': checkpoint.get('micro_step'),
        'optimizer_backend': optimizer_backend_from_instance(optimizer),
        'checkpoint_optimizer_backend': _checkpoint_optimizer_backend(checkpoint, config=config),
        'optimizer_restored': False,
        'rng_restored': False
    }
    model_to_load = model.module if isinstance(model, DDP) else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])

    optimizer_state_dict = checkpoint.get('optimizer_state_dict')
    if optimizer_state_dict is not None:
        if _optimizer_state_is_compatible(optimizer, optimizer_state_dict):
            _load_optimizer_state_dict(optimizer, optimizer_state_dict)
            resume_state['optimizer_restored'] = True
        elif _cfg("ALLOW_OPTIMIZER_MIGRATION_TO_8BIT", False, config=config):
            resume_state['optimizer_skipped_reason'] = (
                "optimizer backend changed; ALLOW_OPTIMIZER_MIGRATION_TO_8BIT=True"
            )
        else:
            checkpoint_format = _optimizer_state_format(optimizer_state_dict)
            current_backend = optimizer_backend_from_instance(optimizer)
            raise RuntimeError(
                "Optimizer checkpoint is not compatible with the current optimizer. "
                f"Checkpoint format={checkpoint_format!r}, current backend={current_backend!r}. "
                "Leave OPTIMIZER_BACKEND='auto' to resume exactly, or set "
                "ALLOW_OPTIMIZER_MIGRATION_TO_8BIT=True if you intentionally want "
                "to skip the old optimizer moments and continue with the new optimizer."
            )
    else:
        resume_state['optimizer_skipped_reason'] = "checkpoint has no optimizer_state_dict"

    checkpoint_lrs = _checkpoint_base_lrs(checkpoint, optimizer, config=config)
    lr_changed = not _lrs_match_config(checkpoint_lrs, _cfg("LR", config=config))
    scheduler_restored = 'scheduler_state_dict' in checkpoint
    resume_start_lrs = [group['lr'] for group in optimizer.param_groups]

    if scheduler_restored and not lr_changed:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        for group, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
            group['lr'] = lr

    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    resume_state['rng_restored'] = _restore_rng_state(checkpoint.get('rng_state'), device)

    step = checkpoint.get('step', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    del checkpoint
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return step, best_val_loss, scheduler_restored, lr_changed, resume_start_lrs, resume_state
