import math
import random
import os
import json
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.checkpoint import checkpoint as activation_checkpoint
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from improvnet.model.ar_config import *
from improvnet.model.ar_model import ARContextModel
from improvnet.utils.ar_utils import ProcessData
import contextlib

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _amp_dtype():
    if not torch.cuda.is_available(): return None
    major = torch.cuda.get_device_capability()[0]
    return torch.bfloat16 if major >= 8 else torch.float16

AMP_DTYPE = _amp_dtype()
OPTIMIZER_BACKEND = globals().get("OPTIMIZER_BACKEND", "auto")
ALLOW_OPTIMIZER_MIGRATION_TO_8BIT = globals().get("ALLOW_OPTIMIZER_MIGRATION_TO_8BIT", False)
PREFER_PAGED_8BIT_OPTIMIZER = globals().get("PREFER_PAGED_8BIT_OPTIMIZER", True)
LM_HEAD_CHUNK_SIZE = globals().get("LM_HEAD_CHUNK_SIZE", 512)
RESUME_CHECKPOINT_BATCH_SIZE = globals().get("RESUME_CHECKPOINT_BATCH_SIZE", None)
RESUME_CHECKPOINT_ACCUM_STEPS = globals().get("RESUME_CHECKPOINT_ACCUM_STEPS", None)
RESUME_CHECKPOINT_WORLD_SIZE = globals().get("RESUME_CHECKPOINT_WORLD_SIZE", None)

class ARContextDataset(Dataset):
    def __init__(self, jsonl_files: list[str], processor: ProcessData, augment: bool = True):
        self.jsonl_files = jsonl_files
        self.processor = processor
        self.augment = augment
        self.file_handles = {}
        self.global_indices = []
        
        for file_idx, jsonl_path in enumerate(self.jsonl_files):
            index_path = jsonl_path.replace('.jsonl', '_index.pkl')
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Missing index! Run build_index.py on {jsonl_path}")
            with open(index_path, 'rb') as f:
                offsets = pickle.load(f)
            for offset in offsets:
                self.global_indices.append((file_idx, offset))

    def __len__(self):
        return len(self.global_indices)

    def _get_file_handle(self, jsonl_path):
        if jsonl_path not in self.file_handles:
            self.file_handles[jsonl_path] = open(jsonl_path, 'rb')
        return self.file_handles[jsonl_path]

    def _lists_to_tuples(self, tokens_raw):
        tokens = []
        for event in tokens_raw:
            if isinstance(event, list) and len(event) > 0 and all(x == event[0] for x in event) and isinstance(event[0], str):
                tokens.append(event[0])
            elif isinstance(event, list) and len(event) == 5 and isinstance(event[0], list) and len(event[0]) == 2:
                inst_val = event[0][1]
                pitch_val = event[1][1]
                vel_val = event[2][1]
                onset_val = event[3][1]
                dur_val = event[4][1]
                
                if inst_val in ('<P>', '<BLANK>', '<MASK>', '<S>', '<E>', '<T>'):
                    if inst_val not in ('<P>', '<BLANK>'):
                        tokens.append(inst_val)
                else:
                    if inst_val == 'drum':
                        tokens.append((inst_val, pitch_val))
                    else:
                        tokens.append((inst_val, pitch_val, vel_val))
                    tokens.append(('onset', onset_val))
                    tokens.append(('dur', dur_val))
            else:
                if isinstance(event, list):
                    tokens.append(tuple(event))
                else:
                    tokens.append(event)
        return tokens

    def __getitem__(self, idx):
        file_idx, offset = self.global_indices[idx]
        f = self._get_file_handle(self.jsonl_files[file_idx])
        f.seek(offset)
        line_bytes = f.readline()
        
        entry = json.loads(line_bytes.decode('utf-8'))
        tokens_raw = entry.get("tokens", [])
        tokens = self._lists_to_tuples(tokens_raw)
        
        genre_id = torch.tensor(self.processor.get_genre_id(entry.get("genre", "unknown")), dtype=torch.long)
        
        if self.augment:
            tokens = self.processor.pitch_augmentation(tokens)

        if len(tokens) == 0: tokens = ['<S>']
        
        # We need SEQ_LEN + 1 tokens total to create [0:T] inputs and [1:T+1] targets
        total_needed = SEQ_LEN + 1
        
        if len(tokens) > total_needed:
            start_idx = random.randint(0, len(tokens) - total_needed)
            sliced_tokens = tokens[start_idx : start_idx + total_needed]
        else:
            sliced_tokens = tokens
            
        multi_hot = self.processor.get_instrument_multihot(sliced_tokens)
        tensor_seq = self.processor.format_variable_sequence(sliced_tokens, total_needed, pad_id=PAD_ID)

        # Standard Next-Token Prediction Offset
        input_tensor = tensor_seq[:-1]
        target_tensor = tensor_seq[1:]

        return {
            "input": input_tensor,
            "target": target_tensor,
            "genre": genre_id,
            "multi_hot": multi_hot
        }

def build_dataloader(jsonl_files: list[str], split: str, batch_size: int, augment: bool = False, num_workers: int = 4, distributed: bool = False) -> DataLoader:
    processor = ProcessData()
    dataset = ARContextDataset(jsonl_files=jsonl_files, processor=processor, augment=augment)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    
    def collate_fn(batch):
        return {
            "input": torch.stack([b["input"] for b in batch]),
            "target": torch.stack([b["target"] for b in batch]),
            "genre": torch.stack([b["genre"] for b in batch]),
            "multi_hot": torch.stack([b["multi_hot"] for b in batch])
        }
        
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=(not distributed),
        sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn
    )

def _distributed_rank_world():
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
        rank, world_size = _distributed_rank_world()
        remaining_indices = _epoch_indices(loader, epoch or 0)[start_global_sample_idx:]
        return _loader_from_indices(loader, remaining_indices[rank::world_size])

    if start_sample_idx <= 0:
        return loader
    if loader.sampler is None:
        return loader

    remaining_indices = list(loader.sampler)[start_sample_idx:]
    return _loader_from_indices(loader, remaining_indices)

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

_BNB_STATE_KEYS = {
    "state1", "state2", "qmap1", "qmap2", "absmax1", "absmax2",
    "max1", "max2", "new_max1", "new_max2", "unorm_vec",
    "__bnb_optimizer_quant_state__"
}
_TORCH_ADAMW_STATE_KEYS = {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}

def _checkpoint_path():
    return os.path.join(SAVE_DIR, "latest_checkpoint.pt")

def _load_training_checkpoint():
    checkpoint_path = _checkpoint_path()
    # checkpoint_path = os.path.join(SAVE_DIR, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        return None
    try:
        return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location='cpu')

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

def _default_8bit_backend():
    if bnb is None:
        return None
    if PREFER_PAGED_8BIT_OPTIMIZER and hasattr(bnb.optim, "PagedAdamW8bit"):
        return "paged_adamw8bit"
    if hasattr(bnb.optim, "AdamW8bit"):
        return "adamw8bit"
    return None

def _optimizer_backend_from_instance(optimizer):
    module_name = optimizer.__class__.__module__.lower()
    class_name = optimizer.__class__.__name__.lower()
    if module_name.startswith("bitsandbytes"):
        if "paged" in class_name:
            return "paged_adamw8bit"
        if "8bit" in class_name:
            return "adamw8bit"
    return "adamw"

def _checkpoint_optimizer_backend(checkpoint):
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
        return _default_8bit_backend() or "adamw8bit"
    if state_format == "torch_adamw":
        return "adamw"
    return None

def _is_bitsandbytes_backend(backend):
    return backend in ("adamw8bit", "paged_adamw8bit")

def _is_bitsandbytes_optimizer(optimizer):
    return optimizer.__class__.__module__.lower().startswith("bitsandbytes")

def _choose_optimizer_backend(checkpoint):
    requested_backend = _normalize_optimizer_backend(OPTIMIZER_BACKEND)
    valid_backends = {"auto", "adamw", "adamw8bit", "paged_adamw8bit"}
    if requested_backend not in valid_backends:
        raise ValueError(
            f"Unsupported OPTIMIZER_BACKEND={OPTIMIZER_BACKEND!r}. "
            f"Use one of {sorted(valid_backends)}."
        )

    if requested_backend != "auto":
        return requested_backend

    checkpoint_backend = _checkpoint_optimizer_backend(checkpoint)
    if checkpoint_backend is not None:
        return checkpoint_backend

    return _default_8bit_backend() or "adamw"

def build_optimizer(parameters, checkpoint=None, is_main_process=True):
    params = list(parameters)
    backend = _choose_optimizer_backend(checkpoint)
    requested_backend = _normalize_optimizer_backend(OPTIMIZER_BACKEND)
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

    optimizer_kwargs = dict(lr=LR, weight_decay=1e-2, betas=(0.9, 0.95))
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
):
    os.makedirs(SAVE_DIR, exist_ok=True)
    model_to_save = model.module if isinstance(model, DDP) else model
    optimizer_state_dict = _move_state_to_cpu(optimizer.state_dict())
    checkpoint = {
        'checkpoint_version': 3,
        'step': step, 
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer_state_dict,
        'optimizer_backend': optimizer_backend or _optimizer_backend_from_instance(optimizer),
        'optimizer_class': f"{optimizer.__class__.__module__}.{optimizer.__class__.__name__}",
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'epoch': epoch,
        'next_batch_idx': next_batch_idx,
        'next_sample_idx': next_sample_idx,
        'next_global_sample_idx': next_global_sample_idx,
        'batch_size': BATCH_SIZE,
        'accum_steps': ACCUM_STEPS,
        'world_size': dist.get_world_size() if dist.is_initialized() else 1,
        'micro_step': micro_step
    }
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    if rng_state is not None:
        checkpoint['rng_state'] = rng_state
    torch.save(checkpoint, _checkpoint_path())
    if is_best:
        torch.save(checkpoint, os.path.join(SAVE_DIR, "best_model.pt"))
    del checkpoint, optimizer_state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _checkpoint_base_lrs(checkpoint, optimizer=None):
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
        return [LR]
    return [
        group.get('initial_lr', group['lr'])
        for group in optimizer_state_dict['param_groups']
    ]

def _lrs_match_config(checkpoint_lrs, config_lr: float) -> bool:
    return all(math.isclose(lr, config_lr, rel_tol=1e-12, abs_tol=1e-16) for lr in checkpoint_lrs)

def load_checkpoint(model, optimizer, scheduler, device, scaler=None, checkpoint=None):
    if checkpoint is None:
        checkpoint = _load_training_checkpoint()
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
        'checkpoint_batch_size': checkpoint.get('batch_size', RESUME_CHECKPOINT_BATCH_SIZE),
        'checkpoint_accum_steps': checkpoint.get('accum_steps', RESUME_CHECKPOINT_ACCUM_STEPS),
        'checkpoint_world_size': checkpoint.get('world_size', RESUME_CHECKPOINT_WORLD_SIZE),
        'micro_step': checkpoint.get('micro_step'),
        'optimizer_backend': _optimizer_backend_from_instance(optimizer),
        'checkpoint_optimizer_backend': _checkpoint_optimizer_backend(checkpoint),
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
        elif ALLOW_OPTIMIZER_MIGRATION_TO_8BIT:
            resume_state['optimizer_skipped_reason'] = (
                "optimizer backend changed; ALLOW_OPTIMIZER_MIGRATION_TO_8BIT=True"
            )
        else:
            checkpoint_format = _optimizer_state_format(optimizer_state_dict)
            current_backend = _optimizer_backend_from_instance(optimizer)
            raise RuntimeError(
                "Optimizer checkpoint is not compatible with the current optimizer. "
                f"Checkpoint format={checkpoint_format!r}, current backend={current_backend!r}. "
                "Leave OPTIMIZER_BACKEND='auto' to resume exactly, or set "
                "ALLOW_OPTIMIZER_MIGRATION_TO_8BIT=True if you intentionally want "
                "to skip the old optimizer moments and continue with the new optimizer."
            )
    else:
        resume_state['optimizer_skipped_reason'] = "checkpoint has no optimizer_state_dict"

    checkpoint_lrs = _checkpoint_base_lrs(checkpoint, optimizer)
    lr_changed = not _lrs_match_config(checkpoint_lrs, LR)
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

def build_scheduler(optimizer, warmup_steps: int, total_steps: int, last_epoch: int = -1) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps: return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def build_resume_warmup_scheduler(
    optimizer,
    start_lrs: list[float],
    target_lr: float,
    warmup_steps: int,
    total_steps: int,
    resume_step: int
) -> LambdaLR:
    target_lrs = [target_lr for _ in optimizer.param_groups]
    start_factors = [
        start_lr / target_lr if target_lr > 0 else 1.0
        for start_lr in start_lrs
    ]
    
    for group, lr in zip(optimizer.param_groups, target_lrs):
        group['lr'] = lr
        group['initial_lr'] = lr

    def make_lr_lambda(start_factor: float):
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                alpha = step / max(1, warmup_steps)
                return start_factor + (1.0 - start_factor) * alpha
            progress = (step - warmup_steps) / max(1, total_steps - resume_step - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_lambda

    return LambdaLR(optimizer, [make_lr_lambda(factor) for factor in start_factors])

def _unwrap_model(model):
    return model.module if isinstance(model, DDP) else model

def chunked_lm_head_cross_entropy(
    model,
    hidden: torch.Tensor,
    target: torch.Tensor,
    chunk_size: int = LM_HEAD_CHUNK_SIZE,
    use_checkpoint: bool = False
) -> torch.Tensor:
    if hidden.shape[:2] != target.shape:
        raise ValueError(
            f"Hidden/target shape mismatch: hidden={tuple(hidden.shape)}, "
            f"target={tuple(target.shape)}"
        )

    model_to_use = _unwrap_model(model)
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        chunk_size = hidden.size(1)

    valid_tokens = target.ne(PAD_ID).sum().clamp_min(1)
    loss_sum = hidden.new_zeros((), dtype=torch.float32)

    for start in range(0, hidden.size(1), chunk_size):
        end = min(start + chunk_size, hidden.size(1))
        hidden_chunk = hidden[:, start:end, :]
        target_chunk = target[:, start:end].contiguous()

        def chunk_loss(chunk_hidden, chunk_target):
            logits = model_to_use.lm_head(chunk_hidden)
            return F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                chunk_target.reshape(-1),
                ignore_index=PAD_ID,
                reduction='sum'
            )

        if use_checkpoint and torch.is_grad_enabled():
            loss_sum = loss_sum + activation_checkpoint(
                chunk_loss,
                hidden_chunk,
                target_chunk,
                use_reentrant=False
            )
        else:
            loss_sum = loss_sum + chunk_loss(hidden_chunk, target_chunk)

    return loss_sum / valid_tokens

def train(n_steps: int = N_STEPS):
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = (local_rank == 0)
    _, world_size = _distributed_rank_world()

    writer = SummaryWriter(log_dir=os.path.join(SAVE_DIR, "runs", RUN_NAME)) if is_main_process else None

    train_loader = build_dataloader(JSONL_FILES, "train", BATCH_SIZE, augment=True, distributed=True)
    val_loader = build_dataloader(JSONL_FILES, "validation", BATCH_SIZE, augment=False, distributed=True)

    if is_main_process: print("Initializing AR Context Pretraining Tower...")
    model = ARContextModel().to(device)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    checkpoint = _load_training_checkpoint() if RESUME_TRAINING else None
    optimizer, optimizer_backend = build_optimizer(
        model.parameters(),
        checkpoint=checkpoint,
        is_main_process=is_main_process
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(AMP_DTYPE == torch.float16))

    update_step, best_val_loss = 0, float('inf')
    resume_state = None
    scheduler = build_scheduler(optimizer, WARMUP_STEPS, n_steps)
    if RESUME_TRAINING:
        update_step, best_val_loss, scheduler_restored, lr_changed, resume_start_lrs, resume_state = load_checkpoint(
            model, optimizer, scheduler, device, scaler, checkpoint=checkpoint
        )
        if lr_changed and resume_start_lrs is not None:
            scheduler = build_resume_warmup_scheduler(
                optimizer, resume_start_lrs, LR, WARMUP_STEPS, n_steps, update_step
            )
        elif not scheduler_restored:
            scheduler = build_scheduler(optimizer, WARMUP_STEPS, n_steps, last_epoch=update_step - 1 if update_step > 0 else -1)
    del checkpoint
    
    has_resume_position = resume_state is not None and resume_state['has_resume_position']
    checkpoint_accum_steps = resume_state.get('checkpoint_accum_steps') if resume_state is not None else None
    if has_resume_position and checkpoint_accum_steps in (None, ACCUM_STEPS):
        micro_step = resume_state['micro_step']
    else:
        micro_step = update_step * ACCUM_STEPS

    start_epoch = resume_state['epoch'] if has_resume_position else micro_step // len(train_loader)
    checkpoint_world_size = resume_state.get('checkpoint_world_size') if resume_state is not None else None
    if has_resume_position and resume_state.get('next_global_sample_idx') is not None:
        resume_global_sample_idx = resume_state['next_global_sample_idx']
    elif has_resume_position and resume_state.get('next_sample_idx') is not None:
        resume_global_sample_idx = resume_state['next_sample_idx'] * (checkpoint_world_size or world_size)
    elif has_resume_position:
        checkpoint_batch_size = resume_state.get('checkpoint_batch_size') or BATCH_SIZE
        resume_global_sample_idx = (
            resume_state['next_batch_idx']
            * checkpoint_batch_size
            * (checkpoint_world_size or world_size)
        )
    else:
        resume_global_sample_idx = 0

    if has_resume_position and resume_state.get('next_sample_idx') is not None:
        resume_sample_idx = resume_state['next_sample_idx']
    else:
        resume_sample_idx = resume_global_sample_idx // world_size
    resume_batch_idx = resume_global_sample_idx // max(1, BATCH_SIZE * world_size)

    if is_main_process and resume_state is not None:
        if has_resume_position:
            print(
                f"Resuming from step {update_step}, epoch {start_epoch}, "
                f"global sample {resume_global_sample_idx}, "
                f"rank sample {resume_sample_idx}, batch {resume_batch_idx}."
            )
            if resume_state.get('next_global_sample_idx') is None:
                print(
                    "Checkpoint has no next_global_sample_idx; inferred resume "
                    "position from legacy metadata."
                )
            if checkpoint_world_size not in (None, world_size):
                print(
                    f"World size changed from {checkpoint_world_size} to {world_size}; "
                    "re-sharding remaining epoch samples."
                )
            if checkpoint_accum_steps not in (None, ACCUM_STEPS):
                print(
                    f"ACCUM_STEPS changed from {checkpoint_accum_steps} to {ACCUM_STEPS}; "
                    "starting a fresh accumulation window."
                )
        else:
            print(f"Resuming legacy checkpoint from step {update_step}; starting at epoch {start_epoch}, batch 0.")
        if resume_state.get('optimizer_restored'):
            print(f"Restored optimizer state ({resume_state['optimizer_backend']}).")
        else:
            print(f"Optimizer state not restored: {resume_state.get('optimizer_skipped_reason', 'not available')}.")
        if resume_state.get('rng_restored'):
            print("Restored saved RNG state for training ranks.")
    if is_main_process:
        print(f"Using chunked LM-head CE with chunk size {LM_HEAD_CHUNK_SIZE}.")
    
    running_loss, log_steps = 0.0, 0
    model.train()
    epoch = start_epoch

    while update_step < n_steps:
        if hasattr(train_loader.sampler, 'set_epoch'): train_loader.sampler.set_epoch(epoch)
        epoch_loader = resume_epoch_loader(
            train_loader,
            epoch=epoch,
            start_global_sample_idx=resume_global_sample_idx
        )
        
        for local_batch_idx, batch in enumerate(epoch_loader):
            if update_step >= n_steps: break
            global_sample_idx = resume_global_sample_idx + (local_batch_idx * BATCH_SIZE * world_size)
            batch_idx = global_sample_idx // max(1, BATCH_SIZE * world_size)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            is_accumulating = (micro_step + 1) % ACCUM_STEPS != 0
            ddp_context = model.no_sync() if is_accumulating else contextlib.nullcontext()
            
            with ddp_context:
                with torch.amp.autocast(DEVICE, dtype=AMP_DTYPE):
                    hidden = model(
                        target=batch["input"],          
                        genre=batch["genre"],
                        multi_hot=batch["multi_hot"],
                        return_hidden=True
                    )
                    loss = chunked_lm_head_cross_entropy(
                        model,
                        hidden,
                        batch["target"],
                        chunk_size=LM_HEAD_CHUNK_SIZE,
                        use_checkpoint=True
                    ) / ACCUM_STEPS

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            running_loss += (loss.item() * ACCUM_STEPS)
            log_steps += 1
            micro_step += 1

            if not is_accumulating:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()
                    
                scheduler.step()
                optimizer.zero_grad()
                update_step += 1

                if is_main_process and (update_step % LOG_EVERY == 0) and log_steps > 0:
                    avg_loss = running_loss / log_steps
                    perplexity = math.exp(min(avg_loss, 20.0))
                    print(f"  Step {update_step:>6} | CE Loss: {avg_loss:>6.4f} | PPL: {perplexity:>6.2f}")
                    writer.add_scalar('Train/AR_CE_Loss', avg_loss, update_step)
                    writer.add_scalar('Train/AR_Perplexity', perplexity, update_step)
                    writer.add_scalar('Hyperparameters/LR', optimizer.param_groups[0]['lr'], update_step)
                    running_loss, log_steps = 0.0, 0

                if update_step > 0 and update_step % VAL_EVERY == 0:
                    val_loss = evaluate_validation(model, val_loader, local_rank, device)
                    checkpoint_rng_state = collect_rng_state_for_checkpoint(device)
                    if is_main_process:
                        val_ppl = math.exp(min(val_loss, 20.0))
                        print(f"\n--- Validation Step {update_step} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} ---")
                        writer.add_scalar('Validation/AR_CE_Loss', val_loss, update_step)
                        is_best = val_loss < best_val_loss
                        if is_best: best_val_loss = val_loss
                        next_global_sample_idx = global_sample_idx + (BATCH_SIZE * world_size)
                        next_sample_idx = next_global_sample_idx // world_size
                        next_batch_idx = next_global_sample_idx // max(1, BATCH_SIZE * world_size)
                        next_epoch = epoch
                        if next_global_sample_idx >= len(train_loader.dataset):
                            next_epoch += 1
                            next_batch_idx = 0
                            next_sample_idx = 0
                            next_global_sample_idx = 0
                        save_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            update_step,
                            best_val_loss,
                            next_epoch,
                            next_batch_idx,
                            next_sample_idx,
                            next_global_sample_idx,
                            micro_step,
                            scaler,
                            is_best,
                            optimizer_backend=optimizer_backend,
                            rng_state=checkpoint_rng_state
                        )
        resume_global_sample_idx = 0
        resume_sample_idx = 0
        resume_batch_idx = 0
        epoch += 1

    if is_main_process and writer: writer.close()
    cleanup_ddp()
    return model

@torch.no_grad()
def evaluate_validation(model, val_loader, local_rank, device, max_batches: int = 50):
    model.eval()
    total_loss, steps = 0.0, 0
    iterator = tqdm(val_loader, desc="Validation", total=min(len(val_loader), max_batches), leave=False) if local_rank == 0 else val_loader

    for batch in iterator:
        if steps >= max_batches: break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        with torch.amp.autocast(DEVICE, dtype=AMP_DTYPE):
            hidden = model(
                target=batch["input"],
                genre=batch["genre"],
                multi_hot=batch["multi_hot"],
                return_hidden=True
            )
            loss = chunked_lm_head_cross_entropy(
                model,
                hidden,
                batch["target"],
                chunk_size=LM_HEAD_CHUNK_SIZE,
                use_checkpoint=False
            )
            
        total_loss += loss.item()
        steps += 1

    local_avg_loss = total_loss / max(1, steps)
    if torch.distributed.is_initialized():
        metrics = torch.tensor([local_avg_loss], device=device)
        torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
        global_avg_loss = (metrics / torch.distributed.get_world_size())[0].item()
    else:
        global_avg_loss = local_avg_loss

    model.train()
    return global_avg_loss

if __name__ == "__main__":
    model = train(n_steps=N_STEPS)
