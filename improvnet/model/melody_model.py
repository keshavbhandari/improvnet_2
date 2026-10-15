"""The existing AR transformer with a scalar corruption-strength condition."""
import torch
import hashlib
from pathlib import Path
from torch import nn
from improvnet.model.ar_model import ARContextModel
from improvnet.utils.melody import MelodyTokenizer


class MelodyRefinementModel(ARContextModel):
    def __init__(self, **kwargs):
        kwargs.setdefault('vocab_size', MelodyTokenizer().vocab_size)
        super().__init__(**kwargs)
        dim = self.token_emb.embedding_dim
        width = min(128, dim)
        self.strength_proj = nn.Sequential(nn.Linear(1, width), nn.SiLU(),
                                           nn.Linear(width, dim, bias=False))
        # Start as the pretrained model; learn to use the new condition.
        nn.init.zeros_(self.strength_proj[-1].weight)
        self.model_kwargs = dict(kwargs)

    def forward(self, target, strength=None, **kwargs):
        extra = None
        if kwargs.get('past_key_values') is None:
            if strength is None:
                raise ValueError('Provide corruption strength for the initial prompt')
            strength = torch.as_tensor(strength, device=target.device, dtype=torch.float32).reshape(-1, 1)
            if strength.shape[0] != target.shape[0] or not torch.isfinite(strength).all() or not ((strength >= 0) & (strength <= 1)).all():
                raise ValueError('One finite strength in [0,1] is required per example')
            extra = self.strength_proj(strength)
        return super().forward(target, extra_condition=extra, **kwargs)


def initialize_from_ar(model, checkpoint):
    """Strict weight-only migration; append rows without changing any old ID."""
    state = checkpoint.get('model_state_dict', checkpoint)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    expected = model.state_dict()
    migrated = {}
    for key, value in state.items():
        if key not in expected:
            raise ValueError(f'Unexpected pretrained parameter: {key}')
        if key in ('token_emb.weight', 'lm_head.weight'):
            base = MelodyTokenizer().base_vocab_size
            if value.shape != (base, expected[key].shape[1]):
                raise ValueError(f'AR vocabulary/width mismatch for {key}: {value.shape}')
            expanded = expected[key].clone()
            expanded[:base].copy_(value)
            migrated[key] = expanded
        elif value.shape != expected[key].shape:
            raise ValueError(f'Architecture mismatch for {key}: {value.shape} vs {expected[key].shape}')
        else:
            migrated[key] = value
    missing = set(expected) - set(migrated)
    if missing != {k for k in expected if k.startswith('strength_proj.')}:
        raise ValueError(f'Missing pretrained parameters: {missing}')
    model.load_state_dict(migrated, strict=False)


def checkpoint_metadata(model, config):
    index = Path(config.DATA_FILE).with_suffix('.index.json')
    return dict(version=1, model_kwargs=model.model_kwargs,
                tokenizer=MelodyTokenizer().signature(), genres=list(config.GENRES),
                window_ms=config.WINDOW_MS, context_windows=config.CONTEXT_WINDOWS,
                skyline_tolerance_ms=config.SKYLINE_TOLERANCE_MS,
                seq_len=config.SEQ_LEN, max_corruptions=config.MAX_CORRUPTIONS,
                task_weights=list(config.TASK_WEIGHTS), data_seed=config.DATA_SPLIT_SEED,
                data_file=str(Path(config.DATA_FILE).resolve()),
                data_index_sha256=hashlib.sha256(index.read_bytes()).hexdigest() if index.exists() else None)


def validate_checkpoint(checkpoint, expected=None):
    meta = checkpoint.get('refinement')
    if not meta or meta.get('version') != 1:
        raise ValueError('Expected a melody-refinement checkpoint, not the original AR checkpoint')
    if meta['tokenizer'] != MelodyTokenizer().signature():
        raise ValueError('Tokenizer differs from the checkpoint')
    if expected is not None and meta != expected:
        raise ValueError('Resume settings differ from the checkpoint; use a fresh run directory')
    return meta
