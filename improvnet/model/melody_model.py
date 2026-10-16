"""AR transformer with corruption strength and masked aggregate style controls."""
import torch
import hashlib
from pathlib import Path
from torch import nn
from improvnet.model.ar_model import ARContextModel
from improvnet.utils.melody import MelodyTokenizer
from improvnet.utils.melody_style import STYLE_FEATURES


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
        self.style_proj = nn.Sequential(nn.Linear(2 * len(STYLE_FEATURES), width), nn.SiLU(),
                                        nn.Linear(width, dim, bias=False))
        nn.init.zeros_(self.style_proj[-1].weight)
        self.model_kwargs = dict(kwargs)

    def forward(self, target, strength=None, style_values=None, style_mask=None, **kwargs):
        extra = None
        if kwargs.get('past_key_values') is None:
            if strength is None:
                raise ValueError('Provide corruption strength for the initial prompt')
            strength = torch.as_tensor(strength, device=target.device, dtype=torch.float32).reshape(-1, 1)
            if strength.shape[0] != target.shape[0] or not torch.isfinite(strength).all() or not ((strength >= 0) & (strength <= 1)).all():
                raise ValueError('One finite strength in [0,1] is required per example')
            extra = self.strength_proj(strength)
            shape = (target.shape[0], len(STYLE_FEATURES))
            if style_values is None and style_mask is None:
                style_values = torch.zeros(shape, device=target.device)
                style_mask = torch.zeros_like(style_values)
            elif style_values is None or style_mask is None:
                raise ValueError('Provide style values and presence mask together')
            style_values = torch.as_tensor(style_values, device=target.device, dtype=torch.float32)
            style_mask = torch.as_tensor(style_mask, device=target.device, dtype=torch.float32)
            if style_values.shape != shape or style_mask.shape != shape or not torch.isfinite(style_values).all() or not ((style_mask == 0) | (style_mask == 1)).all():
                raise ValueError('Invalid style values or presence mask')
            extra = extra + self.style_proj(torch.cat((style_values * style_mask, style_mask), dim=-1))
        return super().forward(target, extra_condition=extra, **kwargs)


def initialize_from_ar(model, checkpoint):
    """Weight-only migration from AR/earlier refiners; retain all existing IDs."""
    state = checkpoint.get('model_state_dict', checkpoint)
    meta = checkpoint.get('refinement', {})
    if meta.get('tokenizer') and meta['tokenizer'] != MelodyTokenizer().signature()[:len(meta['tokenizer'])]:
        raise ValueError('Pretrained tokenizer is not a prefix of the current tokenizer')
    if meta.get('style_features') and meta['style_features'] != list(STYLE_FEATURES):
        raise ValueError('Pretrained style feature order differs')
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    expected = model.state_dict()
    migrated = {}
    for key, value in state.items():
        if key not in expected:
            raise ValueError(f'Unexpected pretrained parameter: {key}')
        if key in ('token_emb.weight', 'lm_head.weight'):
            base = MelodyTokenizer().base_vocab_size
            if value.shape[0] not in (base, base + 3, base + 4, expected[key].shape[0]) or value.shape[1:] != expected[key].shape[1:]:
                raise ValueError(f'AR vocabulary/width mismatch for {key}: {value.shape}')
            expanded = expected[key].clone()
            expanded[:value.shape[0]].copy_(value)
            migrated[key] = expanded
        elif value.shape != expected[key].shape:
            raise ValueError(f'Architecture mismatch for {key}: {value.shape} vs {expected[key].shape}')
        else:
            migrated[key] = value
    missing = set(expected) - set(migrated)
    if any(not k.startswith(('strength_proj.', 'style_proj.')) for k in missing):
        raise ValueError(f'Missing pretrained parameters: {missing}')
    model.load_state_dict(migrated, strict=False)


def checkpoint_metadata(model, config):
    index = Path(config.DATA_FILE).with_suffix('.index.json')
    return dict(version=4, model_kwargs=model.model_kwargs,
                tokenizer=MelodyTokenizer().signature(), genres=list(config.GENRES),
                chunk_tokens=config.CHUNK_TOKENS, history_tokens=config.HISTORY_TOKENS,
                history_prob=config.HISTORY_PROB, harmony_expansion=config.HARMONY_EXPANSION,
                history_policy='clean_target_view', input_mono_prob=config.INPUT_MONO_PROB,
                harmony_history_view='poly', stem_history_view='full_mix',
                stem_grouping='tokenizer_instrument_class', stem_drum_weight=config.STEM_DRUM_WEIGHT,
                multi_instrument_prob=config.MULTI_INSTRUMENT_PROB,
                classical_prob=config.CLASSICAL_PROB,
                harmony_style_prob=config.HARMONY_STYLE_PROB, rhythm_style_prob=config.RHYTHM_STYLE_PROB,
                style_features=list(STYLE_FEATURES),
                skyline_tolerance_ms=config.SKYLINE_TOLERANCE_MS,
                seq_len=config.SEQ_LEN, max_corruptions=config.MAX_CORRUPTIONS,
                task_weights=list(config.TASK_WEIGHTS), data_seed=config.DATA_SPLIT_SEED,
                data_file=str(Path(config.DATA_FILE).resolve()),
                data_index_sha256=hashlib.sha256(index.read_bytes()).hexdigest() if index.exists() else None)


def validate_checkpoint(checkpoint, expected=None):
    meta = checkpoint.get('refinement')
    if meta and meta.get('version') in (1, 2, 3):
        raise ValueError('Old refinement protocol: start a fresh targeted-stem run; old weights may initialize it but cannot resume it')
    if not meta or meta.get('version') != 4:
        raise ValueError('Expected a melody-refinement checkpoint, not the original AR checkpoint')
    if meta['tokenizer'] != MelodyTokenizer().signature():
        raise ValueError('Tokenizer differs from the checkpoint')
    if meta.get('style_features') != list(STYLE_FEATURES):
        raise ValueError('Style feature schema differs from the checkpoint')
    if expected is not None and meta != expected:
        raise ValueError('Resume settings differ from the checkpoint; use a fresh run directory')
    return meta
