"""Three-task melody model with shared groove and retained-mix denoising."""
import hashlib
from pathlib import Path

import torch
from torch import nn

from improvnet.model.ar_model import ARContextModel
from improvnet.utils.melody import MelodyTokenizer, TASKS, TASK_TOKENS
from improvnet.utils.melody_style import STYLE_FEATURES
from improvnet.utils.melody_stems import DRUM_POLICY, DRUM_KIT_PITCHES, DRUM_KIT_ALIASES


class MelodyRefinementModel(ARContextModel):
    def __init__(self, style_feature_count=None, **kwargs):
        self.style_feature_count=style_feature_count or len(STYLE_FEATURES)
        kwargs.setdefault('vocab_size', MelodyTokenizer().vocab_size)
        super().__init__(**kwargs)
        dim = self.token_emb.embedding_dim
        width = min(128, dim)
        self.strength_proj = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, dim, bias=False))
        self.style_proj = nn.Sequential(nn.Linear(2 * self.style_feature_count, width), nn.SiLU(), nn.Linear(width, dim, bias=False))
        nn.init.zeros_(self.strength_proj[-1].weight)
        nn.init.zeros_(self.style_proj[-1].weight)
        self.model_kwargs = dict(kwargs,style_feature_count=self.style_feature_count)

    def forward(self, target, strength=None, style_values=None, style_mask=None, **kwargs):
        extra = None
        if kwargs.get('past_key_values') is None:
            strength = torch.as_tensor(strength, device=target.device, dtype=torch.float32).reshape(-1, 1)
            if strength.shape != (target.shape[0], 1) or not torch.isfinite(strength).all() or not ((strength >= 0) & (strength <= 1)).all():
                raise ValueError('Provide one finite strength in [0,1] per example')
            shape = (target.shape[0], self.style_feature_count)
            if style_values is None and style_mask is None:
                style_values = torch.zeros(shape, device=target.device)
                style_mask = torch.zeros_like(style_values)
            if style_values is None or style_mask is None:
                raise ValueError('Provide rhythm values and mask together')
            values = torch.as_tensor(style_values, device=target.device, dtype=torch.float32)
            mask = torch.as_tensor(style_mask, device=target.device, dtype=torch.float32)
            if values.shape != shape or mask.shape != shape or not torch.isfinite(values).all() or not ((mask == 0) | (mask == 1)).all():
                raise ValueError('Invalid rhythm-only controls')
            extra = self.strength_proj(strength) + self.style_proj(torch.cat((values * mask, mask), -1))
        return super().forward(target, extra_condition=extra, **kwargs)


def initialize_from_ar(model, checkpoint):
    state = {k.removeprefix('module.'): v for k, v in checkpoint.get('model_state_dict', checkpoint).items()}
    meta = checkpoint.get('refinement', {})
    tok = MelodyTokenizer()
    if meta.get('tokenizer') and meta['tokenizer'] != tok.signature()[:len(meta['tokenizer'])]:
        raise ValueError('Initialization tokenizer is not a prefix of the new tokenizer')
    expected, migrated = model.state_dict(), {}
    for key, value in state.items():
        if key not in expected:
            raise ValueError(f'Unexpected parameter: {key}')
        if key in ('token_emb.weight', 'lm_head.weight') and value.shape[0] < expected[key].shape[0]:
            if value.shape[1:] != expected[key].shape[1:]:
                raise ValueError(f'Architecture mismatch: {key}')
            expanded = expected[key].clone()
            expanded[:value.shape[0]] = value
            old = MelodyTokenizer()
            if old.tok_to_id['<HARMONIZE>'] < value.shape[0] <= old.tok_to_id['<ARRANGE>']:
                expanded[tok.tok_to_id['<ARRANGE>']] = value[old.tok_to_id['<HARMONIZE>']]
            migrated[key] = expanded
        elif key == 'style_proj.0.weight' and value.shape!=expected[key].shape:
            old_features=meta.get('style_features',[])
            if not old_features or value.shape[1]!=2*len(old_features):
                raise ValueError('Unknown rhythm initialization schema')
            expanded=expected[key].clone()
            for i,name in enumerate(old_features):
                if name in STYLE_FEATURES:
                    j=STYLE_FEATURES.index(name)
                    expanded[:,j]=value[:,i]
                    expanded[:,j+len(STYLE_FEATURES)]=value[:,i+len(old_features)]
            migrated[key]=expanded
        elif value.shape == expected[key].shape:
            migrated[key] = value
        else:
            raise ValueError(f'Architecture mismatch for {key}: {value.shape}')
    missing = set(expected) - set(migrated)
    if any(not k.startswith(('style_proj.', 'strength_proj.')) for k in missing):
        raise ValueError(f'Missing parameters: {missing}')
    model.load_state_dict(migrated, strict=False)


def checkpoint_metadata(model, config):
    fields=('CHUNK_TOKENS','HISTORY_TOKENS','HISTORY_PROB','POLY_HISTORY_PROB',
            'MIX_RESPONSE_TOKENS','HARMONIZE_EXPANSION','HISTORY_CORRUPTION_PROB',
            'HISTORY_EDIT_FRACTION','HISTORY_ADDITION_WEIGHT','HISTORY_DELETION_WEIGHT',
            'SKYLINE_TOLERANCE_MS','MAX_CORRUPTIONS','RHYTHM_STYLE_PROB','SEQ_LEN',
            'CLASSICAL_RATIO','JAZZ_BLUES_REPEATS','DATA_SPLIT_SEED','MULTI_INSTRUMENT_PROB',
            'STEM_DRUM_PROB','JAZZ_INSTRUMENT_WEIGHT','AR_MIXTURE_PROB',
            'GENERIC_MERGE_PROB','STEM_TIMING_JITTER_MS','STEM_PITCH_SHIFT_PROB',
            'STEM_PITCH_SHIFT_SEMITONES','STEM_PITCH_SHIFT_MIN_PROB','STEM_PITCH_CLEAN_PROB',
            'HARMONIZE_MIN_MASK_RATIO','HARMONIZE_FULL_MASK_PROB',
            'PITCH_CORRUPTION_SEMITONES','TIMING_CORRUPTION_MS')
    manifest=Path(config.DATA_FILE)
    return dict(version=config.PROTOCOL_VERSION,model_kwargs=model.model_kwargs,tokenizer=MelodyTokenizer().signature(),
                genres=list(config.GENRES),style_features=list(STYLE_FEATURES),
                tasks=list(TASKS),task_tokens={task:TASK_TOKENS[task] for task in TASKS},task_weights=list(config.TASK_WEIGHTS),
                **{name.lower():getattr(config,name) for name in fields},
                data_file=str(manifest.resolve()),
                data_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                history_policy='mono_refine_poly_other_tasks',harmony_conditioning=False,
                harmonize_output='pitched_mix',stem_output='clean_full_mix_from_corrupted_retained_mix',
                drum_policy=DRUM_POLICY,drum_kit_pitches=sorted(DRUM_KIT_PITCHES),
                drum_articulation_aliases={str(k):v for k,v in DRUM_KIT_ALIASES.items()},
                stem_pitch_scope='non_skyline_pitched_notes',
                stem_pitch_probability='per_example_uniform_min_to_max_with_explicit_clean_cases',
                stem_pitch_order='preserve_same_onset_retained_note_order',
                stem_onset_jitter='shared_exact_attack_groups_order_preserved',
                refinement_corruptions='ornaments_first_then_contour40_generic_merge10_pitch25_timing25',
                refinement_strength='max_requested_and_fraction_of_attacks_removed_by_ornament_merge',
                refinement_additional_strength_sampling='5_percent_zero_otherwise_equal_10_to_100_percent_levels',
                refinement_pitch_timing_rate='requested_fraction_of_remaining_notes_or_attack_groups',
                groove_protocol='local_eight_bar_windows_shared_inferred_pulse_v1',
                data_policy_revision=10,prepared_split_ratios=dict(config.DATA_SPLIT_RATIOS),
                validation_sampling='one_window_per_file',ar_header_revision=2,
                midi_recovery_policy='invalid_key_metadata_and_sysex_only',
                jazz_blues_sampling='four_full_file_passes_plus_eligible_mixture_tasks')


def validate_checkpoint(checkpoint,expected=None,allow_data_policy_update=False):
    meta=checkpoint.get('refinement',{})
    if meta.get('version') != 9:
        raise ValueError('Expected the V5 three-task protocol; use the older branch for older checkpoints')
    if meta.get('tokenizer') != MelodyTokenizer().signature() or meta.get('style_features') != list(STYLE_FEATURES):
        raise ValueError('Tokenizer or rhythm features differ')
    if meta.get('tasks') != list(TASKS):
        raise ValueError('Checkpoint tasks differ')
    if expected is not None and meta!=expected:
        raise ValueError('Resume settings differ; use a fresh run directory')
    return meta
