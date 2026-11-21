"""V15: original AR blocks conditioned by a projected reference-memory prefix."""
import hashlib
from pathlib import Path

import torch
from torch import nn

from improvnet.model.ar_model import ARContextModel
from improvnet.utils.melody import MelodyTokenizer, TASKS, TASK_TOKENS
from improvnet.utils.melody_stems import DRUM_POLICY, DRUM_KIT_PITCHES, DRUM_KIT_ALIASES, PERCUSSION_MODES

from improvnet.model.reference_memory import ReferenceMemory
from improvnet.utils.drum_profile import DRUM_PROFILE_DIM, DRUM_PROFILE_SCHEMA

PROTOCOL_VERSION = 20
INFERENCE_PROTOCOLS = (18, PROTOCOL_VERSION)


class MelodyRefinementModel(ARContextModel):
    def __init__(self, style_dim=256, beat_dim=256, memory_tokens=80, memory_dim=256,
                 memory_heads=4, drum_memory_tokens=None, **kwargs):
        if style_dim!=256 or beat_dim!=256:raise ValueError('Reference feature width must be 256')
        if drum_memory_tokens is None:
            drum_memory_tokens=0 if memory_tokens==64 else 16
        if ((memory_tokens,drum_memory_tokens) not in ((64,0),(80,16))
                or memory_dim<4 or memory_heads<1 or memory_dim%memory_heads):
            raise ValueError('Expected V15 80-token separate memory or V14 64-token shared memory')
        self.memory_tokens=memory_tokens
        self.drum_memory_tokens=drum_memory_tokens
        self.prefix_tokens=2+memory_tokens
        kwargs.setdefault('vocab_size',MelodyTokenizer().vocab_size)
        super().__init__(**kwargs)
        dim=self.token_emb.embedding_dim
        width=min(128,dim)
        self.strength_proj=nn.Sequential(nn.Linear(1,width),nn.SiLU(),nn.Linear(width,dim,bias=False))
        self.percussion_embedding=nn.Embedding(len(PERCUSSION_MODES),dim,padding_idx=0)
        self.reference_memory=ReferenceMemory(256,memory_dim,32,memory_heads,drum_memory_tokens//2)
        self.memory_projection=nn.Linear(memory_dim,dim,bias=False)
        self.drum_memory_projection=nn.Linear(memory_dim,dim,bias=False) if drum_memory_tokens else None
        nn.init.normal_(self.memory_projection.weight,std=.02)
        if self.drum_memory_projection is not None:
            nn.init.normal_(self.drum_memory_projection.weight,std=.02)
        nn.init.zeros_(self.strength_proj[-1].weight)
        nn.init.zeros_(self.percussion_embedding.weight)
        self.model_kwargs=dict(kwargs,style_dim=style_dim,beat_dim=beat_dim,memory_tokens=memory_tokens,
            memory_dim=memory_dim,memory_heads=memory_heads)
        if drum_memory_tokens:
            self.model_kwargs['drum_memory_tokens']=drum_memory_tokens

    def project_memory(self,memory):
        if not self.drum_memory_tokens:
            return self.memory_projection(memory)
        memory=memory.reshape(memory.size(0),2,self.memory_tokens//2,-1)
        references=self.memory_projection(memory[:,:,:32])
        drums=self.drum_memory_projection(memory[:,:,32:])
        return torch.cat((references,drums),2).flatten(1,2)

    def forward(self,target,strength=None,percussion_mode=None,reference_features=None,
                reference_mask=None,reference_positions=None,drum_profile=None,**kwargs):
        extra=memory=None
        if kwargs.get('past_key_values') is None:
            batch=target.size(0)
            strength=torch.as_tensor(strength,device=target.device,dtype=torch.float32).reshape(-1,1)
            if strength.shape!=(batch,1) or not torch.isfinite(strength).all() or not ((strength>=0)&(strength<=1)).all():
                raise ValueError('Provide one finite strength in [0,1] per example')
            palette=torch.zeros(batch,device=target.device,dtype=torch.long) if percussion_mode is None else percussion_mode
            if palette.shape!=(batch,) or palette.dtype!=torch.long or not ((palette>=0)&(palette<len(PERCUSSION_MODES))).all():
                raise ValueError('Invalid percussion mode')
            extra=self.strength_proj(strength)+self.percussion_embedding(palette)
            if reference_features is None:
                reference_features=torch.zeros(batch,2,1,256,device=target.device)
                reference_mask=torch.zeros(batch,2,1,dtype=torch.bool,device=target.device)
                reference_positions=torch.zeros(batch,2,1,3,device=target.device)
            if drum_profile is None:drum_profile=torch.zeros(batch,2,DRUM_PROFILE_DIM,device=target.device)
            if reference_mask is None or reference_positions is None:
                raise ValueError('Provide reference features, positions and mask together')
            if (reference_features.ndim!=4 or reference_features.shape[:2]!=(batch,2) or reference_features.shape[-1]!=256
                    or reference_mask.shape!=reference_features.shape[:3] or reference_mask.dtype!=torch.bool
                    or reference_positions.shape!=(*reference_features.shape[:3],3)
                    or drum_profile.shape!=(batch,2,DRUM_PROFILE_DIM)):
                raise ValueError('Invalid reference features/mask/positions or drum profile')
            if not all(torch.isfinite(t).all() for t in (reference_features,reference_positions,drum_profile)):
                raise ValueError('Reference features and drum profiles must be finite')
            memory=self.project_memory(self.reference_memory(reference_features,reference_mask,reference_positions,drum_profile))
        return super().forward(target,extra_condition=extra,prefix_memory=memory,**kwargs)


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
        elif value.shape == expected[key].shape:
            migrated[key] = value
        else:
            raise ValueError(f'Architecture mismatch for {key}: {value.shape}')
    missing = set(expected) - set(migrated)
    if any(not (k.startswith(('reference_memory.', 'memory_projection.', 'drum_memory_projection.', 'strength_proj.', 'percussion_embedding.'))) for k in missing):
        raise ValueError(f'Missing parameters: {missing}')
    model.load_state_dict(migrated, strict=False)


def checkpoint_metadata(model, config):
    fields=('CHUNK_TOKENS','HISTORY_TOKENS','HISTORY_PROB','POLY_HISTORY_PROB',
            'HISTORY_CORRUPTION_PROB','STEM_SOURCE_CORRUPTION_PROB',
            'HISTORY_EDIT_FRACTION','HISTORY_ADDITION_WEIGHT','HISTORY_DELETION_WEIGHT',
            'SKYLINE_TOLERANCE_MS','MAX_CORRUPTIONS','REFERENCE_PROB','SEQ_LEN',
            'CLASSICAL_RATIO','JAZZ_BLUES_REPEATS','DATA_SPLIT_SEED','MULTI_INSTRUMENT_PROB',
            'STEM_DRUM_PROB','JAZZ_INSTRUMENT_WEIGHT','AR_MIXTURE_PROB',
            'GENERIC_MERGE_PROB','STEM_TIMING_JITTER_MS','STEM_PITCH_SHIFT_PROB',
            'STEM_PITCH_SHIFT_SEMITONES','STEM_PITCH_SHIFT_MIN_PROB','STEM_PITCH_CLEAN_PROB',
            'REFINE_MIN_MASK_RATIO','REFINE_FULL_MASK_PROB','HARMONIZE_MIN_MASK_RATIO','HARMONIZE_FULL_MASK_PROB',
            'PITCH_CORRUPTION_SEMITONES','TIMING_CORRUPTION_MS','KIT_DRUM_EXAMPLE_PROB','REFERENCE_EXACT_PROB','REFERENCE_WINDOWS','REFERENCE_LOCAL_BINS','REFERENCE_PITCH_SEMITONES','JAZZ_RAMP_START_STEP','JAZZ_RAMP_END_STEP')
    manifest=Path(config.DATA_FILE)
    tok=MelodyTokenizer()
    return dict(version=config.PROTOCOL_VERSION,model_kwargs=model.model_kwargs,tokenizer=tok.signature(),
                genres=list(config.GENRES),style_dim=config.STYLE_DIM,
                style_encoder=str(config.STYLE_ENCODER),style_encoder_sha256=config.STYLE_ENCODER_SHA256,
                beat_dim=config.BEAT_DIM,beat_encoder=str(config.BEAT_ENCODER),beat_encoder_sha256=config.BEAT_ENCODER_SHA256,
                reference_tokens=config.REFERENCE_TOKENS,reference_guard_ms=config.REFERENCE_GUARD_MS,
                reference_max_distance_ms=config.REFERENCE_MAX_DISTANCE_MS,
                reference_architecture='paired_separate_drum_prefix_memory_v3',prefix_tokens=model.prefix_tokens,
                reference_memory_tokens_per_window=32,drum_memory_tokens_per_window=8,drum_profile_schema=DRUM_PROFILE_SCHEMA,
                tasks=list(TASKS),task_tokens={task:TASK_TOKENS[task] for task in TASKS},task_weights=list(config.TASK_WEIGHTS),
                **{name.lower():getattr(config,name) for name in fields},
                data_file=str(manifest.resolve()),
                data_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                history_policy='mono_refine_poly_other_tasks',
                harmonize_output='pitched_mix',stem_output='clean_full_mix_from_retained_mix_optional_corruption',
                drum_policy=DRUM_POLICY,percussion_modes=dict(PERCUSSION_MODES),drum_kit_pitches=sorted(DRUM_KIT_PITCHES),
                drum_articulation_aliases={str(k):v for k,v in DRUM_KIT_ALIASES.items()},
                stem_pitch_scope='non_skyline_pitched_notes',
                stem_pitch_probability='per_example_uniform_min_to_max_with_explicit_clean_cases',
                stem_pitch_order='preserve_same_onset_retained_note_order',
                stem_onset_jitter='shared_exact_attack_groups_order_preserved',
                refinement_corruptions='ornaments_first_then_contour40_generic_merge10_pitch25_timing25',
                refinement_strength='max_requested_and_fraction_of_attacks_removed_by_ornament_merge',
                refinement_additional_strength_sampling='5_percent_zero_otherwise_equal_10_to_100_percent_levels',
                refinement_pitch_timing_rate='requested_fraction_of_remaining_notes_or_attack_groups',
                reference_policy='two_consecutive_safe_shared_windows_pairwise_pitch_augmentation_exact_or_trio_v5',
                refinement_input_velocity=90,reference_conditioned_tasks=list(TASKS),refinement_timing='90_to_100_percent_masked_after_melodic_corruptions',data_policy_revision=21,prepared_split_ratios=dict(config.DATA_SPLIT_RATIOS),
                validation_sampling='one_window_per_file',ar_header_revision=3,
                midi_recovery_policy='invalid_or_truncated_key_time_signature_metadata_and_sysex_only',
                jazz_blues_sampling='step_scheduled_jazz_weight_with_fixed_task_weights',
                drum_profile_source='one_profile_per_conditioning_reference_window')


def validate_checkpoint(checkpoint,expected=None):
    meta=checkpoint.get('refinement',{})
    version=meta.get('version')
    legacy=version==18
    prefix,architecture=(66,'paired_self_attention_prefix_memory_v2') if legacy else (82,'paired_separate_drum_prefix_memory_v3')
    model_kwargs=meta.get('model_kwargs',{})
    if (version not in INFERENCE_PROTOCOLS or meta.get('prefix_tokens')!=prefix
            or meta.get('percussion_modes')!=PERCUSSION_MODES
            or meta.get('reference_windows')!=2 or meta.get('refinement_input_velocity')!=90
            or meta.get('reference_conditioned_tasks')!=list(TASKS)
            or meta.get('reference_architecture')!=architecture
            or model_kwargs.get('memory_tokens')!=(64 if legacy else 80)
            or (not legacy and (model_kwargs.get('drum_memory_tokens')!=16
                or meta.get('drum_memory_tokens_per_window')!=8 or meta.get('reference_memory_tokens_per_window')!=32))
            or meta.get('drum_profile_schema')!=DRUM_PROFILE_SCHEMA):
        raise ValueError('Expected V15 protocol 20 separate-memory or V14 protocol 18 checkpoint; initialize fresh training from pretrained AR')
    if meta.get('tokenizer') != MelodyTokenizer().signature() or meta.get('style_dim') != 256 or not meta.get('style_encoder_sha256') or meta.get('beat_dim') != 256 or not meta.get('beat_encoder_sha256'):
        raise ValueError('Tokenizer or reference encoder metadata differ')
    if meta.get('tasks') != list(TASKS):
        raise ValueError('Checkpoint tasks differ')
    if expected is not None and meta!=expected:
        raise ValueError('Resume settings differ; use a fresh run directory')
    return meta
