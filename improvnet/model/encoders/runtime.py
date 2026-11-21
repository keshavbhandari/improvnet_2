"""Frozen encoder used identically by AR training and MIDI inference."""
from dataclasses import replace
import hashlib
import json
import random
from pathlib import Path
from contextlib import nullcontext
import tempfile
import torch
from improvnet.model.encoders.model import ContinuityConfig
from improvnet.model.encoders.style.model import StyleEncoder
from improvnet.model.encoders.beat.model import BeatEncoder
from improvnet.utils.melody import MelodyTokenizer, notes_to_tokens, ordered
from improvnet.utils.melody_reference import crop_notes
from improvnet.utils.melody_chunks import chunk_end
from improvnet.utils.drum_profile import drum_profile, reference_kit, profile_report
from improvnet.utils.melody_stems import DRUM_KIT_ALIASES


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pin_checkpoint(source, directory):
    """Copy one opened checkpoint inode into an immutable, content-named file.

    Encoder best/latest files may be replaced while AR training is running.
    AR resume/inference always load this fixed snapshot instead.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        digest = hashlib.sha256()
        with Path(source).open('rb') as stream, tempfile.NamedTemporaryFile(dir=directory, prefix='.encoder-', delete=False) as output:
            temporary = Path(output.name)
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(block)
                output.write(block)
        destination = directory/('reference_encoder-'+digest.hexdigest()+'.pt')
        if not destination.exists():
            temporary.replace(destination)
        return str(destination.resolve())
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class FrozenReferenceEncoder:
    def __init__(self, path, device, *, expected_sha256=None, kind='style'):
        self.device = torch.device(device)
        self.batch_size=32
        self.path = str(Path(path).resolve())
        self.sha256 = file_digest(path)
        if expected_sha256 and self.sha256 != expected_sha256:
            raise ValueError('Reference encoder changed since AR training; use its pinned encoder checkpoint')
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        tok = MelodyTokenizer()
        digest = hashlib.sha256(json.dumps(tok.signature()).encode()).hexdigest()
        formats={'style':{'style_v1','melody_reference_encoder_v6'},'beat':{'beat_v1'}}
        if kind not in formats or state.get('format') not in formats[kind] or state.get('vocabulary_digest') != digest:
            raise ValueError(f'Expected a {kind} encoder with this exact tokenizer')
        model_class = StyleEncoder if kind == 'style' else BeatEncoder
        self.model = model_class(ContinuityConfig(**state['config']))
        self.model.load_state_dict(state['model_state_dict'])
        self.model.to(self.device).eval().requires_grad_(False)
        self.dim = state['config']['projection_dim']
        self.max_tokens = state['config']['max_length']
        self.step = state['trained_steps']

    @torch.no_grad()
    def encode(self, ids, attention_mask, present):
        ids, attention_mask = ids.to(self.device), attention_mask.to(self.device)
        present = present.to(self.device).bool()
        result = torch.zeros(ids.size(0), self.dim, device=self.device)
        if present.any():
            if not attention_mask[present].any(1).all():
                raise ValueError('A visible reference cannot contain only padding')
            context = (torch.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
                       if self.device.type == 'cuda' else nullcontext())
            with context:
                result[present] = self.model.encode(ids[present], attention_mask[present]).float()
        return result

    def from_notes(self,notes,tokenizer,rng,*,instruments):
        required=set(instruments)
        selected=[n for n in notes if n.instrument in required]
        if not required or {n.instrument for n in selected}!=required:raise ValueError('Reference is missing requested instruments')
        for _ in range(64):
            crop,start,end=crop_notes(selected,rng,self.max_tokens)
            if {n.instrument for n in crop}==required:break
        else:raise ValueError('No reference window contains all requested instruments')
        return dict(embedding=self.embed_notes(crop,tokenizer),start_ms=start,end_ms=end,
            instruments=sorted(required),tokens=len(notes_to_tokens(crop,tokenizer)),
            encoder_sha256=self.sha256,encoder_step=self.step,_notes=crop)

    def embed_notes(self,notes,tokenizer):
        ids=torch.tensor([[tokenizer.tok_to_id[t] for t in notes_to_tokens(notes,tokenizer)]],device=self.device)
        if ids.size(1)>self.max_tokens:raise ValueError('Reference exceeds the encoder token budget')
        return self.encode(ids,torch.ones_like(ids,dtype=torch.bool),torch.ones(1,device=self.device,dtype=torch.bool))[0]

    @torch.no_grad()
    def feature_windows(self,ids,mask,bins):
        ids,mask=ids.to(self.device),mask.to(self.device)
        features=torch.zeros(ids.size(0),bins+1,self.dim,device=self.device)
        valid=torch.zeros(ids.size(0),bins+1,device=self.device,dtype=torch.bool)
        tok=MelodyTokenizer()
        onset_values=torch.zeros(tok.vocab_size,device=self.device,dtype=torch.long)
        instrument=torch.zeros(tok.vocab_size,device=self.device,dtype=torch.bool)
        for token,index in tok.tok_to_id.items():
            if isinstance(token,tuple) and token[0]=='onset':onset_values[index]=token[1]
            if isinstance(token,tuple) and token[0]=='instrument':instrument[index]=True
        for offset in range(0,len(ids),self.batch_size):
            block,attention=ids[offset:offset+self.batch_size],mask[offset:offset+self.batch_size]
            present=attention.any(-1)
            if not present.any():continue
            block,attention=block[present],attention[present]
            context=(torch.autocast('cuda',dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
                     if self.device.type=='cuda' else nullcontext())
            with context:
                hidden=self.model.encode(block,attention,return_hidden=True)
                global_hidden=hidden.float().sum(1)/attention.sum(1,keepdim=True)
                note_mask=instrument[block]&attention
                onsets=onset_values[block.roll(-2,1)]+block.eq(tok.tok_to_id['<T>']).cumsum(1)*tok.abs_time_step_ms
                maximum=onsets.masked_fill(~note_mask,0).amax(1,keepdim=True).clamp_min(1)
                groups=(onsets*bins//(maximum+1)).clamp(0,bins-1)
                pooled=torch.zeros(block.size(0),bins,hidden.size(-1),device=self.device)
                counts=torch.zeros(block.size(0),bins,device=self.device)
                pooled.scatter_add_(1,groups[:,:,None].expand_as(hidden),hidden.float()*note_mask[:,:,None])
                counts.scatter_add_(1,groups,note_mask.float())
                local=pooled/counts.clamp_min(1)[:,:,None]
                all_hidden=torch.cat((global_hidden[:,None],local),1).to(hidden.dtype)
                encoded=torch.nn.functional.normalize(self.model.projector(all_hidden).float(),dim=-1)
            rows=present.nonzero().flatten()+offset
            features[rows]=encoded
            valid[rows,0]=True
            valid[rows,1:]=counts.gt(0)
        return features,valid


class FrozenConditioners:
    """Two immutable encoders; only the AR conditioning projections are trained."""
    def __init__(self,style_path,beat_path,device,*,style_sha256=None,beat_sha256=None,windows=2,local_bins=8,batch_size=32):
        if min(windows,local_bins,batch_size)<1:raise ValueError('Reference bank sizes must be positive')
        if windows!=2:raise ValueError('Conditioning uses a consecutive reference pair')
        self.local_bins=local_bins
        self.style=FrozenReferenceEncoder(style_path,device,expected_sha256=style_sha256,kind='style')
        self.beat=FrozenReferenceEncoder(beat_path,device,expected_sha256=beat_sha256,kind='beat')
        self.style.batch_size=self.beat.batch_size=batch_size

    def condition_batch(self,batch):
        features,mask,positions=self.feature_bank(batch['reference_style_ids'],batch['reference_attention_mask'],beat_ids=batch['reference_beat_ids'])
        batch.update(reference_features=features,reference_mask=mask,reference_positions=positions)
        return batch

    @torch.no_grad()
    def feature_bank(self,ids,mask,*,beat_ids=None):
        """Keep window axes separate so neither features nor statistics can mix."""
        batch,windows,length=ids.shape
        parts,masks,coordinates=[],[],[]
        for kind,encoder in enumerate((self.style,self.beat)):
            streams=beat_ids if kind==1 and beat_ids is not None else ids
            features,valid=encoder.feature_windows(streams.reshape(-1,length),mask.reshape(-1,length),self.local_bins)
            features=features.reshape(batch,windows,self.local_bins+1,encoder.dim)
            valid=valid.reshape(batch,windows,self.local_bins+1)
            parts.append(torch.cat((features[:,:,:1],features),2))
            masks.append(torch.cat((valid[:,:,:1],valid),2))
            pos=torch.zeros(batch,windows,self.local_bins+2,3,device=ids.device)
            pos[:,:,:,2]=kind
            pos[:,:,:,0]=(torch.arange(windows,device=ids.device)%2)[None,:,None]
            pos[:,:,:2,1]=-1
            pos[:,:,2:,1]=torch.linspace(0,1,self.local_bins,device=ids.device)
            coordinates.append(pos)
        return torch.cat(parts,2),torch.cat(masks,2),torch.cat(coordinates,2)

    def from_notes(self,notes,tokenizer,*,mode='trio',stage_instruments=None,aggregation='window',rng=None,tempo_bpm=None):
        """Global and local features from the same selected windows, cached by instrument view."""
        from improvnet.utils.melody_reference import reference_instruments
        original_notes = list(notes)
        notes = original_notes
        tempo_adjustment = None
        if tempo_bpm is not None:
            from improvnet.utils.melody_reference import retime_reference
            notes, tempo_adjustment = retime_reference(notes, tempo_bpm)
        if aggregation not in ('window','average'):raise ValueError('Reference aggregation must be window or average')
        rng=rng or random.Random(42)
        available={n.instrument for n in notes}
        stages=stage_instruments or {'reference':sorted(available)}
        cache,conditions={},{}
        for stage,target in stages.items():
            wanted=reference_instruments(target,'exact' if stage=='refine' else mode,available)
            key=tuple(sorted(wanted))
            if key not in cache:
                if not wanted or not wanted<=available:
                    raise ValueError(f'Reference lacks the required {stage} instruments: {sorted(wanted-available)}')
                selected=ordered([n for n in notes if n.instrument in wanted])
                streams,regions,instrument_sets,crops=[],[],[],[]
                begin=0
                while begin<len(selected):
                    end=chunk_end(selected,begin,self.style.max_tokens)
                    crop=selected[begin:end];origin=crop[0].onset
                    streams.append([tokenizer.tok_to_id[t] for t in notes_to_tokens([replace(n,onset=n.onset-origin) for n in crop],tokenizer)])
                    regions.append(dict(start_ms=origin,end_ms=max(n.onset+n.duration for n in crop)))
                    instrument_sets.append({n.instrument for n in crop})
                    crops.append(crop)
                    begin=end
                # Local excerpts must contain every requested instrument.
                eligible=[i for i,present in enumerate(instrument_sets) if wanted<=present]
                if not eligible:
                    raise ValueError(f'Reference has no complete {stage} excerpt containing {sorted(wanted)}')
                eligible=set(eligible)
                pairs=[(i,i+1) for i in range(len(streams)-1) if i in eligible and i+1 in eligible]
                if not pairs:
                    raise ValueError(f'Reference has no consecutive {stage} pair containing {sorted(wanted)} in both windows')
                chosen=[rng.choice(pairs)] if aggregation=='window' else pairs
                indices=[i for pair in chosen for i in pair]
                width=max(len(streams[i]) for i in indices)
                ids=torch.full((1,len(indices),width),tokenizer.pad_id,device=self.style.device,dtype=torch.long)
                for w,index in enumerate(indices):ids[0,w,:len(streams[index])]=torch.tensor(streams[index],device=ids.device)
                features,mask,positions=self.feature_bank(ids,ids.ne(tokenizer.pad_id))
                # Inference requests the standard kit. No training target is filtered.
                palette=reference_kit(selected)
                profiles=[]
                for index in indices:
                    profile_notes=[n for n in crops[index] if n.instrument!='drum' or DRUM_KIT_ALIASES.get(n.pitch,n.pitch) in palette]
                    profile_notes=[replace(n,pitch=DRUM_KIT_ALIASES.get(n.pitch,n.pitch)) if n.instrument=='drum' else n for n in profile_notes]
                    # The crop's own origin makes interval statistics window-local.
                    origin=crops[index][0].onset
                    profiles.append(drum_profile([replace(n,onset=n.onset-origin) for n in profile_notes]))
                profile=torch.stack(profiles).to(features.device).reshape(len(chosen),2,-1).mean(0)
                if aggregation=='average':
                    # Average matching positions in consecutive pairs, never merge
                    # first-window and second-window banks or their drum profiles.
                    features=features.reshape(1,len(chosen),2,*features.shape[2:])
                    mask=mask.reshape(1,len(chosen),2,-1)
                    features=(features*mask[...,None]).sum(1)/mask.sum(1).clamp_min(1)[...,None]
                    features=torch.nn.functional.normalize(features,dim=-1)
                    mask=mask.any(1)
                    positions=positions[:,:2]
                cache[key]=dict(reference_features=features[0],reference_mask=mask[0],reference_positions=positions[0],
                    drum_profile=profile,drum_palette=palette,drum_summary=[profile_report(p) for p in profile],
                    reference_region=dict(candidate_windows=len(streams),candidate_pairs=len(pairs),
                        selected_windows=[regions[i] for i in indices],aggregation=aggregation,
                        global_aggregation='independent_window_banks' if aggregation=='window' else 'normalized_mean_by_pair_position',
                        instruments=sorted(wanted)))
            conditions[stage]=dict(cache[key],reference_mode='exact' if stage=='refine' else mode,output_instruments=list(target))
        return dict(stage_conditions=conditions,aggregation=aggregation,reference_mode=mode,tempo_adjustment=tempo_adjustment,
                    _reference_notes=original_notes,_conditioning_notes=list(notes))
