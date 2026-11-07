"""Frozen encoder used identically by AR training and MIDI inference."""
import hashlib
import json
from pathlib import Path
from contextlib import nullcontext
import tempfile
import torch
from improvnet.model.encoders.model import ContinuityConfig
from improvnet.model.encoders.style.model import StyleEncoder
from improvnet.model.encoders.beat.model import BeatEncoder
from improvnet.utils.melody import MelodyTokenizer, notes_to_tokens
from improvnet.utils.melody_reference import PIANOS, crop_notes


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

    def condition_batch(self, batch):
        batch['style_embedding'] = self.encode(batch['style_ids'], batch['style_attention_mask'], batch['style_present'])
        batch['style_mask'] = batch['style_present'].float()
        return batch

    def from_notes(self, notes, tokenizer, rng):
        # Retain polyphonic piano and kit drums; never skyline a reference.
        pianos = {n.instrument for n in notes if n.instrument in PIANOS}
        piano_instrument = ('Acoustic Piano' if 'Acoustic Piano' in pianos else sorted(pianos)[0] if pianos else None)
        selected = [n for n in notes if n.instrument in (piano_instrument, 'drum')]
        if not any(n.instrument != 'drum' for n in selected):
            raise ValueError('Reference MIDI needs a supported piano track (drums are optional)')
        # Prefer a window where both piano and drums actually play.
        candidates = []
        for _ in range(24):
            crop, start, end = crop_notes(selected, rng, self.max_tokens)
            if any(n.instrument in PIANOS for n in crop):
                candidates.append((crop, start, end))
            if any(n.instrument == 'drum' for n in crop) and any(n.instrument in PIANOS for n in crop):
                break
        if not candidates:
            candidates.append(crop_notes([n for n in selected if n.instrument in PIANOS], rng, self.max_tokens))
        crop, start, end = candidates[-1]
        ids = torch.tensor([[tokenizer.tok_to_id[t] for t in notes_to_tokens(crop, tokenizer)]], device=self.device)
        mixture = self.encode(ids, torch.ones_like(ids, dtype=torch.bool), torch.tensor([True], device=self.device))[0]
        piano = [n for n in crop if n.instrument in PIANOS]
        # Exactly the same interval/rebasing as the full piano+drum view.
        solo_ids = torch.tensor([[tokenizer.tok_to_id[t] for t in notes_to_tokens(piano, tokenizer)]], device=self.device)
        solo = self.encode(solo_ids, torch.ones_like(solo_ids, dtype=torch.bool), torch.tensor([True], device=self.device))[0]
        has_drums = any(n.instrument == 'drum' for n in crop)
        return dict(solo_embedding=solo, mixture_embedding=mixture if has_drums else None,
                    start_ms=start, end_ms=end, instruments=sorted({n.instrument for n in crop}),
                    tokens=ids.size(1), solo_tokens=solo_ids.size(1),
                    encoder_sha256=self.sha256, encoder_step=self.step, _notes=crop)


class FrozenConditioners:
    """Two immutable encoders; only the AR conditioning projections are trained."""
    def __init__(self,style_path,beat_path,device,*,style_sha256=None,beat_sha256=None):
        self.style=FrozenReferenceEncoder(style_path,device,expected_sha256=style_sha256,kind='style')
        self.beat=FrozenReferenceEncoder(beat_path,device,expected_sha256=beat_sha256,kind='beat')

    def condition_batch(self,batch):
        for name,encoder in (('style',self.style),('beat',self.beat)):
            batch[name+'_embedding']=encoder.encode(batch[name+'_ids'],batch[name+'_attention_mask'],batch[name+'_present'])
            batch[name+'_mask']=batch[name+'_present'].float()
        return batch

    def from_notes(self,notes,tokenizer,rng,regions='same'):
        from improvnet.utils.melody_reference import neighboring_reference,PIANOS
        if regions not in ('same','different'):raise ValueError('Reference regions must be same or different')
        style=self.style.from_notes(notes,tokenizer,rng)
        style_notes=style.pop('_notes')
        beat_notes=style_notes
        beat_start,beat_end=style['start_ms'],style['end_ms']
        if regions=='different':
            selected=[n for n in notes if n.instrument in set(style['instruments'])|{'drum'}]
            maximum=max(n.onset+n.duration for n in selected)+1
            neighbor=neighboring_reference(selected,beat_start,beat_end,rng,max_tokens=self.beat.max_tokens,
                guard_ms=250,max_distance_ms=maximum)
            if neighbor is None:raise ValueError('Reference is too short for different disjoint regions; use --reference-regions same or a longer MIDI')
            beat_notes=neighbor['notes'];beat_start,beat_end=neighbor['start_ms'],neighbor['end_ms']
        ids=torch.tensor([[tokenizer.tok_to_id[t] for t in notes_to_tokens(beat_notes,tokenizer)]],device=self.beat.device)
        beat_embedding=self.beat.encode(ids,torch.ones_like(ids,dtype=torch.bool),torch.tensor([True],device=self.beat.device))[0]
        return dict(style_solo_embedding=style.pop('solo_embedding'),style_mixture_embedding=style.pop('mixture_embedding'),
            beat_embedding=beat_embedding,style_region=style,
            beat_region=dict(start_ms=beat_start,end_ms=beat_end,tokens=ids.size(1),instruments=sorted({n.instrument for n in beat_notes}),
                encoder_sha256=self.beat.sha256,encoder_step=self.beat.step),regions=regions)
