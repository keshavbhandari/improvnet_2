"""Simultaneous, disjoint instrument views; never a solo-file fallback."""
from dataclasses import replace
import random
from bisect import bisect_left
from collections import defaultdict
import torch
from improvnet.model.encoders.data import PiecePairDataset, transpose_view, mask_view
from improvnet.utils.melody import Note, ordered, notes_to_tokens
from improvnet.utils.melody_reference import PIANOS
from improvnet.utils.melody_chunks import token_count
from improvnet.utils.melody_stems import kit_only_notes
from improvnet.metrics.continuity.data import PairCollator


def partitions(instruments, rng):
    """Full nonempty bipartition, including mixture-v-mixture combinations."""
    names = sorted(set(instruments))
    if len(names) < 2:
        raise ValueError('Beat training requires multiple audible instruments')
    rng.shuffle(names)
    split = rng.randint(1,len(names)-1)
    a, b = set(names[:split]), set(names[split:])
    piano = next((x for x in names if x in PIANOS),None)
    if piano and 'drum' in names and rng.random()<.5:
        # Half the partitions explicitly contrast piano and drums. Other draws
        # can put piano+drums together against the remaining instruments.
        a.discard(piano);b.discard(piano);a.discard('drum');b.discard('drum')
        a.add(piano);b.add('drum')
    return a,b


def aligned_window(notes, groups, rng, max_tokens=512, *, excluded=None, gap_ms=1000):
    notes=ordered(notes)
    onsets=sorted({n.onset for n in notes})
    by_onset=defaultdict(list)
    for n in notes:by_onset[n.onset].append(n)
    starts=list(onsets);rng.shuffle(starts)
    for start in starts[:128]:
        if excluded and not (start >= excluded[1]+gap_ms or start < excluded[0]-gap_ms):
            continue
        a,b=[],[]
        stop=None
        for onset_index in range(bisect_left(onsets,start),len(onsets)):
            onset=onsets[onset_index]
            if excluded and start<excluded[0] and onset>=excluded[0]-gap_ms:
                stop=excluded[0]-gap_ms;break
            aa=[n for n in by_onset[onset] if n.instrument in groups[0]]
            bb=[n for n in by_onset[onset] if n.instrument in groups[1]]
            count=lambda view:4*len(view)+(round((view[-1].onset-start)/10)*10)//5000+2 if view else 2
            if max(count(a+aa),count(b+bb))>max_tokens:
                stop=onset;break
            a.extend(aa);b.extend(bb)
        if len({n.onset for n in a})<2 or len({n.onset for n in b})<2:
            continue
        end=stop if stop is not None else max(n.onset+n.duration for n in a+b)
        if end<=start:continue
        # Shared origin and endpoint: offbeats survive; no independently rebased
        # drum view and no sounding target tail can enter a separate-region negative.
        views=tuple([replace(n,onset=n.onset-start,duration=min(n.duration,end-n.onset)) for n in view] for view in (a,b))
        return views,(start,end)
    raise ValueError('No shared two-view window fits complete onset groups')


def encode_views(views,rng,tokenizer,*,augment,max_semitones,mask_prob,mask_rate_range):
    a,b=views
    shifts=(0,0)
    if augment:
        original_a,original_b=a,b
        a,sa=transpose_view(a,rng,max_semitones)
        b,sb=transpose_view(b,rng,max_semitones)
        # At least one nonzero transposition where MIDI bounds allow it, and
        # prefer differing view keys. Drum numbers are never transposed.
        if sa==sb:
            for index,view in enumerate((original_a,original_b)):
                pitches=[n.pitch for n in view if n.instrument!='drum']
                if not pitches:continue
                choices=[s for s in range(max(-max_semitones,-min(pitches)),min(max_semitones,127-max(pitches))+1) if s!=(sb if index==0 else sa) and s!=0]
                if choices:
                    shift=rng.choice(choices)
                    changed=[replace(n,pitch=n.pitch+shift) if n.instrument!='drum' else n for n in view]
                    if index==0:a,sa=changed,shift
                    else:b,sb=changed,shift
                    break
        shifts=sa,sb
    ids=[]
    for view in (a,b):
        tokens=[tokenizer.tok_to_id[t] for t in notes_to_tokens(view,tokenizer)]
        if augment:tokens=mask_view(tokens,tokenizer,rng,probability=mask_prob,rate_range=mask_rate_range)
        ids.append(torch.tensor(tokens,dtype=torch.long))
    return ids,shifts


class BeatPairDataset(PiecePairDataset):
    def __init__(self,*args,negative_gap_ms=1000,**kwargs):
        super().__init__(*args,**kwargs)
        self.rows=[r for r in self.rows if len(set(r.get('instruments',[])))>=2 and set(r.get('instruments',[]))-{'drum'}]
        if len(self.rows)<2:raise ValueError('Need at least two multi-instrument pieces in this split')
        self.piano_drums=[i for i,r in enumerate(self.rows) if 'drum' in r.get('instruments',[]) and PIANOS.intersection(r.get('instruments',[]))]
        self.jazz_piano_drums=[i for i in self.piano_drums if self.rows[i]['genre'] in ('jazz','blues')]
        self.negative_gap_ms=negative_gap_ms
        self.fingerprint+=':beat-disjoint-windows-v1'

    def load_notes(self,row):
        rec=self.base.load_record(row)
        return ordered([Note('Acoustic Piano' if n[0]=='piano' else n[0],*n[1:]) for n in rec['notes']]+kit_only_notes([Note(*n) for n in rec.get('drums',[])]))

    @staticmethod
    def identity(row):
        import hashlib
        group=row.get('group') or row.get('source_group') or row.get('sha256') or row.get('path') or str(row['offset'])
        return int.from_bytes(hashlib.sha256(str(group).encode()).digest()[:8],'little')%(2**63-1)

    @staticmethod
    def record_identity(row):
        import hashlib
        identity=row.get('sha256') or row.get('raw_path') or row.get('path') or (str(row.get('jsonl'))+':'+str(row.get('offset')))
        return int.from_bytes(hashlib.sha256(str(identity).encode()).digest()[:8],'little')%(2**63-1)

    def __getitem__(self,key):
        index,draw=key if isinstance(key,tuple) else (key,key)
        rng=random.Random(self.seed+(self.epoch if self.split=='train' else 0)*1000000007+draw*9176+index)
        if self.split=='train' and self.piano_drums and rng.random()<self.piano_drum_draw_prob:
            pool=self.jazz_piano_drums if self.jazz_piano_drums and rng.random()<self.jazz_piano_drum_prob else self.piano_drums
            index=rng.choice(pool)
        errors=[]
        for attempt in range(64):
            try:
                row=self.rows[index];notes=self.load_notes(row)
                groups=partitions({n.instrument for n in notes},rng)
                views,bounds=aligned_window(notes,groups,rng,self.max_tokens)
                piece_id=self.identity(row)
                negative_id=piece_id
                record_id=self.record_identity(row);negative_record_id=record_id
                try:
                    negative,negative_bounds=aligned_window(notes,groups,rng,self.max_tokens,excluded=bounds,gap_ms=self.negative_gap_ms)
                except ValueError:
                    # Short pieces still contribute positives. Their explicit
                    # negative comes from a different canonical piece in this fold.
                    for _ in range(32):
                        other=self.rows[rng.randrange(len(self.rows))]
                        if self.identity(other)==piece_id:continue
                        try:
                            other_notes=self.load_notes(other)
                            other_groups=partitions({n.instrument for n in other_notes},rng)
                            negative,negative_bounds=aligned_window(other_notes,other_groups,rng,self.max_tokens)
                            negative_id=self.identity(other);negative_record_id=self.record_identity(other);break
                        except (ValueError,OSError,KeyError):continue
                    else:raise ValueError('No different-piece negative available')
                options=dict(augment=self.augment,max_semitones=self.pitch_range,mask_prob=self.mask_prob,mask_rate_range=self.mask_rate_range)
                (a,b),shifts=encode_views(views,rng,self.tokenizer,**options)
                (na,nb),nshifts=encode_views(negative,rng,self.tokenizer,**options)
                return dict(a=a,b=b,negative_a=na,negative_b=nb,piece_id=piece_id,negative_piece_id=negative_id,record_id=record_id,negative_record_id=negative_record_id,
                    bounds=bounds,negative_bounds=negative_bounds,shifts=shifts,negative_shifts=nshifts,
                    a_instruments=sorted({n.instrument for n in views[0]}),b_instruments=sorted({n.instrument for n in views[1]}),attempts=attempt+1)
            except (ValueError,OSError,KeyError) as error:
                errors.append(str(error)[:120]);index=rng.randrange(len(self.rows))
        raise RuntimeError(f'Could not sample a multitrack beat pair in {self.split}: {errors[-3:]}')


class BeatCollator(PairCollator):
    def __call__(self,rows):
        batch=super().__call__(rows)
        negatives=super().__call__([dict(a=r['negative_a'],b=r['negative_b'],piece_id=r['negative_piece_id']) for r in rows])
        batch.update(negative_a=negatives['a'],negative_b=negatives['b'],negative_a_mask=negatives['a_mask'],negative_b_mask=negatives['b_mask'],
            negative_piece_ids=negatives['piece_ids'],record_ids=torch.tensor([r['record_id'] for r in rows]),negative_record_ids=torch.tensor([r['negative_record_id'] for r in rows]),bounds=torch.tensor([r['bounds'] for r in rows]),negative_bounds=torch.tensor([r['negative_bounds'] for r in rows]))
        return batch
