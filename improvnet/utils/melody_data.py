"""Three-task examples over lazy performance MIDI and original AR mixtures."""
from bisect import bisect_left
from collections import Counter, OrderedDict
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from improvnet.model import melody_config as cfg
from improvnet.scripts.melody_prepare import make_record, read_job, melody_fingerprint
from improvnet.utils.ar_utils import ProcessData
from improvnet.utils.melody import (Note, MelodyTokenizer, TASKS, make_prompt, mask_timing,
    melody_voice, notes_to_tokens, ordered, sample_strength, harmonize_strength, skyline, tokens_to_notes, window)
from improvnet.utils.melody_chunks import (chunk_bounds, chunk_end, chunk_starts,
                                          history_tokens, stem_bounds, shorten_chunk, validate_lengths)
from improvnet.utils.melody_corruptions import corrupt_history, corrupt_stem_source, sample_stem_pitch_probability, refinement_source, refinement_strength
from improvnet.utils.melody_reference import neighboring_reference, reference_ids
from improvnet.utils.melody_stems import choose_missing_stem, instrument_weight, DRUM_KIT_PITCHES, DRUM_KIT_ALIASES, kit_only_notes


DATA_STAT_NAMES=('attempts','retries','routed','no_pitched','malformed','protected','token_budget','other')


def retry_reason(error):
    text=str(error).lower()
    if 'protected' in text:
        return 'protected'
    if 'no pitched' in text:
        return 'no_pitched'
    if 'token budget' in text or 'onset group' in text:
        return 'token_budget'
    if any(key in text for key in ('data byte','keysignature','mthd','eoferror','malformed','track boundary')):
        return 'malformed'
    return 'other'


class NoStemWindow(ValueError):
    """No meaningful multi-instrument stem target window is available."""


def stem_source(full_mix,rng,config,target_instrument=None):
    """A complete retained mixture paired with the original mix, never labels."""
    instruments=sorted({n.instrument for n in full_mix})
    if len(instruments)<2 or not set(instruments)-{'drum'}:
        raise NoStemWindow('Stem addition requires at least two audible instruments')
    candidates=instruments.copy()
    if target_instrument is not None:
        if target_instrument not in candidates:
            raise NoStemWindow('Requested stem is absent from the clean mixture')
        candidates=[target_instrument]
    full_onsets=[n.onset for n in full_mix]
    while candidates:
        selected=choose_missing_stem(candidates,rng,config.STEM_DRUM_PROB,config.JAZZ_INSTRUMENT_WEIGHT)
        source=[n for n in full_mix if n.instrument!=selected]
        if not any(n.instrument!='drum' for n in source):
            candidates.remove(selected);continue
        attacks=[n.onset for n in full_mix if n.instrument==selected]
        valid=[]
        for begin in chunk_starts(source,config.CHUNK_TOKENS):
            stop=chunk_end(source,begin,config.CHUNK_TOKENS)
            start,end=stem_bounds(source,begin,stop)
            target=full_mix[bisect_left(full_onsets,start):bisect_left(full_onsets,end)]
            if bisect_left(attacks,end)>bisect_left(attacks,start) and len({n.instrument for n in target})>=2:
                valid.append(begin)
        if valid:
            return selected,source,valid
        candidates.remove(selected)
    raise NoStemWindow('No overlapping missing-stem window')


def ar_record(entry):
    """Recover audible notes lazily from the exact AR tokenization.

    Legacy drum entries sometimes omit velocity/duration. Those pieces must be
    read from their resolved source MIDI, not interpreted as silent hits.
    """
    tokens = ProcessData().serialized_tokens_to_tokens(entry['tokens'])
    notes,segment,i = [],0,0
    while i<len(tokens):
        t=tokens[i]
        if t=='<T>':
            segment+=5000
        elif isinstance(t,tuple) and t[0]=='instrument':
            group=tokens[i:i+4]
            if len(group)!=4 or [x[0] if isinstance(x,tuple) else None for x in group]!=['instrument','note','onset','dur']:
                raise ValueError('Malformed AR note attributes')
            inst,pv,onset,duration=group
            if inst[1]=='drum' and (pv[2]<=0 or duration[1]<=0):
                raise ValueError('Legacy AR drum attributes unavailable; require source MIDI')
            if pv[2]>0 and duration[1]>0:
                notes.append(Note(inst[1],pv[1],pv[2],segment+onset[1],duration[1]))
            i+=3
        i+=1
    record,fingerprint=make_record((Path(entry['midi_filepath']),'classical','ar_tokens','',False),ordered(notes))
    record['genre']=entry.get('genre') or 'unknown'
    return record,fingerprint


def prepare_example(record, rng, config=cfg, task=None, strength=None, target_instrument=None):
    tok = MelodyTokenizer()
    task = task or rng.choices(TASKS, config.TASK_WEIGHTS)[0]
    if task not in TASKS:
        raise ValueError(f'Unknown v5 task: {task}')
    notes = ordered([Note('Acoustic Piano' if n[0] == 'piano' else n[0], *n[1:]) for n in record['notes']])
    drums = kit_only_notes([Note(*n) for n in record.get('drums', [])])
    nonkit_drums = sum(n[1] not in DRUM_KIT_PITCHES and n[1] not in DRUM_KIT_ALIASES for n in record.get('drums', []))
    if not notes:
        raise ValueError('No pitched notes')
    melody = melody_voice(notes if record['source'] in ('weimar', 'wikifonia', 'curated') else
                          skyline(notes, config.SKYLINE_TOLERANCE_MS)[0])
    if record.get('_augment'):
        shift = max(-min(n.pitch for n in notes), min(127 - max(n.pitch for n in notes), rng.randint(-5, 5)))
        shifted = {n: replace(n, pitch=n.pitch + shift) for n in notes}
        melody = ordered([shifted[n] for n in melody])
        notes = [shifted[n] for n in notes]
    full_mix = ordered(notes + drums)
    domain = melody if task in ('refine', 'harmonize') else notes
    target_domain = melody if task == 'refine' else notes
    target_instrument = target_instrument if task == 'stem' else None
    if task == 'stem':
        target_instrument,domain,starts=stem_source(full_mix,rng,config,target_instrument)
        target_domain=full_mix
    else:
        starts=chunk_starts(domain,config.CHUNK_TOKENS)
    strength = (harmonize_strength(rng, strength, config.HARMONIZE_MIN_MASK_RATIO, config.HARMONIZE_FULL_MASK_PROB)
                if task == 'harmonize' else sample_strength(rng) if strength is None else strength)
    if task == 'stem':
        strength=1.
    use_history = rng.random() < (config.HISTORY_PROB if task == 'refine' else config.POLY_HISTORY_PROB)
    if use_history:
        noninitial = [i for i in starts if i > 0]
        if not noninitial and task != 'stem':
            noninitial = [i for i in range(1, len(domain)) if domain[i].onset > domain[i - 1].onset]
        starts = noninitial or starts
    begin = rng.choice(starts)
    start = min(n.onset for n in target_domain) if task == 'harmonize' and begin == 0 else domain[begin].onset
    left_notes = [n for n in target_domain if n.onset < start]
    left = history_tokens(left_notes, config.HISTORY_TOKENS, tok) if use_history else []
    history_corrupted = bool(task != 'refine' and record.get('_augment') and left
                             and rng.random() < config.HISTORY_CORRUPTION_PROB)
    if history_corrupted:
        hist = tokens_to_notes(left, tok)
        origin = min(n.onset for n in left_notes[-len(hist):])
        hist = corrupt_history(hist, rng, config.HISTORY_EDIT_FRACTION,
                               config.HISTORY_ADDITION_WEIGHT, config.HISTORY_DELETION_WEIGHT,
                               end_ms=start - origin, pitch_semitones=config.PITCH_CORRUPTION_SEMITONES,
                               timing_ms=config.TIMING_CORRUPTION_MS, time_step_ms=tok.time_step_ms)
        left = history_tokens(hist, config.HISTORY_TOKENS, tok)
    stop = chunk_end(domain, begin, config.CHUNK_TOKENS)
    stem_pitch_probability = sample_stem_pitch_probability(rng, config) if task == 'stem' else None
    state, bounded = rng.getstate(), False
    while True:
        rng.setstate(state)
        bounds=stem_bounds if task=='stem' else chunk_bounds
        start, end = bounds(domain, begin, stop)
        if task == 'harmonize':
            if begin == 0:
                start = min(n.onset for n in notes)
            if stop == len(domain):
                end = max(n.onset + n.duration for n in notes)
        core = window(domain, start, end)
        target_notes = window(target_domain, start, end)
        condition_strength = refinement_strength(core, strength, config) if task == 'refine' else strength
        if task == 'refine':
            source = refinement_source(core, strength, rng, tok, config, bounded)
        elif task == 'harmonize':
            source = mask_timing(core, strength, rng, tok, round_up=True)
        elif task == 'stem':
            if not any(n.instrument==target_instrument for n in target_notes) or len({n.instrument for n in target_notes})<2:
                raise NoStemWindow('Shrunk target has no requested missing-stem attacks')
            source=notes_to_tokens(corrupt_stem_source(core,rng,tok,config,end-start,pitch_probability=stem_pitch_probability),tok)
        target = notes_to_tokens(target_notes, tok)[1:]
        prompt = make_prompt(task, source, left, target_instrument)
        cap = config.CHUNK_TOKENS if task=='refine' else config.MIX_RESPONSE_TOKENS
        if len(source) <= config.CHUNK_TOKENS and len(target) + 1 <= cap and len(prompt) + len(target) + 2 <= config.SEQ_LEN:
            break
        available = config.SEQ_LEN - len(left) - 6 - 2*int(target_instrument is not None)
        fraction = min(config.CHUNK_TOKENS / len(source), cap / (len(target) + 1), available / (len(source) + len(target)))
        try:
            stop = shorten_chunk(domain, begin, stop, .98 * fraction)
        except ValueError:
            if task != 'refine' or bounded or len(target) + 1 > cap:
                raise ValueError('Complete onset group exceeds the training token budget')
            bounded = True
    reference = None
    beat_reference = None
    if task in ('harmonize', 'stem') and rng.random() < config.REFERENCE_PROB:
        # Exclude all sounds owned by this target, including releases after end.
        protected_end = max([end]+[n.onset+n.duration for n in full_mix if start<=n.onset<end])
        reference = neighboring_reference(full_mix, start, protected_end, rng,
            max_tokens=config.REFERENCE_TOKENS, guard_ms=config.REFERENCE_GUARD_MS,
            max_distance_ms=config.REFERENCE_MAX_DISTANCE_MS,
            instruments={n.instrument for n in target_notes})
    if task in ('harmonize','stem') and rng.random()<config.REFERENCE_PROB:
        protected_end=max([end]+[n.onset+n.duration for n in full_mix if start<=n.onset<end])
        beat_reference=neighboring_reference(full_mix,start,protected_end,rng,
            max_tokens=config.REFERENCE_TOKENS,guard_ms=config.REFERENCE_GUARD_MS,
            max_distance_ms=config.REFERENCE_MAX_DISTANCE_MS)
    beat_ids=reference_ids(beat_reference,tok)
    ref_ids = reference_ids(reference, tok)
    ids = [tok.tok_to_id[t] for t in prompt + target]
    labels = [config.IGNORE_INDEX] * len(prompt) + [tok.tok_to_id[t] for t in target]
    stem_mask, current = [False] * len(prompt), None
    for t in target:
        instrument = isinstance(t, tuple) and t[0] == 'instrument'
        if instrument:
            current = t[1]
        stem_mask.append(isinstance(t, tuple) and target_instrument is not None and current == target_instrument)
    changed_fraction = None
    if task == 'refine':
        from collections import Counter
        clean = Counter(tokens_to_notes(target, tok))
        changed = Counter(tokens_to_notes(source, tok))
        changed_fraction = 1 - sum((clean & changed).values()) / max(1, sum(clean.values()))
    genre = record['genre'] if record['genre'] in config.GENRES else 'unknown'
    exposure=torch.zeros(len(TASKS)*2+1,dtype=torch.long)
    exposure[TASKS.index(task)*2+int(genre in ('jazz','blues'))]=1
    exposure[-1]=int(target_instrument=='drum')
    return dict(input=torch.tensor(ids[:-1]), target=torch.tensor(labels[1:]),
                genre=torch.tensor(config.GENRES.index(genre)),
                multi_hot=ProcessData().get_instrument_multihot(notes_to_tokens(target_notes, tok)),
                strength=torch.tensor(condition_strength, dtype=torch.float32), requested_strength=strength, task=TASKS.index(task),
                beat_ids=torch.tensor(beat_ids,dtype=torch.long),beat_present=torch.tensor(bool(beat_ids)),
                beat_bounds=None if beat_reference is None else (beat_reference['start_ms'],beat_reference['end_ms']),
                style_ids=torch.tensor(ref_ids, dtype=torch.long),
                style_present=torch.tensor(bool(ref_ids)),
                style_bounds=None if reference is None else (reference['start_ms'], reference['end_ms']),
                style_view=None if reference is None else reference['view'], exposure=exposure, history_length=len(left), history_corrupted=history_corrupted,
                input_mono=task in ('refine', 'harmonize'), history_mono=task == 'refine',
                refinement_changed_fraction=changed_fraction,
                nonkit_drum_notes_filtered=nonkit_drums,
                drum_articulations_normalized=sum(n[1] in DRUM_KIT_ALIASES for n in record.get('drums', [])),
                drum_attacks_deduplicated=len(record.get('drums', []))-nonkit_drums-len(drums), timing_mask_fraction=source.count('<MASK>')/(2*len(core)) if task == 'harmonize' else None, stem_pitch_error_rate=stem_pitch_probability,
                source_length=len(source), response_length=len(target), start_ms=start, end_ms=end,
                stem_mask=torch.tensor(stem_mask[1:], dtype=torch.bool),
                stem_instrument=torch.tensor(ProcessData().INSTRUMENT_CLASSES.index(target_instrument) if target_instrument else -1),
                stem_name=target_instrument)


class MelodyDataset(Dataset):
    def __init__(self, data_file, split, config=cfg):
        validate_lengths(config)
        self.config, self.split, self.epoch = config, split, 0
        self.manifest = json.loads(Path(data_file).read_text())
        if (self.manifest['version'] != 6 or self.manifest['ratio'] != config.CLASSICAL_RATIO
                or self.manifest['seed'] != config.DATA_SPLIT_SEED):
            raise ValueError('Lazy manifest does not match the current configuration')
        self.prepared = Path(self.manifest['prepared'])
        raw_index = self.prepared.with_suffix('.index.json').read_bytes()
        if hashlib.sha256(raw_index).hexdigest() != self.manifest['prepared_index_sha256']:
            raise ValueError('Prepared split index changed after lazy manifest creation')
        all_rows=json.loads(raw_index)['rows']+self.manifest.get('prepared_extra_rows',[])
        overrides=self.manifest.get('prepared_split_overrides',{})
        rows=[]
        for r in all_rows:
            effective=overrides.get(str(r.get('jsonl',''))+':'+str(r['offset']),r['split'])
            if effective==split:
                group=r.get('group') or r.get('source_group') or r['sha256']
                canonical=self.manifest.get('prepared_group_aliases',{}).get(group,group)
                rows.append(dict(r,kind='prepared',split=effective,group=canonical))
        self.eval_rows=rows if split!='train' else []
        self.jazz_blues = [r for r in rows if r['genre'] in ('jazz', 'blues')]
        self.other = [r for r in rows if r['genre'] not in ('jazz', 'blues')]
        if split == 'train':
            self.other += [dict(r, kind='raw') for r in self.manifest['raw_train']]
        if not self.jazz_blues or not self.other:
            raise ValueError(f'No jazz/blues or non-jazz supervision for {split}')
        self.protected_hashes = set(self.manifest['protected_hashes'])
        self.protected_fingerprints = set(self.manifest['protected_melody_fingerprints'])
        self.cache, self.rejected = OrderedDict(), set()
        self.retry_counts,self.routed_counts,self.samples_seen=Counter(),Counter(),0
        self.ar_mixtures = self.manifest.get('ar_mixtures',[]) if split=='train' else []
        self.stem_pools = {b: [r for r in pool if self.stem_candidate(r)]
                                   for b, pool in ((True, self.jazz_blues), (False, self.other))}
        self.task_pools={}
        for bucket,pool in list(self.stem_pools.items())+[('ar',self.ar_mixtures)]:
            self.task_pools[bucket,'drum']=[r for r in pool if 'drum' in r.get('instruments',[])]
        self.set_epoch(0)

    @staticmethod
    def stem_candidate(row):
        instruments=set(row.get('instruments',[]))
        return bool(instruments-{'drum'} and len(instruments)>=2
                    or row['kind'] == 'raw' and row.get('mixture_hint'))

    def main_slots(self):
        return len(self.jazz_blues)*(self.config.CLASSICAL_RATIO+self.config.JAZZ_BLUES_REPEATS)

    def __len__(self):
        if self.split!='train':
            return len(self.eval_rows)
        fraction=sum(self.config.TASK_WEIGHTS[:2])/sum(self.config.TASK_WEIGHTS)
        # Full file traversals for the two universally eligible tasks. Separate
        # mixture draws preserve these traversals and the requested task weights.
        return math.ceil(self.main_slots()/fraction) if fraction else self.main_slots()


    def set_epoch(self, epoch):
        self.epoch = epoch
        rng = random.Random(self.config.DATA_SPLIT_SEED + (epoch * 9176 if self.split == 'train' else 0))
        # One traversal covers every jazz/blues row and 8x as many non-jazz
        # candidate slots. Rotate a wider pool each epoch without changing splits.
        self.other_order = list(range(len(self.other)))
        rng.shuffle(self.other_order)
        if self.split == 'train':
            # Prefer verified mixtures and orchestral/band-source candidates,
            # without repeating a small pool to fill the enlarged epoch.
            preferred = [i for i in self.other_order if self.other[i].get('has_multi_instrument')
                         or self.other[i].get('mixture_hint')]
            rest = [i for i in self.other_order if not (self.other[i].get('has_multi_instrument')
                    or self.other[i].get('mixture_hint'))]
            wanted = self.config.CLASSICAL_RATIO * len(self.jazz_blues)
            take = min(len(preferred), round(wanted * self.config.MULTI_INSTRUMENT_PROB))
            take = max(take, min(len(preferred), wanted - len(rest)))
            first = preferred[:take] + rest[:max(0, wanted - take)]
            rng.shuffle(first)
            self.other_order = first + preferred[take:] + rest[max(0, wanted - take):]

    def load_record(self, row):
        key=(row['kind'],row.get('jsonl'),row.get('offset',row.get('path')))
        if key in self.cache or key in self.rejected:
            return self._load_record(row)
        source=Path(row.get('raw_path') or (row['jsonl'] if row['kind']=='ar' else
                    str(row.get('jsonl') or self.prepared) if row['kind']=='prepared' else row['path']))
        identity=str(source)+(':'+str(row['offset']) if row['kind'] in ('ar','prepared') and not row.get('raw_path') else '')
        digest=hashlib.sha256(identity.encode()).hexdigest()
        failed=Path(self.config.PARSED_CACHE)/'rejected_v2'/digest[:2]/(digest+'.json')
        try:
            previous=json.loads(failed.read_text())
            stat=source.stat()
            if previous.get('signature')==[stat.st_size,stat.st_mtime_ns]:
                self.rejected.add(key)
                self.retry_counts['cached_parse_failures']+=1
                raise ValueError('Cached unusable source: '+previous['reason'])
        except (FileNotFoundError,json.JSONDecodeError):
            pass
        try:
            return self._load_record(row)
        except (ValueError,OSError,json.JSONDecodeError) as exc:
            self.rejected.add(key)
            # Copy protection depends on this run's protected set, not just the
            # source file. Keep checking it against the positive parse cache.
            permanent=any(reason in str(exc) for reason in (
                'No pitched notes','Legacy AR drum attributes unavailable',
                'data byte must be in range','KeySignatureError','MThd not found',
                'EOFError','Malformed AR note attributes'))
            if permanent and 'protected' not in str(exc).lower():
                try:
                    stat=source.stat()
                    failed.parent.mkdir(parents=True,exist_ok=True)
                    temporary=failed.with_suffix(f'.{os.getpid()}.tmp')
                    temporary.write_text(json.dumps(dict(version=2,source=str(source),
                        signature=[stat.st_size,stat.st_mtime_ns],reason=str(exc)[:240])))
                    temporary.replace(failed)
                except OSError:
                    pass
            raise

    def _load_record(self, row):
        key = (row['kind'],row.get('jsonl'),row.get('offset',row.get('path')))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if key in self.rejected:
            raise ValueError('Previously rejected candidate')
        if row['kind'] == 'prepared':
            with Path(row.get('jsonl') or self.prepared).open('rb') as f:
                f.seek(row['offset'])
                record = json.loads(f.readline())
        elif row['kind']=='ar' and not row.get('raw_path'):
            with Path(row['jsonl']).open('rb') as f:
                f.seek(row['offset'])
                entry=json.loads(f.readline())
            record,fingerprint=ar_record(entry)
            if fingerprint in self.protected_fingerprints:
                self.rejected.add(key)
                raise ValueError('AR training record is a copy of protected music')
        else:
            source_path=row.get('raw_path') or row['path']
            digest = hashlib.sha256(source_path.encode()).hexdigest()
            cache = Path(self.config.PARSED_CACHE) / digest[:2] / (digest + '.json')
            try:
                payload = json.loads(cache.read_text())
            except FileNotFoundError:
                job, sha, notes, error = read_job((Path(source_path), 'classical' if row['genre'] == 'classical' else 'jazz',
                                                  row['source'], '', False))
                if error:
                    self.rejected.add(key)
                    raise ValueError(error)
                record, fingerprint = make_record(job, notes)
                record['genre'] = row['genre']
                record['notes'] = [list(n) for n in record['notes']]
                record['drums'] = [list(n) for n in record['drums']]
                payload = dict(version=5, sha256=sha, fingerprint=fingerprint, record=record)
                cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_suffix(f'.{os.getpid()}.tmp')
                temporary.write_text(json.dumps(payload, separators=(',', ':')))
                temporary.replace(cache)
            if payload.get('version') != 5:
                raise ValueError('Invalid lazy parsed record cache')
            if payload['sha256'] in self.protected_hashes or payload['fingerprint'] in self.protected_fingerprints:
                self.rejected.add(key)
                raise ValueError('Candidate is a copy of protected jazz/blues or held-out music')
            record = dict(payload['record'],genre=row['genre'],source=row['source'])
        self._cache_record(key,record)
        return record

    def _cache_record(self,key,record):
        """Bound retained notes as well as files; never reject a large piece."""
        limit=self.config.WORKER_RECORD_CACHE
        budget=getattr(self.config,'WORKER_CACHE_NOTES',100000)
        size=len(record['notes'])+len(record.get('drums',[]))
        if limit<=0 or size>budget:
            return
        self.cache[key]=record
        self.cache.move_to_end(key)
        while self.cache and (len(self.cache)>limit or sum(
                len(r['notes'])+len(r.get('drums',[])) for r in self.cache.values())>budget):
            self.cache.popitem(last=False)

    def __getitem__(self, idx):
        rng = random.Random(self.config.DATA_SPLIT_SEED + idx * 1000003 + (self.epoch * 9176 if self.split == 'train' else 0))
        ratio,repeats=self.config.CLASSICAL_RATIO,self.config.JAZZ_BLUES_REPEATS
        weights=self.config.TASK_WEIGHTS
        if self.split=='train':
            fraction=sum(weights[:2])/sum(weights)
            before=math.floor(idx*fraction+1e-9)
            is_main=math.floor((idx+1)*fraction+1e-9)>before
            if is_main:
                position=before%(self.main_slots())
                block,within=divmod(position,ratio+repeats)
                jazz_blues=within<repeats
                pool=self.jazz_blues if jazz_blues else self.other
                slot=block if jazz_blues else block*ratio+within-repeats
                first=pool[slot if jazz_blues else self.other_order[slot%len(self.other_order)]]
                task=rng.choices(TASKS[:2],weights[:2])[0]
            else:
                auxiliary=idx-before
                jazz_blues=auxiliary%(ratio+repeats)<repeats
                task=rng.choices(TASKS[2:],weights[2:])[0]
                use_ar=not jazz_blues and self.ar_mixtures and rng.random()<self.config.AR_MIXTURE_PROB
                pool=self.ar_mixtures if use_ar else self.stem_pools[jazz_blues]
                if rng.random()<self.config.STEM_DRUM_PROB:
                    drummed=self.task_pools['ar' if use_ar else jazz_blues,'drum']
                    pool=drummed or pool
                if not pool:
                    pool=self.jazz_blues if jazz_blues else self.other
                # Jazz instrument weighting applies to mixture draws, without
                # removing any file from the four full jazz/blues traversals.
                sample=rng.sample(pool,min(32,len(pool)))
                first=rng.choices(sample,[max([instrument_weight(i,self.config.JAZZ_INSTRUMENT_WEIGHT)
                                       for i in r.get('instruments',[])] or [1.]) for r in sample])[0]
        else:
            first=self.eval_rows[idx]
            jazz_blues=first['genre'] in ('jazz','blues')
            pool=self.jazz_blues if jazz_blues else self.other
            task=rng.choices(TASKS,weights)[0]
        stats=Counter()
        for attempt in range(64):
            row = first if attempt == 0 else rng.choice(pool)
            try:
                record = dict(self.load_record(row), _augment=self.split == 'train')
                try:
                    result=prepare_example(record, rng, self.config, task)
                except NoStemWindow:
                    if task != 'stem':
                        raise
                    # Header instrument hints can disagree with the audible MIDI.
                    # Reuse this same piece for a valid task instead of discarding
                    # it or pretending that a missing second stem exists.
                    weights=self.config.TASK_WEIGHTS[:2]
                    fallback=rng.choices(TASKS[:2],weights if any(weights) else (1.,1.))[0]
                    result=prepare_example(record,rng,self.config,fallback)
                    self.routed_counts[task+' -> '+fallback]+=1
                    stats['routed']+=1
                self.samples_seen+=1
                stats.update(attempts=attempt+1,retries=attempt)
                result['data_stats']=torch.tensor([stats[k] for k in DATA_STAT_NAMES])
                return result
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                self.retry_counts[task+': '+str(exc)[:160]]+=1
                stats[retry_reason(exc)]+=1
        raise RuntimeError(f'Could not sample a valid {task} {"jazz/blues" if jazz_blues else "non-jazz"} example; '
                           f'retry reasons: {dict(stats)}')


def collate_examples(batch):
    length = max(len(x['input']) for x in batch)
    inputs = torch.full((len(batch), length), cfg.PAD_ID, dtype=torch.long)
    targets = torch.full_like(inputs, cfg.IGNORE_INDEX)
    masks = {k: torch.zeros_like(inputs, dtype=torch.bool) for k in ('stem_mask',)}
    for i, x in enumerate(batch):
        inputs[i, :len(x['input'])] = x['input']
        targets[i, :len(x['target'])] = x['target']
        for k in masks:
            masks[k][i, :len(x[k])] = x[k]
    data_stats=sum((x.get('data_stats',torch.tensor([1]+[0]*(len(DATA_STAT_NAMES)-1))) for x in batch),
                   start=torch.zeros(len(DATA_STAT_NAMES),dtype=torch.long))
    corruption = torch.zeros(6, dtype=torch.float64)
    for example in batch:
        actual = example.get('refinement_changed_fraction')
        if actual is not None:
            corruption += torch.tensor([1., example['requested_strength'], float(example['strength']),
                                        actual, float(actual <= .2), float(actual >= .8)], dtype=torch.float64)
    references={}
    for name in ('style','beat'):
        length=max(1,max(len(x[name+'_ids']) for x in batch))
        ids=torch.full((len(batch),length),cfg.PAD_ID,dtype=torch.long)
        for i,x in enumerate(batch):ids[i,:len(x[name+'_ids'])]=x[name+'_ids']
        references[name+'_ids']=ids
        references[name+'_attention_mask']=ids.ne(cfg.PAD_ID)
    return dict(**references,input=inputs,target=targets,task=torch.tensor([x['task'] for x in batch]),data_stats=data_stats,corruption_summary=corruption,**masks,
        **{k:torch.stack([x[k] for x in batch]) for k in ('genre','multi_hot','strength','style_present','beat_present','stem_instrument','exposure')})


def build_dataloader(split, batch_size, distributed=False, config=cfg):
    dataset = MelodyDataset(config.DATA_FILE, split, config)
    sampler = DistributedSampler(dataset, num_replicas=None if distributed else 1,
                                 rank=None if distributed else 0, shuffle=split == 'train', seed=config.DATA_SPLIT_SEED)
    options = dict(num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available(),
                   collate_fn=collate_examples, generator=torch.Generator().manual_seed(config.DATA_SPLIT_SEED))
    if config.NUM_WORKERS:
        options['prefetch_factor'] = config.PREFETCH_FACTOR
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, **options)
