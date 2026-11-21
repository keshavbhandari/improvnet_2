"""Constrained unmarked-note decoding for all three tasks."""
import torch
from tqdm import tqdm
from improvnet.inference.melody_runtime import TOP_K
from improvnet.model.kv_cache import reserve_kv_cache
from improvnet.inference.melody_stem_anchors import StemAnchors
from improvnet.utils.melody import Note, ordered, PREFIX_TOKENS
from improvnet.utils.melody_stems import DRUM_KIT_PITCHES, PERCUSSION_MODES

@torch.no_grad()
def generate_notes(self, prompt, genre, strength, instruments, window_ms,
             temperature=0.9, fixed_notes=None, monophonic=False,
             max_new_tokens=None, progress_desc='Decoding tokens',
             reference_features=None,reference_mask=None,reference_positions=None,drum_profile=None,drum_palette=None,stop_at_window_end=False,
             target_instrument=None, stem_add=False,percussion_palette='kit',
             protected_pitches=None):
    """Generate notes while retaining every source anchor in stem addition."""
    tok, device = self.tokenizer, self.device
    if stem_add and fixed_notes is None:
        raise ValueError('Stem decoding requires source anchors')
    if fixed_notes is not None and not stem_add:
        raise ValueError('Source anchors apply only to stem addition')
    if target_instrument and (target_instrument not in instruments or not stem_add):
        raise ValueError('The missing stem must be part of the requested output mix')
    if percussion_palette not in PERCUSSION_MODES:
        raise ValueError('Unknown percussion palette')
    addition_instruments=[target_instrument] if target_instrument else []
    self.last_decode_stats = dict(duplicate_attacks_removed=0, end_reason=None,
                                 decoded_tokens=0,history_tokens_dropped=0)
    if genre not in self.meta['genres'] or not 0 <= strength <= 1:
        raise ValueError('Unknown genre or strength outside [0,1]')
    budget = self.meta['seq_len'] - len(prompt) - PREFIX_TOKENS
    if max_new_tokens is not None:
        budget = min(budget, max_new_tokens)
    fixed_onsets = [n.onset for n in fixed_notes] if fixed_notes is not None else []
    repairs = (StemAnchors(fixed_notes, tok, self.meta, window_ms, protected_pitches)
               if stem_add and self.meta.get('stem_source_corruption_prob',1.)>0 else None)
    fixed_length = 4 * len(fixed_onsets) + (fixed_onsets[-1] // tok.abs_time_step_ms if fixed_onsets else 0) + 1
    if budget < 5 or (fixed_notes is not None and budget < fixed_length):
        raise ValueError('Prompt leaves insufficient room for a complete response')
    with self.model_autocast(), tqdm(total=budget, desc=progress_desc, unit='tok', position=getattr(self, 'progress_position', 0)+1,
              leave=False, dynamic_ncols=True,disable=getattr(self,'progress_disabled',False)) as token_progress:
        conditions = self.processor.get_instrument_multihot(
            [('instrument', inst) for inst in instruments]).unsqueeze(0).to(device)
        token_progress.set_postfix_str('prefill')
        ids = torch.tensor([[tok.tok_to_id[t] for t in prompt]], device=device)
        def prefill(ids):
            result = self.model(ids, genre=torch.tensor([self.meta['genres'].index(genre)], device=device),
                                       multi_hot=conditions, strength=torch.tensor([strength], device=device),
                                       percussion_mode=torch.tensor([PERCUSSION_MODES[percussion_palette]],device=device),
                                       reference_features=reference_features.unsqueeze(0).to(device) if reference_features is not None else None,
                                       reference_mask=reference_mask.unsqueeze(0).to(device) if reference_mask is not None else None,
                                       reference_positions=reference_positions.unsqueeze(0).to(device) if reference_positions is not None else None,
                                       drum_profile=drum_profile.unsqueeze(0).to(device) if drum_profile is not None else None,
                                       use_cache=True)
            logits, cache = result
            if getattr(self, 'preallocate_kv', True):
                cache = reserve_kv_cache(cache, self.meta['seq_len'])
            return logits, cache
        logits, cache = prefill(ids)
        token_progress.set_postfix_str('decoding')
        offset = len(prompt) + PREFIX_TOKENS
        output, stage, segment, last_onset, count = ['<S>'], 0, 0, -tok.time_step_ms, 0
        addition = False
        completed, seen_attacks = [], set()
        def finish_chunk(reason):
            if stem_add:
                if count != len(fixed_notes):
                    raise RuntimeError('Stem decoding ended before preserving all retained source notes')
                added=sum(n.instrument==target_instrument for n in completed)
                self.last_decode_stats.update(stem_note_count=added,stem_rest=added==0)
                if repairs:
                    self.last_decode_stats.update(repairs.statistics())
            self.last_decode_stats.update(end_reason=reason,response_token_budget=budget)
            return ordered(completed)
        inst_ids = [tok.tok_to_id[('instrument', inst)] for inst in instruments]
        if not inst_ids:
            raise ValueError('No supported output instrument')
        pitch_ids = [tok.tok_to_id[t] for t in tok.note_tokens if t[2] > 0]
        allowed_drums=DRUM_KIT_PITCHES if drum_palette is None else DRUM_KIT_PITCHES.intersection(drum_palette)
        drum_pitch_ids = {tok.tok_to_id[t] for t in tok.note_tokens if t[1] in allowed_drums and t[2] > 0}
        duration_ids = [tok.tok_to_id[t] for t in tok.dur_tokens if t[1] > 0]
        def note_start_tokens(remaining):
            """Legal unmarked note starts, preserving the complete anchor reserve."""
            if stem_add:
                next_note=fixed_notes[count] if count<len(fixed_notes) else None
                clock_tokens=max(0,(fixed_onsets[-1]-segment)//tok.abs_time_step_ms) if next_note else 0
                reserve=4*(len(fixed_notes)-count)+clock_tokens+1
                legal=[]
                lower, upper = (repairs.onset_bounds(count, last_onset) if repairs and next_note
                                else (next_note.onset, next_note.onset) if next_note else (0, 0))
                if next_note and remaining>=reserve and lower < segment+tok.abs_time_step_ms and upper >= segment:
                    choices={next_note.instrument}
                    legal=[tok.tok_to_id[('instrument',inst)] for inst in sorted(choices)
                        if repairs or (inst,next_note.pitch,next_note.onset) not in seen_attacks]
                    if not legal:
                        legal=[tok.tok_to_id[('instrument',inst)] for inst in sorted(choices)]
                minimum=max(segment,last_onset)
                latest=min(segment+tok.abs_time_step_ms,window_ms)
                if next_note:
                    latest=min(latest,upper+tok.time_step_ms)
                if addition_instruments and remaining>=reserve+4 and minimum<latest:
                    legal.extend(tok.tok_to_id[('instrument',inst)] for inst in addition_instruments)
                boundary=segment+tok.abs_time_step_ms
                if next_note is None:
                    # A missing part can rest in this chunk after retaining all
                    # source notes; no forced final-tick percussion hits.
                    legal.append(tok.tok_to_id['<E>'])
                    if (boundary<window_ms and remaining>=6) or boundary>=window_ms:
                        legal.append(tok.tok_to_id['<T>'])
                elif upper>=boundary and remaining >= 4*(len(fixed_notes)-count)+max(0,(fixed_onsets[-1]-boundary)//tok.abs_time_step_ms)+2:
                    legal.append(tok.tok_to_id['<T>'])
            else:
                legal = [tok.tok_to_id['<E>']] if count else []
                minimum = max(segment, last_onset + (tok.time_step_ms if monophonic else 0))
                if remaining>=5 and minimum < min(segment + tok.abs_time_step_ms, window_ms):
                    legal += inst_ids
                if segment + tok.abs_time_step_ms < window_ms and remaining>=6:
                    legal.append(tok.tok_to_id['<T>'])
                elif stop_at_window_end and segment + tok.abs_time_step_ms >= window_ms:
                    legal.append(tok.tok_to_id['<T>'])
            return legal

        while self.last_decode_stats['decoded_tokens'] < budget:
            # Reclaim the prompt's oldest history if a long response
            # needs more room. Rebuild its cache with the response intact.
            remaining = budget-self.last_decode_stats['decoded_tokens']
            reserve = 4*(len(fixed_notes)-count)+2 if stem_add else 0
            if (stage==0 and max_new_tokens is None
                    and remaining <= max(64,reserve+32)):
                first_sep = prompt.index('<SEP>')
                header = 3 if prompt[0]=='<ADD_STEM>' else 1
                if first_sep>header:
                    removed=first_sep-header
                    prompt=prompt[:header]+prompt[first_sep:]
                    rebuilt=prompt+output[1:]
                    logits,cache=prefill(torch.tensor([[tok.tok_to_id[t] for t in rebuilt]],device=device))
                    offset=len(rebuilt)+PREFIX_TOKENS
                    budget+=removed
                    token_progress.total=budget
                    self.last_decode_stats['history_tokens_dropped']+=removed
            if stage == 0:
                remaining = budget - (len(output)-1)
                legal = note_start_tokens(remaining)
            elif stage == 1:
                legal = ([tok.tok_to_id[t] for t in repairs.pitch_tokens(count)]
                         if repairs and not addition else
                         [tok.tok_to_id[('note', fixed_notes[count].pitch,
                          tok._quantize_velocity(fixed_notes[count].velocity))]]
                         if fixed_notes is not None and not addition else pitch_ids)
            elif stage == 2:
                if stem_add:
                    if addition:
                        minimum=max(segment,last_onset)
                        maximum=segment+tok.abs_time_step_ms
                        if count<len(fixed_notes):
                            upper = (repairs.onset_bounds(count, last_onset)[1] if repairs else fixed_onsets[count])
                            maximum=min(maximum,upper+tok.time_step_ms)
                        legal=[tok.tok_to_id[t] for t in tok.onset_tokens
                               if minimum<=segment+t[1]<maximum]
                    elif repairs:
                        minimum, maximum = repairs.onset_bounds(count, last_onset)
                        legal = [tok.tok_to_id[t] for t in tok.onset_tokens
                                 if minimum <= segment+t[1] <= maximum and segment+t[1] >= segment]
                    else:
                        legal=[tok.tok_to_id[('onset',fixed_onsets[count]-segment)]]
                else:
                    # Keep notes ordered, permit simultaneous harmony notes.
                    minimum = max(segment, last_onset)
                    if monophonic:
                        minimum = max(segment, last_onset + tok.time_step_ms)
                    maximum = (segment + tok.abs_time_step_ms if stop_at_window_end
                               else min(window_ms, segment + tok.abs_time_step_ms))
                    legal = [tok.tok_to_id[t] for t in tok.onset_tokens
                             if minimum <= segment + t[1] < maximum]
            else:
                legal = ([tok.tok_to_id[t] for t in repairs.duration_tokens(count)]
                         if repairs and not addition else
                         [tok.tok_to_id[('dur', tok._quantize_dur(fixed_notes[count].duration))]]
                         if stem_add and not addition else duration_ids)
            if stage == 1 and output[-1] == ('instrument', 'drum') and (addition or fixed_notes is None) and percussion_palette == 'kit':
                legal = [token_id for token_id in legal if token_id in drum_pitch_ids]
            if not legal:
                raise RuntimeError('No valid next token; reduce timing masking or shorten the excerpt')
            scores = logits[0, -1, legal].float()
            if stage == 1 and repairs and not addition:
                # A repair is optional. Favor retaining accompaniment pitches
                # unless the model has clear evidence, in addition to the cap.
                scores -= torch.tensor([0. if tok.id_to_tok[i][1] == fixed_notes[count].pitch else 2.
                                        for i in legal], device=device)
            # Transfer just the chosen index. A non-finite row uses a sentinel
            # in the same transfer, avoiding a second GPU synchronization.
            finite = torch.isfinite(scores).any()
            if temperature <= 0:
                pick = torch.where(finite,scores.argmax(),-1).item()
            else:
                scores = scores / temperature
                if TOP_K < len(legal):
                    threshold = scores.topk(TOP_K).values[-1]
                    scores[scores < threshold] = -torch.inf
                safe_scores = torch.where(finite,scores,torch.zeros_like(scores))
                sampled = torch.multinomial(safe_scores.softmax(-1), 1, generator=getattr(self, 'sampling_generator', None))
                pick = torch.where(finite,sampled,-1).item()
            if pick < 0:
                raise RuntimeError('No finite legal logits')
            token_id = legal[pick]
            token = tok.id_to_tok[token_id]
            if stage==0 and isinstance(token,tuple):
                addition=stem_add and token[1] in addition_instruments
            token_progress.update(1)
            self.last_decode_stats['decoded_tokens'] += 1
            if stop_at_window_end and (
                    stage == 0 and token == '<T>' and segment + tok.abs_time_step_ms >= window_ms
                    or stage == 2 and segment + token[1] >= window_ms):
                # Instrument/pitch may already be sampled for a future note;
                # only complete in-window notes are returned. Held durations
                # are untouched, including notes sounding across the boundary.
                return finish_chunk('window_boundary')
            output.append(token)
            if token == '<E>':
                return finish_chunk('end_token')
            if token == '<T>':
                segment += tok.abs_time_step_ms
            else:
                if stage == 2:
                    last_onset = segment + token[1]
                if stage == 3:
                    inst, pitch_velocity = output[-4], output[-3]
                    note = Note(inst[1], pitch_velocity[1], pitch_velocity[2], last_onset, token[1])
                    if repairs and not addition:
                        repairs.commit(count, note)
                    key = (note.instrument, note.pitch, note.onset)
                    if (fixed_notes is None or stem_add) and key in seen_attacks:
                        self.last_decode_stats['duplicate_attacks_removed'] += 1
                    else:
                        completed.append(note)
                        seen_attacks.add(key)
                    if not addition:
                        count += 1
                stage = (stage + 1) % 4
            logits, cache = self.model(torch.tensor([[token_id]], device=device),
                                      use_cache=True, past_key_values=cache, seq_offset=offset)
            offset += 1
        raise RuntimeError('Response exhausted its token budget before <E>')
