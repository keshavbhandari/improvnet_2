"""Small, bounded repairs to retained notes during V5 stem generation.

The accompaniment can move slightly while the melody pitches, every retained
instrument/note, exact chord grouping and the chunk's ownership clock survive.
"""
from collections import defaultdict
import math

from improvnet.utils.melody import skyline


class StemAnchors:
    def __init__(self, notes, tokenizer, metadata, window_ms, protected=None):
        self.notes, self.tok = notes, tokenizer
        self.timing = min(50, max(0, int(metadata.get('stem_timing_jitter_ms', 50))))
        self.pitch_radius = min(7, max(0, int(metadata.get('stem_pitch_shift_semitones', 7))))
        self.times = sorted({n.onset for n in notes})
        self.groups = defaultdict(list)
        for index, note in enumerate(notes):
            self.groups[note.onset].append(index)
        self.selected_onsets = {}
        self.last_group_onset = None
        self.last_group_time = None
        if protected is None:
            melody, _ = skyline([n for n in notes if n.instrument != 'drum'],
                                 metadata.get('skyline_tolerance_ms', 50))
            protected = {id(n) for n in melody}
            self.protected = [id(n) in protected for n in notes]
        else:
            self.protected = list(protected)
            if len(self.protected) != len(notes):
                raise ValueError('One melody protection flag is required per retained note')
        eligible = sum(n.instrument != 'drum' and not keep for n, keep in zip(notes, self.protected))
        fraction = min(.05, max(0., float(metadata.get('stem_pitch_shift_prob', .05))))
        self.pitch_budget = math.floor(eligible * fraction)
        self.pitch_changes = 0
        self.timing_changes = 0
        self.duration_changes = 0
        self.used_pitches = defaultdict(set)
        self.last_pitch_key = {}
        # Leave enough room for every later distinct source attack, even when
        # one correction is delayed. This also handles a 4.99s -> 5.00s move.
        latest = max(0, window_ms - tokenizer.time_step_ms)
        self.upper = {}
        for onset in reversed(self.times):
            latest = min(onset + self.timing, latest)
            self.upper[onset] = latest
            latest -= tokenizer.time_step_ms
        self.by_time = {t: i for i, t in enumerate(self.times)}

    def onset_bounds(self, index, last_onset):
        time = self.notes[index].onset
        if time in self.selected_onsets:
            onset = self.selected_onsets[time]
            return onset, onset
        minimum = max(0, time - self.timing, last_onset)
        if self.last_group_time is not None and self.last_group_time != time:
            minimum = max(minimum, self.last_group_onset + self.tok.time_step_ms)
        return minimum, self.upper[time]

    def pitch_tokens(self, index):
        note = self.notes[index]
        velocity = self.tok._quantize_velocity(note.velocity)
        if note.instrument == 'drum' or self.protected[index] or self.pitch_changes >= self.pitch_budget:
            return [('note', note.pitch, velocity)]
        reserved = {self.notes[i].pitch for i in self.groups[note.onset]
                    if i != index and self.notes[i].instrument == note.instrument}
        seen = self.used_pitches[(note.onset, note.instrument)]
        group = self.groups[note.onset]
        position = group.index(index)
        previous = self.notes[group[position-1]] if position else None
        following = self.notes[group[position+1]] if position+1 < len(group) else None
        lower = self.last_pitch_key.get(note.onset,
            (-previous.pitch, previous.instrument) if previous else (-128, ''))
        upper = (-following.pitch, following.instrument) if following else (1, '')
        return [('note', pitch, velocity)
                for pitch in range(max(0, note.pitch-self.pitch_radius), min(127, note.pitch+self.pitch_radius)+1)
                if pitch == note.pitch or pitch not in reserved | seen
                and lower <= (-pitch, note.instrument) <= upper]

    def duration_tokens(self, index):
        original = self.tok._quantize_dur(self.notes[index].duration)
        return [token for token in self.tok.dur_tokens
                if token[1] > 0 and abs(token[1]-original) <= self.timing]

    def commit(self, index, note):
        original = self.notes[index]
        previous = self.selected_onsets.setdefault(original.onset, note.onset)
        if previous != note.onset:
            raise RuntimeError('A stem correction split a retained chord')
        self.last_group_time, self.last_group_onset = original.onset, note.onset
        self.pitch_changes += int(note.pitch != original.pitch)
        self.timing_changes += int(note.onset != original.onset)
        self.duration_changes += int(note.duration != self.tok._quantize_dur(original.duration))
        self.used_pitches[(original.onset, original.instrument)].add(note.pitch)
        self.last_pitch_key[original.onset] = (-note.pitch, note.instrument)

    def statistics(self):
        return dict(retained_pitch_corrections=self.pitch_changes,
                    retained_pitch_correction_budget=self.pitch_budget,
                    retained_onset_corrections=self.timing_changes,
                    retained_duration_corrections=self.duration_changes)
