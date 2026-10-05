"""MIDI listening helpers; scoring never depends on seconds or velocity.

Skyline is optional and deliberately timing-dependent. MIDI outputs use actual
excerpt notes (not a repaired/scaffolded reconstruction of model tokens).
"""

from dataclasses import dataclass, replace
from bisect import bisect_right
import heapq
import math
from numbers import Integral
from pathlib import Path
import random

import mido

from improvnet.tokenizer.midi import MidiDict


@dataclass(frozen=True)
class NoteEvent:
    pitch: int
    onset: float
    end: float
    velocity: int = 80
    channel: int = 0
    program: int = 0

    def __post_init__(self):
        for key, limit in (("pitch", 127), ("velocity", 127), ("channel", 15), ("program", 127)):
            value = getattr(self, key)
            if not isinstance(value, Integral) or isinstance(value, bool) or not 0 <= value <= limit:
                raise ValueError(f"Invalid MIDI {key}")
        if not math.isfinite(self.onset) or not math.isfinite(self.end) or not 0 <= self.onset < self.end or self.velocity < 1:
            raise ValueError("Need a sounding note with finite nonnegative onset and positive duration")


def order_notes(notes):
    # Chord ties sort by pitch, not velocity/instrument/track enumeration.
    return sorted(notes, key=lambda n: (n.onset, -n.pitch, n.channel))


def read_midi(path):
    midi = MidiDict.from_midi(path)
    programs = {}
    for message in sorted(midi.instrument_msgs, key=lambda message: message["tick"]):
        programs.setdefault(message["channel"], []).append((message["tick"], message["data"]))
    tempos = sorted(midi.tempo_msgs, key=lambda message: message["tick"])
    # Precise seconds (not integer-millisecond rounding), including tempo changes.
    ticks, seconds, values = [0], [0.], [500000]
    for message in tempos:
        tick, value = message["tick"], message["data"]
        if tick == ticks[-1]:
            values[-1] = value
        else:
            seconds.append(seconds[-1] + mido.tick2second(tick-ticks[-1], midi.ticks_per_beat, values[-1]))
            ticks.append(tick)
            values.append(value)

    def timestamp(tick):
        index = bisect_right(ticks, tick)-1
        return seconds[index] + mido.tick2second(tick-ticks[index], midi.ticks_per_beat, values[index])

    notes = []
    for message in midi.note_msgs:
        data = message["data"]
        if data["end"] <= data["start"] or data["velocity"] <= 0:
            continue
        onset, end = timestamp(data["start"]), timestamp(data["end"])
        changes = programs.get(message["channel"], [])
        index = bisect_right([change[0] for change in changes], data["start"])-1
        program = changes[index][1] if index >= 0 else 0
        notes.append(NoteEvent(data["pitch"], onset, end, data["velocity"], message["channel"], program))
    return order_notes(notes)


def clip_segment(notes, start, seconds):
    if not math.isfinite(start) or start < 0 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Invalid segment start/length")
    # Select attacks within the interval. Notes already sounding at the start
    # are excluded to avoid inventing attacks. Clip tails and reset time to zero.
    return order_notes([replace(note, onset=note.onset - start, end=min(note.end, start + seconds) - start)
                        for note in notes if start <= note.onset < start + seconds])


def random_segment_start(notes, seconds, rng, *, exclude=None):
    if not notes or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Need music and a positive segment length")
    duration = max(note.end for note in notes)
    latest = max(0., duration - seconds)
    for _ in range(128):
        start = rng.uniform(0., latest)
        if exclude is not None and not (start + seconds <= exclude[0] or start >= exclude[1]):
            continue
        if any(note.channel != 9 for note in clip_segment(notes, start, seconds)):
            return start
    raise ValueError("Cannot find a nonempty pitched segment (or a disjoint second segment); use explicit starts/a shorter duration")


def skyline(notes):
    """Highest sounding non-drum voice, with stable ties and half-open intervals.

    Endpoints before starts handle adjacent notes correctly. A lower note may
    reappear when a higher one ends. Separate same-pitch attacks are retained.
    """
    notes = [note for note in order_notes(notes) if note.channel != 9]
    points = {}
    for i, note in enumerate(notes):
        points.setdefault(note.onset, [[], []])[1].append(i)
        points.setdefault(note.end, [[], []])[0].append(i)
    active, heap, result, previous = set(), [], [], None
    times = sorted(points)
    for position, time in enumerate(times[:-1]):
        ending, starting = points[time]
        active.difference_update(ending)
        for i in starting:
            active.add(i)
            heapq.heappush(heap, (-notes[i].pitch, notes[i].onset, i))
        while heap and heap[0][2] not in active:
            heapq.heappop(heap)
        winner = heap[0][2] if heap else None
        if winner is not None:
            if winner == previous:
                result[-1] = replace(result[-1], end=times[position + 1])
            else:
                result.append(replace(notes[winner], onset=time, end=times[position + 1]))
        previous = winner
    return result


def pitched_sequence(notes, representation="all-notes"):
    if representation not in ("all-notes", "skyline"):
        raise ValueError("Unknown pitch representation")
    selected = skyline(notes) if representation == "skyline" else order_notes(notes)
    return [note.pitch for note in selected if note.channel != 9]


def modify_segment(notes, mode, rng, *, probability=.15, transpose=7):
    if mode not in ("copy", "timing", "ornament", "substitute", "shuffle", "transpose", "mixed", "unrelated"):
        raise ValueError("Unknown modification mode")
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Modification probability must be in [0,1]")
    if not isinstance(transpose, Integral) or isinstance(transpose, bool) or abs(transpose) > 127:
        raise ValueError("Invalid transpose")
    notes = order_notes(notes)
    result = []
    for note in notes:
        pitch = note.pitch
        if note.channel != 9:
            if mode == "transpose":
                pitch += transpose
                if not 0 <= pitch <= 127:
                    raise ValueError("Transposition leaves MIDI pitch range; choose a smaller shift")
            elif mode in ("substitute", "mixed") and rng.random() < probability:
                offsets = [offset for offset in (-2, -1, 1, 2) if 0 <= pitch + offset <= 127]
                pitch += rng.choice(offsets)
            elif mode == "unrelated":
                pitch = rng.randint(48, 84)
        onset, end, velocity = note.onset, note.end, note.velocity
        if mode in ("timing", "mixed"):
            # Strictly increasing shared onset warp preserves note order/ties.
            onset = 1.15 * (note.onset + .04 * math.sin(2 * math.pi * note.onset))
            end = onset + (note.end - note.onset) * rng.uniform(.6, 1.4)
            velocity = rng.randint(40, 110)
        result.append(replace(note, pitch=pitch, onset=onset, end=end, velocity=velocity))
        if mode in ("ornament", "mixed") and note.channel != 9 and rng.random() < probability:
            choices = [pitch + offset for offset in (-1, 1) if 0 <= pitch + offset <= 127]
            duration = end - onset
            result.append(replace(note, pitch=rng.choice(choices), onset=onset + duration * .35,
                                  end=onset + duration * .6, velocity=max(1, velocity - 15)))
    if mode == "shuffle":
        slots = [i for i, note in enumerate(result) if note.channel != 9]
        pitches = [result[i].pitch for i in slots]
        rng.shuffle(pitches)
        for i, pitch in zip(slots, pitches):
            result[i] = replace(result[i], pitch=pitch)
    return order_notes(result)


def save_midi(notes, path):
    """Serialize without clipping pitches, overwriting files or mutating notes."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    notes = order_notes(notes)
    if not notes:
        raise ValueError("Cannot save an empty listening excerpt")
    programs = {}
    for note in notes:
        if note.channel in programs and programs[note.channel] != note.program:
            raise ValueError("Multiple programs on one excerpt channel are unsupported")
        programs[note.channel] = note.program
    midi = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    midi.tracks.append(meta)
    for channel, program in sorted(programs.items()):
        track, events = mido.MidiTrack(), []
        track.append(mido.Message("program_change", program=program, channel=channel, time=0))
        # MIDI cannot independently represent overlapping identical pitches on
        # one channel. Shorten only the listening copy before the next attack;
        # pitches/order used by the metric remain unchanged.
        selected = [note for note in notes if note.channel == channel]
        attacks = {}
        for note in reversed(selected):
            start = round(note.onset * 960)
            end = max(start + 1, round(note.end * 960))
            if note.pitch in attacks and attacks[note.pitch] > start:
                end = min(end, attacks[note.pitch])
            attacks[note.pitch] = start
            events.extend(((start, 1, mido.Message("note_on", note=note.pitch, velocity=note.velocity, channel=channel)),
                           (end, 0, mido.Message("note_off", note=note.pitch, velocity=0, channel=channel))))
        previous = 0
        for tick, _, message in sorted(events, key=lambda event: (event[0], event[1], event[2].note)):
            message.time = tick - previous
            track.append(message)
            previous = tick
        midi.tracks.append(track)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation makes the no-overwrite guarantee race-safe.
    with path.open("xb") as handle:
        midi.save(file=handle)
