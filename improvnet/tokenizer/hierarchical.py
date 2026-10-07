"""Hierarchical single-instrument roll + aligned accompaniment (piano-first).

This is a representation/round-trip prototype, NOT a learned compression model.
Timing is nearest-frame quantized; same-pitch overlaps are explicitly normalized.
The target MIDI is decoded from the canvas itself, never from saved source notes.
"""

from collections import Counter, defaultdict, deque
from bisect import bisect_right
from dataclasses import asdict, dataclass, replace
import io
import json
import math
import random
from pathlib import Path

import mido
import numpy as np

FORMAT_VERSION = "hierarchical_piano_events_v2"
EVENT_FIELDS = ("track", "channel", "program", "is_drum", "pitch", "velocity",
                "onset_bin", "onset_offset", "end_bin", "end_offset", "order")


@dataclass(frozen=True)
class GridConfig:
    coarse_ms: int = 1000
    fine_ms: int = 10
    piano_programs: tuple[int, ...] = tuple(range(8))  # GM piano family; zero-based.
    include_pedal: bool = True
    max_frames: int = 360_000  # One hour at the default resolution.
    min_output_ppq: int = 9600  # Avoid adding coarse source-tick rounding to 10ms frames.

    def __post_init__(self):
        for name in ("coarse_ms", "fine_ms", "max_frames", "min_output_ppq"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.coarse_ms % self.fine_ms or self.min_output_ppq > 32767:
            raise ValueError("coarse_ms must be divisible by fine_ms; min_output_ppq must be <= 32767")
        if not self.piano_programs or any(isinstance(p, bool) or not isinstance(p, int) or not 0 <= p <= 127 for p in self.piano_programs):
            raise ValueError("piano_programs must contain zero-based GM program numbers")

    @property
    def frames_per_bin(self):
        return self.coarse_ms // self.fine_ms

    def frame(self, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid MIDI time")
        # Explicit half-up rounding, not Python's tie-to-even round().
        return int(math.floor(seconds * 1000 / self.fine_ms + .5 + 1e-9))


@dataclass(frozen=True)
class Note:
    track: int
    channel: int
    program: int
    pitch: int
    velocity: int
    start: float
    end: float
    order: int


class TempoMap:
    def __init__(self, ppq, changes):
        if ppq <= 0:
            raise ValueError("SMPTE time division is not supported")
        self.ppq = ppq
        by_tick = {0: 500_000}
        for tick, tempo in changes:
            if tick < 0 or tempo < 1:
                raise ValueError("Invalid tempo map")
            by_tick[int(tick)] = int(tempo)
        self.ticks = np.array(sorted(by_tick), dtype=np.int64)
        self.tempos = np.array([by_tick[int(t)] for t in self.ticks], dtype=np.int64)
        self.seconds = np.zeros(len(self.ticks), dtype=np.float64)
        if len(self.ticks) > 1:
            self.seconds[1:] = np.cumsum(np.diff(self.ticks) * self.tempos[:-1] / (ppq * 1_000_000))

    def to_seconds(self, tick):
        index = int(np.searchsorted(self.ticks, tick, side="right") - 1)
        return float(self.seconds[index] + (tick - self.ticks[index]) * self.tempos[index] / (self.ppq * 1_000_000))

    def to_tick(self, seconds):
        index = int(np.searchsorted(self.seconds, seconds, side="right") - 1)
        tick = self.ticks[index] + (seconds - self.seconds[index]) * self.ppq * 1_000_000 / self.tempos[index]
        return int(math.floor(tick + .5 + 1e-9))


def _unwrap_rmid(raw):
    """Extract SMF bytes from a validated RIFF/RMID container, without writes."""
    if raw[:4] != b"RIFF":
        return raw, False
    if len(raw) < 12 or raw[8:12] != b"RMID":
        raise ValueError("RIFF file is not an RMID MIDI container")
    end = int.from_bytes(raw[4:8], "little") + 8
    if end < 12 or end > len(raw):
        raise EOFError("Truncated RMID container")
    pos, midi = 12, None
    while pos < end:
        if pos + 8 > end:
            raise EOFError("Truncated RMID chunk header")
        kind = raw[pos:pos + 4]
        length = int.from_bytes(raw[pos + 4:pos + 8], "little")
        chunk_end = pos + 8 + length
        if chunk_end + length % 2 > end:
            raise EOFError("Truncated RMID chunk")
        if kind == b"data":
            if midi is not None:
                raise ValueError("Ambiguous RMID container with multiple MIDI data chunks")
            midi = raw[pos + 8:chunk_end]
        pos = chunk_end + length % 2
    if midi is None or not midi.startswith(b"MThd"):
        raise ValueError("RMID container has no valid MIDI data chunk")
    return midi, True


def _recover_midi_bytes(raw):
    """Apply narrow training repairs while preserving notes and absolute ticks.

    Recovery must not use clip=True: that also changes note/controller values.
    Removed events' deltas are transferred to the next event (or an end marker).
    Ignore invalid key-signature/SMPTE-offset metadata, pitchwheel messages and
    bank-select/volume CC0/CC32/CC7 values; normalize invalid note_off release
    velocity to zero. Invalid pitches, attack velocities, other controller
    values and structural corruption remain parser errors.
    """
    from mido.messages.specs import SPEC_BY_STATUS

    def variable(data, pos):
        value = 0
        for _ in range(4):
            if pos >= len(data):
                raise EOFError("Truncated MIDI variable-length field")
            byte = data[pos]
            pos += 1
            value = (value << 7) | (byte & 127)
            if byte < 128:
                return value, pos
        raise ValueError("Invalid MIDI variable-length field")

    def encode_variable(value):
        if value > 0x0fffffff:
            raise ValueError("Recovered MIDI delta exceeds MIDI range")
        result = [value & 127]
        while value > 127:
            value >>= 7
            result.append((value & 127) | 128)
        return bytes(reversed(result))

    raw, changed = _unwrap_rmid(raw)
    output, offset = bytearray(), 0
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise EOFError("Truncated MIDI chunk header")
        kind = raw[offset:offset + 4]
        size = int.from_bytes(raw[offset + 4:offset + 8], "big")
        end = offset + 8 + size
        if end > len(raw):
            raise EOFError("Truncated MIDI chunk")
        data = raw[offset + 8:end]
        offset = end
        if kind != b"MTrk":
            output.extend(kind + size.to_bytes(4, "big") + data)
            continue
        rebuilt, pos, pending_delta, running, channel_status = bytearray(), 0, 0, None, None
        while pos < len(data):
            delta, pos = variable(data, pos)
            event_start = pos
            if pos >= len(data):
                raise EOFError("Missing MIDI event")
            status = data[pos]
            implicit = status < 128
            if status >= 128:
                pos += 1
                if status != 255:
                    running = status
                if status < 240:
                    channel_status = status
            else:
                if running is None:
                    raise ValueError("MIDI running status without a status byte")
                status = running
                # Some XG files omit the channel status after SysEx. Restore
                # the last explicit channel status rather than interpreting a
                # note/controller byte as a SysEx length. No value is clipped.
                if status in (240, 247):
                    if channel_status is None:
                        raise ValueError("MIDI data after SysEx without a channel status")
                    status = running = channel_status
                    changed = True
            discard, replacement = False, None
            if status == 255:
                if pos >= len(data):
                    raise EOFError("Missing MIDI meta type")
                meta_type = data[pos]
                pos += 1
                length, pos = variable(data, pos)
                payload = data[pos:pos + length]
                if meta_type == 0x59:
                    discard = (length != 2 or len(payload) != 2 or
                               payload[0] not in (*range(8), *range(249, 256)) or payload[1] not in (0, 1))
                elif meta_type == 0x54:
                    # Offset metadata is unused by our relative PPQ canvas.
                    # Drop malformed offsets, never guess their frame rate.
                    discard = (length != 5 or len(payload) != 5 or
                               payload[0] > 127 or (payload[0] & 31) > 23 or
                               payload[1] > 59 or payload[2] > 59 or
                               payload[3] >= (24, 25, 30, 30)[payload[0] >> 5] or
                               payload[4] > 99)
                pos += length
            elif status in (240, 247):
                length, pos = variable(data, pos)
                payload = data[pos:pos + length]
                if payload[:1] == b"\xf0":
                    payload = payload[1:]
                if payload[-1:] == b"\xf7":
                    payload = payload[:-1]
                discard = any(byte > 127 for byte in payload)
                pos += length
            else:
                if status not in SPEC_BY_STATUS:
                    raise ValueError("Undefined MIDI status byte")
                value_start = pos
                pos += SPEC_BY_STATUS[status]["length"] - 1
                values = data[value_start:pos]
                if len(values) == 2 and status & 0xf0 == 0x80 and values[0] < 128 and values[1] > 127:
                    # Release velocity is not used for the key-down roll. Keep
                    # the pitch and exact release tick, not a clipped attack.
                    replacement = bytes((status, values[0], 0))
                    changed = True
                elif len(values) == 2 and status & 0xf0 == 0xb0 and values[0] in (0, 7, 32) and values[1] > 127:
                    discard = True
                elif len(values) == 2 and status & 0xf0 == 0xe0 and any(value > 127 for value in values):
                    # Pitch bend is not represented by the key-down canvas.
                    # Retain valid bends, discard invalid ones without clipping.
                    discard = True
            if pos > len(data):
                raise EOFError("Truncated MIDI event")
            if discard:
                pending_delta += delta
                changed = True
            else:
                rebuilt.extend(encode_variable(delta + pending_delta))
                # Removing a SysEx must not change how subsequent running-status
                # bytes are interpreted. Make implicit messages explicit.
                if replacement is not None:
                    rebuilt.extend(replacement)
                else:
                    if implicit:
                        rebuilt.append(status)
                    rebuilt.extend(data[event_start:pos])
                pending_delta = 0
        if pending_delta:
            rebuilt.extend(encode_variable(pending_delta) + b"\xff\x2f\x00")
        output.extend(kind + len(rebuilt).to_bytes(4, "big") + rebuilt)
    return bytes(output), changed


def read_performance(source, *, recover=False, drum_duration_ms=10):
    """Read all tracks using the GLOBAL tempo/channel state (MIDI type 0/1).

    FIFO pairs overlapping same-channel/pitch notes. Velocity-zero note_on is
    note_off. Sustain remains an independent controller, not baked into duration.
    Strict mode rejects dangling attacks. Training recovery retains paired notes,
    closes unresolved pitched attacks at their next same-pitch attack, and drops
    terminal unresolved pitched attacks. Unreleased drums get a short one-frame
    duration, capped at the next attack. RMID is unwrapped; malformed SysEx,
    key-signature/SMPTE-offset metadata, pitch bends and bank-select/volume values
    are ignored when invalid. Valid messages remain intact. Invalid note_off
    release velocity becomes zero; pitch/attack velocity are never clipped.
    No recovery logs/counts are added; sources are untouched.
    Negative delta times and structurally broken files remain errors.
    """
    if drum_duration_ms <= 0:
        raise ValueError("Drum fallback duration must be positive")
    try:
        midi = source if isinstance(source, mido.MidiFile) else mido.MidiFile(source)
    except (ValueError, EOFError, OSError, IndexError, KeyError, mido.KeySignatureError):
        if not recover or isinstance(source, mido.MidiFile):
            raise
        sanitized, changed = _recover_midi_bytes(Path(source).read_bytes())
        if not changed:
            raise
        midi = mido.MidiFile(file=io.BytesIO(sanitized))
    if midi.type == 2 or midi.ticks_per_beat <= 0:
        raise ValueError("Only synchronous PPQ MIDI types 0/1 are supported")
    timed = []
    end_tick = 0
    for track_id, track in enumerate(midi.tracks):
        tick = 0
        for index, message in enumerate(track):
            if message.time < 0:
                raise ValueError("Negative MIDI delta time")
            tick += message.time
            if tick < 0:
                raise ValueError("Negative MIDI timestamp")
            timed.append((tick, track_id, index, message))
        end_tick = max(end_tick, tick)
    timed.sort(key=lambda x: x[:3])
    tempos = [(tick, msg.tempo) for tick, _, _, msg in timed if msg.type == "set_tempo"]
    tempo = TempoMap(midi.ticks_per_beat, tempos)
    programs, active, notes, messages = [0] * 16, defaultdict(deque), [], []
    unmatched_offs = skipped_nonpositive_notes = 0
    attacks = defaultdict(list)

    def close(key, tick):
        nonlocal skipped_nonpositive_notes
        track, program, velocity, start, order = active[key].popleft()
        if tick <= start:
            skipped_nonpositive_notes += 1
            return
        notes.append(Note(track, key[0], program, key[1], velocity,
                          tempo.to_seconds(start), tempo.to_seconds(tick), order))

    for order, (tick, track, _, msg) in enumerate(timed):
        if msg.type == "note_on" and msg.velocity:
            active[msg.channel, msg.note].append((track, programs[msg.channel], msg.velocity, tick, order))
            if recover:
                attacks[msg.channel, msg.note].append(tick)
        elif msg.type == "note_off" or msg.type == "note_on":
            key = msg.channel, msg.note
            if active[key]:
                close(key, tick)
            else:
                unmatched_offs += 1
        elif msg.type != "end_of_track":
            if msg.type == "program_change":
                programs[msg.channel] = msg.program
            if msg.type == "control_change" and msg.control in (120, 123):
                for key in list(active):
                    while key[0] == msg.channel and active[key]:
                        close(key, tick)
            data = msg.dict()
            data.pop("time")
            messages.append(dict(tick=tick, track=track, order=order, is_meta=msg.is_meta, message=data))
    if any(active.values()) and not recover:
        raise ValueError(f"{sum(map(len, active.values()))} dangling note_on messages; no silent repair performed")
    if recover:
        for (channel, pitch), queue in active.items():
            for track, program, velocity, start, order in queue:
                starts = attacks[channel, pitch]
                index = bisect_right(starts, start)
                following = tempo.to_seconds(starts[index]) if index < len(starts) else None
                begin = tempo.to_seconds(start)
                finish = begin + drum_duration_ms / 1000 if channel == 9 else following
                if channel == 9 and following is not None:
                    finish = min(finish, following)
                if finish is not None and finish > begin:
                    notes.append(Note(track, channel, program, pitch, velocity, begin, finish, order))
    notes.sort(key=lambda n: (n.start, n.order))
    return dict(notes=notes, messages=messages, ppq=midi.ticks_per_beat,
                tempo_map=sorted({0: 500_000, **{int(t): int(v) for t, v in tempos}}.items()),
                tracks=len(midi.tracks), end_seconds=tempo.to_seconds(end_tick), unmatched_offs=unmatched_offs,
                skipped_nonpositive_notes=skipped_nonpositive_notes)


def normalize_roll_notes(notes, config):
    """Return quantized, single-voice-per-pitch notes and explicit loss counts.

    A new attack cuts the previous note's key duration, with no resumption of its
    tail. If attacks share a quantized frame, the latest true-time attack wins;
    exact-time ties use the greatest source MIDI event order. Preserve that
    winner's velocity/release, not the displaced note's release. Surviving notes
    occupy at least one frame. No such normalization is applied to accompaniment.
    """
    by_pitch = defaultdict(list)
    for note in sorted(notes, key=lambda n: (n.start, n.order)):
        by_pitch[note.pitch].append(note)
    result = []
    report = dict(overlap_normalizations=0, same_frame_collisions=0,
                  collision_frames=0, short_notes_extended=0)
    for group in by_pitch.values():
        cut = []
        for note in group:
            if cut and cut[-1].end > note.start:
                cut[-1] = replace(cut[-1], end=note.start)
                report["overlap_normalizations"] += 1
            cut.append(note)
        attacks = defaultdict(list)
        for note in cut:
            attacks[config.frame(note.start)].append(note)
        survivors = []
        for start, candidates in attacks.items():
            report["same_frame_collisions"] += len(candidates) - 1
            report["collision_frames"] += int(len(candidates) > 1)
            survivors.append((start, candidates[-1]))
        for index, (start, note) in enumerate(survivors):
            release = config.frame(note.end)
            report["short_notes_extended"] += int(release <= start)
            end = max(start + 1, release)
            if index + 1 < len(survivors):
                end = min(end, survivors[index + 1][0])
            result.append(replace(note, start=start * config.fine_ms / 1000,
                                  end=end * config.fine_ms / 1000))
    return sorted(result, key=lambda n: (n.start, n.order)), report


# Compatibility for the initial piano-only prototype and existing notebooks.
normalize_piano_notes = normalize_roll_notes


def roll_notes(velocity, onset, n_frames, channel, track, program=0, *, strict=False):
    """Scan continuously across bins; every onset retriggers, even at equal velocity.

    Strict mode rejects missing onsets, velocity drift within a key-down span,
    and onsets at zero velocity. Non-strict mode infers missing attacks from
    inactive→positive transitions, ignores zero-velocity attacks, and retains
    the attack velocity despite drift until an explicit onset or release.
    """
    flat = velocity.transpose(1, 0, 2).reshape(128, -1)[:, :n_frames]
    attacks = onset.transpose(1, 0, 2).reshape(128, -1)[:, :n_frames]
    notes = []
    for pitch in range(128):
        row, hits = flat[pitch], attacks[pitch]
        previous = np.r_[0, row[:-1]]
        points = np.flatnonzero(hits | (row != previous))
        start, attack_velocity = None, 0
        for frame in points:
            value, hit = int(row[frame]), bool(hits[frame])
            if strict and hit and not value:
                raise ValueError("Onset at zero velocity")
            if start is not None and (hit or not value):
                notes.append((track, channel, program, pitch, attack_velocity,
                              start, int(frame), len(notes)))
                start = None
            if value and start is None:
                if strict and not hit:
                    raise ValueError("Active velocity without onset")
                start, attack_velocity = int(frame), value
            elif strict and value and value != attack_velocity:
                raise ValueError("Velocity changed without onset")
        if start is not None:
            notes.append((track, channel, program, pitch, attack_velocity,
                          start, n_frames, len(notes)))
    return notes


@dataclass
class EncodedPiece:
    config: GridConfig
    velocity: np.ndarray          # uint8 [bin,128,subframe], key-down velocity 0..127
    onset: np.ndarray             # uint8 [bin,128,subframe], binary attacks
    pedal: np.ndarray             # uint8 [bin,subframe], continuous CC64 0..127
    valid_mask: np.ndarray        # bool [bin,subframe]; real silence is valid
    coarse_velocity: np.ndarray  # uint8 [bin,128], max active velocity across time
    coarse_onsets: np.ndarray    # int32 [bin,128], onset counts across time
    coarse_pedal: np.ndarray     # uint8 [bin], peak sustain value
    events: np.ndarray           # int64 [event,11]; accompaniment only
    event_ptr: np.ndarray        # int64 [bin+1], CSR pointers for onset bins
    metadata: dict

    @property
    def n_frames(self):
        return int(self.valid_mask.sum())

    def refresh_coarse(self):
        """Recompute derived summaries after explicitly editing the fine arrays."""
        self.coarse_velocity = self.velocity.max(axis=-1)
        self.coarse_onsets = self.onset.sum(axis=-1, dtype=np.int32)
        self.coarse_pedal = self.pedal.max(axis=-1)
        return self

    def validate(self):
        if self.velocity.ndim != 3 or not len(self.velocity):
            raise ValueError("Invalid velocity canvas dimensions")
        bins, frames = len(self.velocity), self.config.frames_per_bin
        for name, array, maximum in (("velocity", self.velocity, 127), ("onset", self.onset, 1)):
            if array.shape != (bins, 128, frames) or array.dtype != np.uint8 or np.any(array > maximum):
                raise ValueError(f"Invalid {name} canvas shape/dtype/values")
        if self.valid_mask.shape != (bins, frames) or self.valid_mask.dtype != np.bool_:
            raise ValueError("Invalid padding mask")
        valid = self.valid_mask.ravel()
        if not valid.any() or np.any(np.diff(valid.astype(np.int8)) > 0) or self.n_frames > self.config.max_frames:
            raise ValueError("Padding must be a trailing suffix; canvas exceeds configured limit")
        if any(np.any(a * ~self.valid_mask[:, None, :]) for a in (self.velocity, self.onset)):
            raise ValueError("Nonzero roll velocity/onset in padded frames")
        if self.pedal.shape != (bins, frames) or self.pedal.dtype != np.uint8 or np.any(self.pedal > 127) or np.any(self.pedal * ~self.valid_mask):
            raise ValueError("Invalid sustain/padding values")
        if self.metadata.get("pedal_enabled") is False and self.pedal.any():
            raise ValueError("Pedal canvas is disabled for this target")
        for name, actual, expected in (
                ("velocity", self.coarse_velocity, self.velocity.max(axis=-1)),
                ("onsets", self.coarse_onsets, self.onset.sum(axis=-1, dtype=np.int32)),
                ("pedal", self.coarse_pedal, self.pedal.max(axis=-1))):
            if actual.dtype != expected.dtype or actual.shape != expected.shape or not np.array_equal(actual, expected):
                raise ValueError(f"Coarse {name} summary does not match fine arrays; call refresh_coarse() after edits")
        if self.events.ndim != 2 or self.events.shape[1] != len(EVENT_FIELDS) or self.events.dtype != np.int64:
            raise ValueError("Invalid accompaniment event matrix")
        if self.event_ptr.shape != (bins + 1,) or self.event_ptr.dtype != np.int64 or self.event_ptr[0] != 0 or self.event_ptr[-1] != len(self.events) or np.any(np.diff(self.event_ptr) < 0):
            raise ValueError("Invalid event-bin pointers")
        if len(self.events):
            e = self.events
            starts, ends = e[:, 6] * frames + e[:, 7], e[:, 8] * frames + e[:, 9]
            if (np.any(e < 0) or np.any(e[:, 1] > 15) or np.any(e[:, 2] > 127) or np.any(e[:, 3] != (e[:, 1] == 9)) or
                    np.any(e[:, 4] > 127) or np.any(e[:, 5] < 1) or np.any(e[:, 5] > 127) or
                    np.any(e[:, [7, 9]] >= frames) or np.any(starts >= ends) or np.any(ends > self.n_frames) or np.any(np.diff(starts) < 0)):
                raise ValueError("Invalid/unsorted accompaniment note fields")
            expected = np.r_[0, np.cumsum(np.bincount(e[:, 6], minlength=bins))]
            if not np.array_equal(expected, self.event_ptr):
                raise ValueError("Events are misaligned to their onset-bin pointers")
        return self

    def context_for_bin(self, index):
        """All notes intersecting a bin, including carries from earlier onset bins."""
        if not 0 <= index < len(self.velocity):
            raise IndexError(index)
        frames = self.config.frames_per_bin
        starts = self.events[:, 6] * frames + self.events[:, 7]
        ends = self.events[:, 8] * frames + self.events[:, 9]
        return self.events[(starts < (index + 1) * frames) & (ends > index * frames)]

    def image(self):
        """Optional [bin,3,128,subframe] display view, not the storage format.

        Only this view broadcasts pedal over pitches; the stored controller has
        one value per frame, separate from key-down durations.
        """
        return np.stack((self.velocity, self.onset,
                         np.broadcast_to(self.pedal[:, None, :], self.velocity.shape)), axis=1)

    def save(self, path):
        self.validate()
        metadata = dict(format=FORMAT_VERSION, config=asdict(self.config), metadata=self.metadata)
        with Path(path).open("xb") as handle:
            np.savez_compressed(handle, velocity=self.velocity, onset=self.onset, pedal=self.pedal, valid_mask=self.valid_mask,
                                coarse_velocity=self.coarse_velocity, coarse_onsets=self.coarse_onsets,
                                coarse_pedal=self.coarse_pedal, events=self.events, event_ptr=self.event_ptr,
                                metadata=np.array(json.dumps(metadata)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as arrays:
            data = json.loads(str(arrays["metadata"]))
            if data["format"] != FORMAT_VERSION:
                raise ValueError(f"Unsupported encoding format {data['format']!r}; re-encode the source MIDI as {FORMAT_VERSION}")
            data["config"]["piano_programs"] = tuple(data["config"]["piano_programs"])
            return cls(GridConfig(**data["config"]), **{k: arrays[k].copy() for k in
                       ("velocity", "onset", "pedal", "valid_mask", "coarse_velocity", "coarse_onsets", "coarse_pedal", "events", "event_ptr")},
                       metadata=data["metadata"]).validate()


class HierarchicalTokenizer:
    channel_vocab_sizes = {"velocity": 128, "onset": 2, "pedal": 128}

    def __init__(self, config=None):
        self.config = config or GridConfig()

    def encode(self, source, *, target_channel=None, target_program=None,
               seed=None, piano_channel=None, recover=False):
        """Piano-first roll; otherwise random channel/program part, drums included.

        target_channel explicitly overrides piano priority. target_program can
        disambiguate program changes on that channel. piano_channel is retained
        as a piano-only compatibility option. seed controls only fallback choice.
        recover=True is the training-only permissive source policy; the default
        remains strict for round-trip checks. Recovery never writes source MIDI.
        """
        source_data = read_performance(source, recover=recover, drum_duration_ms=self.config.fine_ms)
        config, notes = self.config, source_data["notes"]
        piano = [n for n in notes if n.channel != 9 and n.program in config.piano_programs]
        channels = Counter(n.channel for n in piano)
        used = {n.channel for n in notes} | {m["message"]["channel"] for m in source_data["messages"] if "channel" in m["message"]}
        if piano_channel is not None and (target_channel is not None or target_program is not None):
            raise ValueError("Use target_channel/target_program or piano_channel, not both")
        if target_program is not None and (target_channel is None or isinstance(target_program, bool)
                                          or not isinstance(target_program, int) or not 0 <= target_program <= 127):
            raise ValueError("target_program requires target_channel and a GM program in [0,127]")
        if piano_channel is not None:
            if isinstance(piano_channel, bool) or not isinstance(piano_channel, int) or piano_channel == 9 or not 0 <= piano_channel <= 15:
                raise ValueError("Select a non-drum MIDI piano channel in [0,15]")
            if piano_channel in used and piano_channel not in channels:
                raise ValueError("Selected channel is occupied by a non-piano instrument")
            target_channel, selection = piano_channel, "explicit-piano"
            candidates = [n for n in piano if n.channel == target_channel]
        elif target_channel is not None:
            if isinstance(target_channel, bool) or not isinstance(target_channel, int) or not 0 <= target_channel <= 15:
                raise ValueError("target_channel must be a MIDI channel in [0,15], including 9 for drums")
            candidates = [n for n in notes if n.channel == target_channel and
                          (target_program is None or n.program == target_program)]
            if not candidates:
                raise ValueError("Selected target channel/program has no valid notes")
            selection = "explicit"
        elif channels:
            target_channel = max(channels, key=lambda c: (channels[c], -c))
            candidates = [n for n in piano if n.channel == target_channel]
            selection = "piano-priority"
        elif notes:
            target_channel, target_program = random.Random(seed).choice(sorted({(n.channel, n.program) for n in notes}))
            candidates = [n for n in notes if (n.channel, n.program) == (target_channel, target_program)]
            selection = "random-fallback"
        else:
            target_channel = next((c for c in range(16) if c != 9 and c not in used), None)
            candidates, selection = [], "empty"
        if target_program is None:
            counts = Counter(n.program for n in candidates)
            target_program = max(counts, key=lambda p: (counts[p], -p)) if counts else 0
        # A single roll denotes one instrument, not a mixture of changing GM
        # programs. Other programs on the same channel remain event context.
        target = [n for n in candidates if n.program == target_program]
        is_piano = target_channel != 9 and target_program in config.piano_programs
        is_drum = target_channel == 9
        pedal_enabled = config.include_pedal and is_piano
        target_orders = {n.order for n in target}
        context = [n for n in notes if n.order not in target_orders]
        target_track = min((n.track for n in target), default=source_data["tracks"])
        pedal_messages = [m for m in source_data["messages"] if pedal_enabled and m["message"]["type"] == "control_change"
                          and m["message"].get("channel") == target_channel and m["message"]["control"] == 64]
        tempo = TempoMap(source_data["ppq"], source_data["tempo_map"])
        spans = {n.order: (config.frame(n.start), max(config.frame(n.start) + 1, config.frame(n.end))) for n in notes}
        frames = max([1, config.frame(source_data["end_seconds"])] + [end for _, end in spans.values()] +
                     [config.frame(tempo.to_seconds(m["tick"])) + 1 for m in pedal_messages])
        bins = (frames + config.frames_per_bin - 1) // config.frames_per_bin
        if frames > config.max_frames or bins * 2 * 128 * config.frames_per_bin > 256_000_000:
            raise ValueError("MIDI exceeds configured duration/canvas-memory limit")
        normalized, normalization = normalize_roll_notes(target, config)
        velocity_flat = np.zeros((128, bins * config.frames_per_bin), dtype=np.uint8)
        onset_flat = np.zeros_like(velocity_flat)
        for n in normalized:
            start, end = config.frame(n.start), config.frame(n.end)
            velocity_flat[n.pitch, start:end] = n.velocity
            onset_flat[n.pitch, start] = 1
        velocity = velocity_flat.reshape(128, bins, config.frames_per_bin).transpose(1, 0, 2).copy()
        onset = onset_flat.reshape(128, bins, config.frames_per_bin).transpose(1, 0, 2).copy()
        valid_mask = (np.arange(bins * config.frames_per_bin) < frames).reshape(bins, config.frames_per_bin)
        pedal = np.zeros(bins * config.frames_per_bin, dtype=np.uint8)
        for message in pedal_messages:
            frame = config.frame(tempo.to_seconds(message["tick"]))
            pedal[frame:frames] = message["message"]["value"]
        pedal = pedal.reshape(bins, config.frames_per_bin)
        event_rows = []
        for n in context:
            start, end = spans[n.order]
            b, f = divmod(start, config.frames_per_bin)
            eb, ef = divmod(end, config.frames_per_bin)
            event_rows.append((n.track, n.channel, n.program, int(n.channel == 9), n.pitch, n.velocity, b, f, eb, ef, n.order))
        event_rows.sort(key=lambda e: (e[6], e[7], e[10]))
        events = np.array(event_rows, dtype=np.int64).reshape(-1, len(EVENT_FIELDS))
        event_ptr = np.r_[0, np.cumsum(np.bincount(events[:, 6], minlength=bins))].astype(np.int64)
        pedal_orders = {m["order"] for m in pedal_messages}
        metadata = dict(target_channel=target_channel, target_track=target_track,
                        target_program=target_program, target_is_piano=is_piano, target_is_drum=is_drum,
                        target_kind="drums" if is_drum else "piano" if is_piano else "pitched",
                        target_selection=selection, selection_seed=seed, pedal_enabled=pedal_enabled,
                        target_present=bool(target), ppq=source_data["ppq"], tempo_map=source_data["tempo_map"],
                        tracks=max(source_data["tracks"], target_track + 1),
                        messages=[m for m in source_data["messages"] if m["order"] not in pedal_orders],
                        source_notes=len(notes), source_target_notes=len(target), target_notes=len(normalized),
                        source_piano_notes=len(target) if is_piano else 0, piano_notes=len(normalized) if is_piano else 0,
                        context_notes=len(context), normalization=normalization,
                        unmatched_note_offs=source_data["unmatched_offs"],
                        skipped_nonpositive_notes=source_data["skipped_nonpositive_notes"],
                        context_short_notes_extended=sum(config.frame(n.end) <= spans[n.order][0] for n in context))
        piece = EncodedPiece(config, velocity, onset, pedal, valid_mask, np.zeros((bins, 128), dtype=np.uint8),
                             np.zeros((bins, 128), dtype=np.int32), np.zeros(bins, dtype=np.uint8), events, event_ptr, metadata)
        # Coarse summaries are derived, not a lossy/learned replacement for the
        # fine canvas. Explicit refresh also makes edited canvases consistent.
        return piece.refresh_coarse().validate()

    def decode(self, piece, *, target_only=False, piano_only=False, strict=False):
        """Decode the selected instrument plus context; piano_only is a legacy alias."""
        target_only = target_only or piano_only
        piece.validate()
        config, metadata = piece.config, piece.metadata
        scale = max(1, math.ceil(config.min_output_ppq / metadata["ppq"]))
        output_ppq = metadata["ppq"] * scale
        tempo = TempoMap(output_ppq, [(tick * scale, value) for tick, value in metadata["tempo_map"]])
        midi = mido.MidiFile(type=1, ticks_per_beat=output_ppq)
        absolute = [[] for _ in range(metadata["tracks"])]

        def put(track, tick, priority, order, message):
            absolute[track].append((tick, priority, order, message))

        for item in metadata["messages"]:
            data = item["message"]
            if target_only and "channel" in data and data["channel"] != metadata["target_channel"]:
                continue
            if data["type"] == "set_tempo" or "channel" not in data:
                tick = item["tick"] * scale
            else:
                frame = config.frame(tempo.to_seconds(item["tick"] * scale))
                tick = tempo.to_tick(frame * config.fine_ms / 1000)
            cls = mido.MetaMessage if item["is_meta"] else mido.Message
            put(item["track"], tick, 1, item["order"], cls.from_dict(data))
        channel = metadata["target_channel"]
        if channel is None and np.any(piece.velocity):
            raise ValueError("All MIDI channels are occupied; assign a dedicated target channel before generating")
        notes = roll_notes(piece.velocity, piece.onset, piece.n_frames, channel, metadata["target_track"], metadata["target_program"], strict=strict)
        if not target_only:
            f = config.frames_per_bin
            notes += [(int(e[0]), int(e[1]), int(e[2]), int(e[4]), int(e[5]), int(e[6]*f+e[7]), int(e[8]*f+e[9]), int(e[10])) for e in piece.events]
        for track, channel, program, pitch, velocity, start, end, order in notes:
            start_tick = tempo.to_tick(start * config.fine_ms / 1000)
            end_tick = max(start_tick + 1, tempo.to_tick(end * config.fine_ms / 1000))
            put(track, start_tick, 2, order, mido.Message("note_on", channel=channel, note=pitch, velocity=velocity))
            put(track, end_tick, 0, order, mido.Message("note_off", channel=channel, note=pitch, velocity=0))
        if not metadata["target_present"] and metadata["target_channel"] is not None:
            put(metadata["target_track"], 0, 1, -1, mido.Message("program_change", channel=metadata["target_channel"], program=metadata["target_program"]))
        if metadata.get("pedal_enabled", config.include_pedal) and metadata["target_channel"] is not None:
            values = piece.pedal.ravel()[:piece.n_frames]
            changes = np.flatnonzero(np.r_[values[0] != 0, values[1:] != values[:-1]])
            for frame in changes:
                put(metadata["target_track"], tempo.to_tick(int(frame) * config.fine_ms / 1000), 1, -1,
                    mido.Message("control_change", channel=metadata["target_channel"], control=64, value=int(values[frame])))
            if values[-1]:
                put(metadata["target_track"], tempo.to_tick(piece.n_frames * config.fine_ms / 1000), 1, -1,
                    mido.Message("control_change", channel=metadata["target_channel"], control=64, value=0))
        end_tick = tempo.to_tick(piece.n_frames * config.fine_ms / 1000)
        for events in absolute:
            track = mido.MidiTrack()
            previous = 0
            for tick, _, _, message in sorted(events, key=lambda e: e[:3]):
                track.append(message.copy(time=tick - previous))
                previous = tick
            track.append(mido.MetaMessage("end_of_track", time=max(0, end_tick - previous)))
            midi.tracks.append(track)
        return midi
