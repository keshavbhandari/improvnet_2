"""Graded syntax diagnostics plus strict MIDI validity for no-pedal tokens.

The recovering scanner counts faults; it NEVER repairs or compiles malformed
predictions. Strict validity still requires the original sequence to round-trip
through MIDI. Open canvases can defer a legal unfinished event to the next
block, without claiming that the unfinished sequence compiles already.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
import math
from numbers import Integral
from typing import Any

import mido

from improvnet.tokenizer.absolute import AbsTokenizer


@dataclass(frozen=True)
class SyntaxIssue:
    code: str
    message: str
    token_index: int | None = None
    token: Any = None


@dataclass(frozen=True)
class MidiSyntaxResult:
    valid: bool
    syntax_valid: bool
    # None: not attempted (invalid content/grammar, or legal but unfinished tail).
    midi_compiles: bool | None
    score: float
    error_count: int
    issues: tuple[SyntaxIssue, ...]
    note_events: int
    sounding_note_events: int
    num_tokens: int
    scored_tokens: int
    noise_level: float | None = None
    # Canvas-prefix validity is distinct from strict, already-compilable validity.
    prefix_valid: bool = False
    pending_tail_fields: tuple[str, ...] = ()
    boundary_issues: tuple[SyntaxIssue, ...] = ()

    @property
    def validity_score(self) -> int:
        return int(self.valid)

    @property
    def issue(self) -> SyntaxIssue | None:
        """First diagnostic, retained for callers using the previous API."""
        return self.issues[0] if self.issues else None


class _InvalidToken(Exception):
    def __init__(self, issue):
        self.issue = issue


def _items(values, name):
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    if hasattr(values, "ndim") and values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; use evaluate_batch for batches")
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of tokens or token IDs")
    return list(values)


def _valid_positions(values, valid_mask):
    if valid_mask is None:
        return values
    mask = _items(valid_mask, "valid_mask")
    if len(mask) != len(values) or any(x not in (False, True, 0, 1) for x in mask):
        raise ValueError("valid_mask must match the sequence and contain boolean values")
    mask = [bool(x) for x in mask]
    if any(not a and b for a, b in zip(mask, mask[1:])):
        raise ValueError("valid_mask must describe right padding, not holes within events")
    return values[:sum(mask)]


def _noise_level(value):
    if value is not None:
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("noise_level must be finite and in [0, 1]")
    return value


def _kind(token):
    if isinstance(token, tuple) and token and isinstance(token[0], str):
        return token[0]
    return token if isinstance(token, str) else None


def _positive(token, index):
    return (isinstance(token, tuple) and len(token) > index
            and isinstance(token[index], Integral) and not isinstance(token[index], bool)
            and token[index] > 0)


class MidiSyntaxMetric:
    """Error-density score and an independent strict compile/validity flag.

    Score = max(0, 1 - error_count / scored_tokens). Each faulty actual token
    counts once even if several rules flag it; each missing required field or
    wrapper counts once. Empty/nonmusical output and actual compiler failures
    score zero. For a canvas, only CANVAS length is the denominator: clean
    history cannot dilute its errors. All joined-context faults are counted.
    Only a legal final event prefix can be deferred for an open canvas; its
    diagnostics are retained separately rather than counted as errors.
    """

    def __init__(self, tokenizer: AbsTokenizer | None = None, *,
                 min_sounding_notes: int = 1, check_time_order: bool = False):
        self.tokenizer = tokenizer or AbsTokenizer()
        if self.tokenizer.include_pedal:
            raise ValueError("MidiSyntaxMetric uses the current no-pedal tokenizer")
        if not isinstance(min_sounding_notes, Integral) or isinstance(min_sounding_notes, bool) or min_sounding_notes < 0:
            raise ValueError("min_sounding_notes must be a nonnegative integer")
        self.min_sounding_notes = int(min_sounding_notes)
        self.check_time_order = check_time_order

    def _token(self, value, index):
        def fail(code, message):
            raise _InvalidToken(SyntaxIssue(code, message, index, value))
        if isinstance(value, Integral) and not isinstance(value, bool):
            if int(value) not in self.tokenizer.id_to_tok:
                fail("unknown_token_id", "ID is outside this tokenizer's vocabulary")
            value = self.tokenizer.id_to_tok[int(value)]
        elif isinstance(value, list):
            value = tuple(value)
        if not isinstance(value, (str, tuple)):
            fail("invalid_token", "Expected a token tuple, string or integer ID")
        if isinstance(value, tuple):
            if any(isinstance(x, bool) for x in value):
                fail("invalid_token", "Boolean values are not musical fields")
            if _kind(value) in ("note", "onset", "dur") and any(not isinstance(x, Integral) for x in value[1:]):
                fail("invalid_token", "Pitch, velocity and timing fields must be integers")
            value = tuple(int(x) if isinstance(x, Integral) else x for x in value)
        try:
            known = value in self.tokenizer.tok_to_id
        except TypeError:
            known = False
        if not known:
            fail("out_of_vocabulary", "Token is outside this tokenizer's vocabulary")
        return value

    def _scan(self, values, require_bos, require_eos):
        t = self.tokenizer
        tokens, known, issues, units, events = [], [], [], set(), []
        diagnostic_keys = set()

        def fault(code, message, index=None, token=None, missing_key=None):
            issue = SyntaxIssue(code, message, index, token)
            key = (code, index, message)
            if key not in diagnostic_keys:
                issues.append(issue)
                diagnostic_keys.add(key)
            units.add(("token", index) if missing_key is None else ("missing", missing_key))

        for index, value in enumerate(values):
            try:
                tokens.append(self._token(value, index))
                known.append(True)
            except _InvalidToken as error:
                tokens.append(tuple(value) if isinstance(value, list) else value)
                known.append(False)
                issue = error.issue
                fault(issue.code, issue.message, index, issue.token)
        seen_bos = seen_eos = seen_dim = seen_delimiter = body = False
        last_onset, segment, index = -1, 0, 0
        boundaries = {t.bos_tok, t.eos_tok, t.time_tok, t.dim_tok, t.delimiter_tok, "instrument", "prefix"}
        while index < len(tokens):
            token, role = tokens[index], _kind(tokens[index])
            if seen_eos:
                fault("tokens_after_end", "Valid tokens occur after <E>", index, token)
                index += 1
                continue
            if role == "prefix":
                if body or seen_bos:
                    fault("prefix_position", "Metadata prefix must precede <S> and events", index, token)
                index += 1
                continue
            if token == t.bos_tok:
                if seen_bos or body:
                    fault("start_position", "<S> must occur once before the body", index, token)
                seen_bos = True
                index += 1
                continue
            if require_bos and not seen_bos:
                fault("missing_start", "Complete sequence requires <S>", missing_key="start")
            body = True
            if token == t.eos_tok:
                seen_eos = True
                index += 1
            elif token == t.time_tok:
                segment += 1
                index += 1
            elif token == t.dim_tok:
                if seen_dim:
                    fault("duplicate_dim", "Only one <D> is allowed", index, token)
                seen_dim = True
                index += 1
            elif t.include_delimiter and token == t.delimiter_tok:
                if seen_delimiter:
                    fault("duplicate_delimiter", "Only one delimiter is allowed", index, token)
                seen_delimiter = True
                index += 1
            elif role == "instrument":
                start, before = index, len(units)
                fields = {}
                index += 1
                for expected in ("note", "onset", "dur"):
                    while index < len(tokens):
                        current_role = _kind(tokens[index])
                        if current_role == expected:
                            fields[expected] = index
                            index += 1
                            break
                        if current_role in boundaries or current_role in {"onset", "dur"}:
                            fault("event_order", f"Expected {expected} field", index, tokens[index], (start, expected))
                            break
                        fault("event_order", f"Unexpected field while expecting {expected}", index, tokens[index])
                        index += 1
                    else:
                        fault("incomplete_event", f"Event ends before its {expected} field", index,
                              missing_key=(start, expected))
                complete = len(fields) == 3 and all(known[p] for p in (start, *fields.values())) and len(units) == before
                if "onset" in fields and known[fields["onset"]]:
                    absolute = segment * t.abs_time_step_ms + tokens[fields["onset"]][1]
                    if self.check_time_order and absolute < last_onset:
                        fault("time_order", "Event onsets move backwards", fields["onset"], tokens[fields["onset"]])
                        complete = False
                    last_onset = absolute
                events.append((start, index, fields, complete))
            else:
                code = "unresolved_token" if token in (t.pad_tok, t.mask_tok, t.blank_tok, t.sep_tok, t.unk_tok) else "event_order"
                fault(code, "Expected instrument or structural token at event boundary", index, token)
                index += 1
        if require_bos and not seen_bos:
            fault("missing_start", "Complete sequence requires <S>", missing_key="start")
        if require_eos and not seen_eos:
            fault("missing_end", "Complete sequence requires final <E>", len(tokens), missing_key="end")
        return tokens, events, issues, units

    def _evaluate(self, values, *, require_bos, require_eos, noise_level, score_range=None,
                  allow_incomplete_tail=False):
        noise_level = _noise_level(noise_level)
        tokens, events, issues, units = self._scan(values, require_bos, require_eos)
        start, end = score_range if score_range is not None else (0, len(tokens))
        scored = end - start
        pending, boundary_issues = (), []
        # Defer ONLY an exact, known-vocabulary prefix of the final note event.
        # A malformed tail, an internal missing field, or an event terminated by
        # <E>/<T>/<D> cannot be fixed simply by appending the next block.
        if allow_incomplete_tail and score_range is not None and end == len(tokens) and events:
            left, right, fields, _ = events[-1]
            roles = tuple(_kind(token) for token in tokens[left:])
            legal_prefixes = (("instrument",), ("instrument", "note"),
                              ("instrument", "note", "onset"))
            if (roles in legal_prefixes and left < end and right > start
                    and self.tokenizer.eos_tok not in tokens
                    and list(fields) == list(roles[1:])):
                try:
                    for index in range(left, len(tokens)):
                        self._token(tokens[index], index)
                except _InvalidToken:
                    pass
                else:
                    pending = ("note", "onset", "dur")[len(roles) - 1:]
                    boundary_issues = [issue for issue in issues
                                       if issue.code == "incomplete_event" and issue.token_index == len(tokens)]
                    issues = [issue for issue in issues if issue not in boundary_issues]
                    for expected in pending:
                        units.discard(("missing", (left, expected)))
        complete = [fields for _, _, fields, ok in events if ok]
        sounding = sum(_positive(tokens[f["note"]], 2) and _positive(tokens[f["dur"]], 1) for f in complete)
        syntax_valid = not issues
        content_ok = True
        if self.min_sounding_notes:
            has_fields = any(_kind(token) in ("instrument", "note", "onset", "dur") for token in tokens[start:end])
            candidates = 0
            for left, right, fields, _ in events:
                if not has_fields or left >= end or right <= start:
                    continue
                note = tokens[fields["note"]] if "note" in fields else None
                dur = tokens[fields["dur"]] if "dur" in fields else None
                if (note is None or _positive(note, 2)) and (dur is None or _positive(dur, 1)):
                    candidates += 1
            if not candidates and has_fields:
                candidates = sum(_kind(token) == "note" and _positive(token, 2) for token in tokens[start:end])
                durations = [token for token in tokens[start:end] if _kind(token) == "dur"]
                if durations and not any(_positive(token, 1) for token in durations):
                    candidates = 0
            terminal = tokens[start:end] == [self.tokenizer.eos_tok] and sounding >= self.min_sounding_notes
            content_ok = scored > 0 and (candidates >= self.min_sounding_notes or terminal)
            pending_note = tokens[events[-1][2]["note"]] if pending and "note" in events[-1][2] else None
            potential_sounding = int(pending_note is not None and _positive(pending_note, 2))
            if syntax_valid and sounding + potential_sounding < self.min_sounding_notes:
                content_ok = False
            if not content_ok:
                code = "empty_canvas" if score_range is not None else "insufficient_notes"
                issues.append(SyntaxIssue(code, "Output must contain sounding music, not empty/all-structural tokens"))
                units.add(("content", None))
        # A reward-quality guard, not a tokenizer grammar rule: consecutive
        # time advances can encode genuine rests. Penalize only generated
        # pairs (including a pair spanning the history/canvas boundary).
        consecutive_time = []
        for index in range(len(tokens) - 1):
            if tokens[index] == self.tokenizer.time_tok and tokens[index + 1] == self.tokenizer.time_tok:
                if index < end and index + 2 > start:
                    consecutive_time.append(index)
        quality_ok = not consecutive_time
        for index in consecutive_time:
            issues.append(SyntaxIssue("consecutive_time_tokens", "Two consecutive <T> tokens fail the generation-quality check", index, tokens[index]))
            units.add(("time_run", index))
        compiles = None
        if syntax_valid and content_ok and not pending:
            try:
                decoded = self.tokenizer.detokenize(tokens)
                if len(decoded.note_msgs) != len(complete):
                    raise ValueError("Detokenizer discarded validated note events")
                midi = decoded.to_midi()
                buffer = BytesIO()
                midi.save(file=buffer)
                buffer.seek(0)
                parsed = mido.MidiFile(file=buffer)
                actual = sum(msg.type == "note_on" and msg.velocity > 0 for track in parsed.tracks for msg in track)
                expected = sum(_positive(tokens[f["note"]], 2) for f in complete)
                if actual != expected:
                    raise ValueError("MIDI serialization changed the number of positive note-on events")
                compiles = True
            except Exception as error:
                compiles = False
                issues.append(SyntaxIssue("midi_compilation", f"{type(error).__name__}: {error}"))
                units.add(("compiler", None))
        valid = syntax_valid and content_ok and quality_ok and compiles is True
        prefix_valid = valid or (bool(pending) and syntax_valid and content_ok and quality_ok)
        score = max(0., 1. - len(units) / max(scored, 1)) if content_ok and quality_ok and compiles is not False else 0.
        return MidiSyntaxResult(valid=valid, syntax_valid=syntax_valid, midi_compiles=compiles,
                                score=score, error_count=len(units), issues=tuple(issues),
                                note_events=len(complete), sounding_note_events=sounding,
                                num_tokens=len(tokens), scored_tokens=scored, noise_level=noise_level,
                                prefix_valid=prefix_valid, pending_tail_fields=pending,
                                boundary_issues=tuple(boundary_issues))

    def evaluate_sequence(self, tokens_or_ids, *, valid_mask=None, require_bos=True,
                          require_eos=True, noise_level=None) -> MidiSyntaxResult:
        values = _valid_positions(_items(tokens_or_ids, "tokens_or_ids"), valid_mask)
        return self._evaluate(values, require_bos=require_bos, require_eos=require_eos, noise_level=noise_level)

    def evaluate_canvas(self, canvas, *, history=(), boundary_suffix=(), valid_mask=None,
                        allow_cropped_history=False, allow_incomplete_tail=True,
                        noise_level=None) -> MidiSyntaxResult:
        """Score an open block; a legal trailing event prefix is not an error.

        ``prefix_valid`` accepts a tail that the next block can complete.
        ``valid`` still requires actual MIDI serialization of the full input;
        it stays false and ``midi_compiles`` stays None while fields are pending.
        Use ``allow_incomplete_tail=False`` for a closed final canvas. An <E>
        always closes the sequence, so an incomplete event before it still fails.
        """
        generated = _valid_positions(_items(canvas, "canvas"), valid_mask)
        prefix, suffix = _items(history, "history"), _items(boundary_suffix, "boundary_suffix")
        if allow_cropped_history and prefix:
            try:
                roles = [_kind(self._token(value, index)) for index, value in enumerate(prefix)]
            except _InvalidToken:
                roles = []
            for fragment in (("note", "onset", "dur"), ("onset", "dur"), ("dur",)):
                if roles[:len(fragment)] == list(fragment):
                    prefix = prefix[len(fragment):]
                    break
        roles = tuple(_kind(tuple(value) if isinstance(value, list) else self.tokenizer.id_to_tok.get(int(value), value)
                            if isinstance(value, Integral) and not isinstance(value, bool) else value) for value in suffix)
        if roles not in ((), ("note", "onset", "dur"), ("onset", "dur"), ("dur",)):
            raise ValueError("boundary_suffix must contain only fields completing a cut note event")
        return self._evaluate(prefix + generated + suffix, require_bos=False, require_eos=False,
                              noise_level=noise_level, score_range=(len(prefix), len(prefix) + len(generated)),
                              allow_incomplete_tail=allow_incomplete_tail)

    def evaluate_batch(self, sequences, *, valid_masks=None, histories=None,
                       boundary_suffixes=None, noise_levels=None, canvas=False,
                       allow_cropped_history=False, allow_incomplete_tail=True) -> list[MidiSyntaxResult]:
        if hasattr(sequences, "detach"):
            sequences = sequences.detach().cpu().tolist()
        rows = list(sequences)
        for name, value in (("valid_masks", valid_masks), ("histories", histories),
                            ("boundary_suffixes", boundary_suffixes), ("noise_levels", noise_levels)):
            if value is not None and len(value) != len(rows):
                raise ValueError(f"{name} must have one entry per sequence")
        if not canvas and (histories is not None or boundary_suffixes is not None):
            raise ValueError("histories/boundary_suffixes require canvas=True")
        results = []
        for index, row in enumerate(rows):
            kwargs = {"valid_mask": None if valid_masks is None else valid_masks[index],
                      "noise_level": None if noise_levels is None else noise_levels[index]}
            if canvas:
                kwargs.update(history=() if histories is None else histories[index],
                              boundary_suffix=() if boundary_suffixes is None else boundary_suffixes[index],
                              allow_cropped_history=allow_cropped_history,
                              allow_incomplete_tail=allow_incomplete_tail)
            results.append((self.evaluate_canvas if canvas else self.evaluate_sequence)(row, **kwargs))
        return results


def summarize_syntax(results, noise_bins=(0., .25, .5, .75, 1.)):
    results = list(results)
    edges = tuple(float(x) for x in noise_bins)
    if len(edges) < 2 or edges[0] != 0 or edges[-1] != 1 or any(not math.isfinite(x) for x in edges) or any(a >= b for a, b in zip(edges, edges[1:])):
        raise ValueError("noise_bins must strictly increase from zero to one")

    def rates(group):
        return {"count": len(group), "valid_count": sum(r.validity_score for r in group),
                "valid_rate": sum(r.validity_score for r in group) / len(group) if group else None,
                "prefix_valid_count": sum(r.prefix_valid for r in group),
                "prefix_valid_rate": sum(r.prefix_valid for r in group) / len(group) if group else None,
                "pending_tail_count": sum(bool(r.pending_tail_fields) for r in group),
                "mean_score": sum(r.score for r in group) / len(group) if group else None,
                "mean_errors": sum(r.error_count for r in group) / len(group) if group else None}
    summary = {**rates(results), "failures": dict(Counter(r.issue.code for r in results if r.issue)),
               "issue_counts": dict(Counter(issue.code for r in results for issue in r.issues)), "noise_bins": []}
    for index, (low, high) in enumerate(zip(edges, edges[1:])):
        group = [r for r in results if r.noise_level is not None and low <= r.noise_level
                 and (r.noise_level < high or index == len(edges) - 2 and r.noise_level <= high)]
        summary["noise_bins"].append({"low": low, "high": high, **rates(group)})
    return summary
