"""Classifier-only reader for the existing JSONL formats; no AR imports."""

import hashlib
import json


def vocabulary_digest(tokenizer):
    return hashlib.sha256(json.dumps(tokenizer.vocab, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def transpose_melodic_notes(tokens, rng, max_semitone_shift=7):
    """One bounded semitone shift for all melodic notes; never clip or shift drums.

    Expects the complete normalized sequence, BEFORE any crop, to resolve
    instruments correctly. Timing, velocity, markers and instrumentation stay
    unchanged. The input is never mutated.
    """
    melodic = []
    instrument = None
    for index, token in enumerate(tokens):
        if isinstance(token, tuple) and token[0] == "instrument":
            instrument = token[1]
        elif isinstance(token, tuple) and token[0] == "note":
            percussion = instrument == "drum" or "Drum" in str(instrument) or "Percuss" in str(instrument)
            if instrument is not None and not percussion:
                melodic.append((index, token))
    result = list(tokens)
    if melodic:
        pitches = [token[1] for _, token in melodic]
        low = max(-max_semitone_shift, -min(pitches))
        high = min(max_semitone_shift, 127 - max(pitches))
        shift = rng.randint(low, high)
        for index, token in melodic:
            result[index] = ("note", token[1] + shift, token[2])
    return result


def classifier_tokens(serialized, tokenizer):
    """Flatten clean stored notes and strip ALL metadata prefixes.

    Handles the current five-field serialized note format, flat split tokens,
    and the legacy combined-note format supported by the training reader.
    Generated predictions should use current IDs, not this clean-data reader.
    """
    tokens = []
    for item in serialized:
        if isinstance(item, list) and item and isinstance(item[0], str) and all(x == item[0] for x in item):
            tokens.append(item[0])
        elif isinstance(item, list) and len(item) == 5 and all(isinstance(x, list) and len(x) == 2 for x in item):
            categories = [x[0] for x in item]
            if categories != ["instrument", "pitch", "velocity", "onset", "dur"]:
                raise ValueError(f"Invalid serialized note field order: {categories}")
            instrument, pitch, velocity, onset, duration = [x[1] for x in item]
            if instrument in ("<P>", "<BLANK>"):
                continue
            if instrument in tokenizer.special_tokens:
                tokens.append(instrument)
            else:
                tokens.extend((("instrument", instrument), ("note", pitch, velocity),
                               ("onset", onset), ("dur", duration)))
        else:
            token = tuple(item) if isinstance(item, list) else item
            if isinstance(token, tuple) and len(token) == 3 and token[0] in tokenizer.instruments_nd:
                tokens.extend((("instrument", token[0]), ("note", token[1], token[2])))
            elif isinstance(token, tuple) and len(token) >= 2 and token[0] == "drum":
                tokens.extend((("instrument", "drum"), ("note", token[1], tokenizer.config["drum_velocity"])))
            else:
                tokens.append(token)
    tokens = [t for t in tokens if t not in (tokenizer.pad_tok, tokenizer.blank_tok)
              and not (isinstance(t, tuple) and t and t[0] == "prefix")]
    if not any(isinstance(t, tuple) and t and t[0] == "note" for t in tokens):
        raise ValueError("Labeled example contains no musical note fields")
    for token in tokens:
        if token not in tokenizer.tok_to_id:
            raise ValueError(f"Unknown classifier training token: {token!r}")
        if token in (tokenizer.mask_tok, tokenizer.sep_tok, tokenizer.unk_tok):
            raise ValueError("Classifier training expects clean music, not flow matching noise")
    return tokens
