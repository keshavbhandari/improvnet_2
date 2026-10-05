"""Values only. Heuristic defaults, not perceptually calibrated thresholds."""

EXACT_PITCH_WEIGHT = 0.15
PITCH_CLASS_WEIGHT = 0.60
SMOOTH_CHROMA_WEIGHT = 0.25
SMOOTH_CHROMA_SIGMA = 0.75

MATCH_BASELINE = 0.25
INSERT_OPEN = 0.25
INSERT_EXTEND = 0.08
DELETE_OPEN = 0.35
DELETE_EXTEND = 0.15
MIN_MATCH_SIMILARITY = 0.10
RETENTION_BETA = 2.0  # Favor reference coverage, but still penalize excessive output.
PATTERN_LENGTHS = (3, 4, 5)
MAX_PATTERN_INSERTIONS = 4

IDENTITY_WEIGHT = 0.70
PATTERN_WEIGHT = 0.20
STRUCTURE_WEIGHT = 0.10
ALLOW_GLOBAL_TRANSPOSITION = False
MAX_TRANSPOSE = 12
MAX_NOTES = 2048  # Bound quadratic alignment memory; never silently truncate.
