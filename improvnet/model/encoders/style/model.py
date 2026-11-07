"""Style encoder: its own weights with the common bidirectional backbone."""
from improvnet.model.encoders.model import ContinuityConfig as EncoderConfig, ContinuityEncoder


class StyleEncoder(ContinuityEncoder):
    """Learn cross-instrument musical style compatibility.

    Inheriting the shared backbone keeps parameter names checkpoint-compatible.
    Style and beat models are separately initialized and trained.
    """
