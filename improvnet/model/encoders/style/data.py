"""Style-positive pairs from solo and mixture views of the same piece."""
from improvnet.model.encoders.data import PiecePairDataset


class StylePairDataset(PiecePairDataset):
    """Piece-level style pairs; views may come from different time windows."""
