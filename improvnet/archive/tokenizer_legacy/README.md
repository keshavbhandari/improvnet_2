# Archived event tokenization

The previous absolute event tokenizer, base, configuration and MIDI helpers were
moved here on branch `hierarchical-pianoroll-tokenizer`. Their original import
paths are thin compatibility wrappers so old AR/flow checkpoints, utilities
and metrics remain usable. New experiments use `improvnet.tokenizer.hierarchical`.

The original `master` branch remains unchanged. No vocabulary or saved checkpoint
has been migrated to the new representation.
