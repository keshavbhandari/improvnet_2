# ImprovNet: classical-to-jazz/blues MIDI transfer

The current branch implements V12 melody refinement, rhythm/harmonization and
stem addition. It initializes from the pretrained AR model and keeps both
contrastive reference encoders frozen.

One shared reference window supplies style and rhythmic features. A learned
resampler combines these with drum statistics into 32 vectors, projects them to
2048 dimensions and inserts them between the genre and instrument-mix tokens.
The original decoder blocks use causal self-attention; there are no decoder
cross-attention adapters. Harmonization masks 90–100% of timing groups.

Jazz/blues sampling stays unboosted through 10k updates, then ramps to the previous
4x weight by 15k. The non-jazz candidate pool remains 2x. Batch size is unchanged.
Inference defaults to a reproducibly random reference window; optional
`--reference-aggregation average` averages all eligible reference excerpts.

See [the current training/inference guide](docs/melody_refinement_plan.md) for
sequence construction, prompt history, pitch augmentation, safety boundaries,
commands, sampling and validation. The sole config is
`improvnet/model/melody_config.py`; the launcher is `run_scripts/melody_launch.sh`.
