# ImprovNet: classical-to-jazz/blues MIDI transfer

The current branch implements V14 melody refinement, rhythm/harmonization and
stem addition. It initializes from the pretrained AR model and keeps both
contrastive reference encoders frozen.

Two consecutive nearby reference windows supply style and rhythmic features.
Each produces its own 32-token memory bank, including that window's drum
statistics. The 64 tokens are projected to decoder width 2048 and placed between
the genre and instrument-mix tokens. Original AR decoder blocks stay unchanged.
Training uses a consistent pitch shift across each encoder's reference pair.

Refinement inputs keep their pitches but use velocity 90 in training and
inference; clean targets and history retain expressive velocities. Refinement
now also masks 90–100% of onset/duration groups and uses two reference windows
matched exactly to the melody's output instruments. Harmonization keeps the same
severe timing-mask policy. `--rhythm-strength` controls masking in both stages. Jazz/blues sampling stays unboosted
through 10k updates, then ramps to 4x by 15k; the non-jazz pool remains 2x.
Batch size is unchanged. Inference defaults to a random consecutive reference
pair; `--reference-aggregation average` averages matching positions across pairs.

Choose references using `--reference-description slow` (or a combination such as
`"slow balanced"`) or `--style-reference /path/to/trio.mid`. The catalogue contains
only pieces with exactly one piano, drums and double bass, with no additional
instruments. It currently has 129 jazz references; no exact-trio blues pieces
were found in the training pool. Tempo descriptors are `slow`,
`moderate`, `fast`; piano-activity descriptors are `sparse`, `balanced`, `dense`.
With neither selection option, `any` is used. Reference IDs and embedding buckets
are no longer selection options.

See [the current training/inference guide](docs/melody_refinement_plan.md) for
sequence construction, prompt history, pitch augmentation, safety boundaries,
commands, sampling and validation. The sole config is
`improvnet/model/melody_config.py`; the launcher is `run_scripts/melody_launch.sh`.

The working tree has been restored to `97a0e0a` (`v14 best version added`).
Use the existing `melody_refinement_v14/best_model.pt`; refinement input `<T>`
tokens are retained. No retraining is needed for the inference changes below.

Reference tempo stays unchanged by default. Add `--reference-tempo source` to
retime its notes to the source's estimated pulse **before** reference encoding.
The only modes are `original` and `source`; manual BPM is not a CLI option. Both consecutive windows,
both encoders and the drum profiles use the retimed music. Automatic estimates
can have half/double-tempo ambiguity; this is soft conditioning, not an output
BPM guarantee. Source and original on-disk reference files remain untouched.
The run folder includes `reference.mid` (original file copy) and, when matching
is enabled, `reference_tempo.mid` (the retimed conditioning notes). Both are
listed in `report.json`. If only prepared notes are available for a catalogue
entry, the original is reconstructed and this is identified in the report.

Decoding now uses request-owned, preallocated KV buffers and records
`generation_seconds` and `tokens_per_second` for each chunk in `report.json`.
The GH200 checkpoint benchmark produced identical logits and sampled notes,
but no reliable throughput gain from preallocation alone (about 48 tokens/s
for either strategy in the constrained-decoder test). Training was not submitted.

An initial-pipeline scheduling bug has been fixed: idle downstream stages no
longer repeatedly wake one another. A controlled V14 refinement replay with
three idle stages improved from 3.05 to 37.61 tokens/s with identical notes.
Restart inference to apply the fix. The temporary diagnostics were removed after validation.
