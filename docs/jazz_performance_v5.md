# Additional jazz performance MIDI for V5

Raw data and author licenses are retained under
`/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/jazz_blues_data/raw`.
`acquisition_manifest.json` records 1,446 downloaded performance records,
source URLs, hashes/ZIP members, metadata and quality flags. Audio was not
needed for this training preparation.

## JazzSAMBA

[JazzSAMBA 1.0.0](https://zenodo.org/records/22943963) supplies recorded piano MIDI
and, for asynchronous performances, recorded drum MIDI. The download includes
100 piano/drum paired takes and 52 piano takes across 76 standards. Official
song splits and alternate-take groupings are retained. See the
[author documentation](https://pnlong.github.io/jazzsamba-demo/index.html).

Only recorded MIDI becomes clean labels. The separately downloaded automatic
bass/wind estimates are excluded from training mixtures. Piano/drum tracks are
merged by absolute performed seconds with their separate tempo maps, preserving
pitches, velocity, pedal and percussion channel assignment. Ten real takes with
70,508 note-on/off events were compared against the originals; maximum onset/
release conversion error was 0.250 ms, below the model's 10-ms resolution.

The raw 152 takes yield 151 unique base records: one identical synchronous piano
take is omitted. All 100 paired piano/drum records remain, including 80 for
training, 10 validation and 10 test. The complete source contributes 120 training,
15 validation and 16 test records. The dataset's original-recordings license
permits use/copy/modification/distribution with its copyright notice; the exact
license is retained at `raw/jazzsamba/LICENSE`.

## Jazz Trio Database

[JTD v0.2](https://github.com/HuwCheston/Jazz-Trio-Database/releases/tag/v02-zenodo)
adds 1,294 piano MIDI transcriptions of performed recordings. These are automatic
transcriptions rather than directly captured keyboard MIDI. Bass/drum onset
annotations do not constitute full bass/drum note tracks and are not invented
as clean labels. The annotations/metadata use MIT; commercial audio rights are
separate. See the [dataset structure](https://huwcheston.github.io/Jazz-Trio-Database/installation/database-structure.html).

Because JTD and existing corpora can refer to the same recordings with different
transcriptions, byte hashes alone are insufficient. Effective existing
validation/test membership is read from the preserved V4 manifest, including
split overrides. A conservative audit excludes JTD entries when a musician's
surname matches a known held-out ARIA composer/performer label, a title matches
held-out non-ARIA music, or a known PiJAMA performer/title pair matches.

This admits 507 JTD files, **training only**, and excludes 787 while keeping all
raw files. The detailed decisions are in `raw/jtd_training_admission_v5.json`;
`raw/jtd_train_paths_v5.json` lists admitted paths. Metadata screening is not a
proof that all recording duplicates are removed: 303 held-out ARIA files lack
usable composer/performer labels, and spelling/pseudonyms/excerpts can escape
matches. Surname/title matches can also conservatively exclude distinct music.
No new JTD validation/test set is claimed.

## Prepared membership and reproducibility

`raw/training_acquisition_v5.json` selects the recorded JazzSAMBA takes and
screened JTD files. `prepared/jazz_performance_v5.jsonl` contains 658 parsed base
records, with no parsing errors: 151 JazzSAMBA and 507 JTD. Their index records
full-melody fingerprints, file hashes, instrument classes and locked source
splits. This is not an expanded set of fixed training windows.

`melody_manifest_v5.json` combines all 658 new records with the preserved
original cohort; no additional legacy hash/fingerprint copies were found.

| V5 prepared cohort | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| All prepared files | 70,752 | 728 | 727 |
| Jazz/blues | 24,173 | 254 | 252 |
| Newly added files | 627 | 15 | 16 |

The main/stem sampler requests 483,460 examples per epoch. All original split
assignments were checked unchanged. A cached manifest recheck took 1.75 seconds;
launching does not parse the expanded original corpus again. Real new JazzSAMBA
records passed refinement, harmonization and stem construction; real JTD records
passed refinement and harmonization, with all source/response/history limits and
clean drum labels checked. Details are in `_provenance/v5/readiness.json` and
`actual_sequence_checks.json`.

The focused V5 suite passed 195 checks across six feature modules. AR initialization shapes, launcher syntax and real source/target
budgets also passed preflight. The pre-kit job `2431554` was stopped and archived for a fresh kit-only run.
All 100 recorded piano/drum pairs remain eligible (80 train, 10 validation,
10 test). On-the-fly normalization preserves 9,619 extended hi-hat articulations
and collapses 22 coincident normalized attacks; all remaining drum hits use the
21 allowed kit sound IDs. Raw files and source split indexes are unchanged.
Replacement job `2432602` loaded the original AR weights with a fresh optimizer
and completed its first training update on four GH200 GPUs.
Existing-performance copies are screened against hashes and complete
melody/timing fingerprints. Distinct new drum takes are retained even if their
piano part is identical; alternate takes stay in one official song split.

Acquisition code: `improvnet/scripts/melody_download_jazz.py`. Preparation code:
`improvnet/scripts/melody_supplement.py` and `melody_manifest.py`. Audit scripts
and preserved-index hashes are retained in `_provenance/v5/`. The original V4
manifest, base index and recovered-data index are checked for changes.

```bash
python -m improvnet.scripts.melody_supplement \
  --manifest /e/scratch/e-dev-2026d09-047/bhandari1/improvnet/jazz_blues_data/raw/training_acquisition_v5.json
python -m improvnet.scripts.melody_manifest
```

These commands only prepare data. They do not submit training. The source-read
cache, dynamic skyline, corruptions, windows and histories are still used by the
V5 DataLoader. JazzSAMBA piano/drum pairs directly supply the stem task; JTD
piano primarily supplies refinement and harmony/rhythm.
