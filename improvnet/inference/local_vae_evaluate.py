"""Small, collection-balanced held-out test of reconstruction and latent health."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
from types import SimpleNamespace

import mido
import numpy as np
import torch

from improvnet.inference.local_vae_inference import (
    LatentStats, Scores, common_args, load_model, output_directory, reconstruct, write_report,
)
from improvnet.utils.local_vae_data import LocalVAEDataset, seed_for


def readiness(metrics, latent, sampled_f1, zero_f1, files, bins, sampled_onset_f1=None):
    """Conservative smoke-test rubric, NOT a learned/validated diffusion predictor."""
    reasons = []
    if files < 20 or bins < 128:
        return dict(status="insufficient_evidence", reasons=["Use at least 20 files and 128 bins"])
    if ((metrics["activity_f1"] or 0) < .9 or (metrics["notes_f1"] or 0) < .85
            or (metrics.get("onset_f1", 1) or 0) < .85 or (metrics["velocity_mae"] or 0) > 10):
        reasons.append("Reconstruction remains a bottleneck")
    if metrics.get("offset_mae_ms") is not None and metrics["offset_mae_ms"] > 100:
        reasons.append("Note-release timing errors are large")
    if metrics.get("boundary_f1") is not None and metrics["boundary_f1"] < .85:
        reasons.append("Activity near patch boundaries needs review")
    if latent["active_channel_fraction"] < .5:
        reasons.append("Many latent channels have very little variation")
    if sampled_f1 is None or sampled_f1 < (metrics["activity_f1"] or 0) - .1:
        reasons.append("Posterior sampling noticeably harms reconstruction")
    if sampled_onset_f1 is not None and sampled_onset_f1 < (metrics.get("onset_f1") or 0) - .1:
        reasons.append("Posterior sampling noticeably harms onsets")
    if zero_f1 is None or (metrics["activity_f1"] or 0) - zero_f1 < .1:
        reasons.append("Zeroing latents does not sufficiently harm activity reconstruction")
    return dict(status="needs_review" if reasons else "promising_not_proven",
                reasons=reasons or ["Try a small diffusion pilot; these checks do not guarantee good generation"])


def candidates(dataset, seed):
    """Round-robin collections, unique group IDs; no repeated piece/file inflation."""
    rng = random.Random(seed)
    pools = {}
    for name, ids in dataset.groups.items():
        pools[name] = ids.tolist()
        rng.shuffle(pools[name])
    seen = set()
    while any(pools.values()):
        for name in sorted(pools):
            if not pools[name]:
                continue
            row = dataset.connect().execute("SELECT id,path,group_id FROM files WHERE id=?", (pools[name].pop(),)).fetchone()
            if row[2] not in seen:
                seen.add(row[2])
                yield row[0], name, Path(row[1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    common_args(parser)
    parser.add_argument("--index", type=Path, help="Defaults to checkpoint's held-out MIDI index")
    parser.add_argument("--num-files", type=int, default=100)
    parser.add_argument("--bins-per-file", type=int, default=8, help="Random contiguous window, including silence")
    parser.add_argument("--save-examples", type=int, default=3, help="Save first N full reconstructed pieces; zero to disable")
    args = parser.parse_args(argv)
    if args.num_files < 1 or args.bins_per_file < 1 or args.save_examples < 0:
        parser.error("File/bin counts must be positive, example count nonnegative")
    device = torch.device(args.device)
    model, tokenizer, step, saved = load_model(args.checkpoint, device)
    config = SimpleNamespace(**saved)
    if args.index:
        config.INDEX_PATH = str(args.index)
    dataset = LocalVAEDataset(config, "test")  # Read-only index, same drum-only subset as training.
    output = output_directory(args.output_dir, "test-")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    total, sampled_total, zero_total = Scores(), Scores(), Scores()
    latent = LatentStats()
    failures, collections = Counter(), {}
    example_sources = []
    files = bins = examples = 0
    try:
        for attempt, (row_id, collection, path) in enumerate(candidates(dataset, args.seed)):
            if files >= args.num_files or attempt >= args.num_files * 5:
                break
            try:
                piece = tokenizer.encode(path, recover=True, seed=seed_for(config.SEED, 0, row_id))
                if not piece.metadata["target_present"]:
                    raise ValueError("No valid target notes")
            except (ValueError, OSError, EOFError, IndexError, KeyError, mido.KeySignatureError) as exc:
                # Bounded category counts, not thousands of per-file log entries.
                failures[str(exc)[:100]] += 1
                continue
            length = min(args.bins_per_file, len(piece.velocity))
            rng = random.Random(seed_for(args.seed, row_id, "window"))
            start = rng.randrange(len(piece.velocity) - length + 1)
            ids = np.arange(start, start + length)
            predicted, stats, sampled, zero = reconstruct(model, piece, indices=ids, diagnostics=True,
                batch_size=args.batch_size, activity_threshold=args.activity_threshold,
                onset_threshold=args.onset_threshold, generator=generator)
            score = Scores(); score.arrays(piece, predicted, ids); score.notes(piece, predicted, indices=ids)
            total.add(score)
            sampled_total.arrays(piece, sampled, ids); zero_total.arrays(piece, zero, ids)
            if latent.total is None:
                latent = stats
            else:
                latent.total += stats.total; latent.squares += stats.squares; latent.noise += stats.noise
                latent.weight += stats.weight; latent.kl_sum += stats.kl_sum; latent.bins += stats.bins
            item = collections.setdefault(collection, {"files": 0, "scores": Scores()})
            item["files"] += 1; item["scores"].add(score)
            files += 1; bins += length
            if examples < args.save_examples:
                # Reconstruct ALL bins for listening, not just the scored window.
                full, _, _, _ = reconstruct(model, piece, batch_size=args.batch_size,
                    activity_threshold=args.activity_threshold, onset_threshold=args.onset_threshold)
                tokenizer.decode(piece, target_only=True).save(output / f"{examples:02d}_reference_target.mid")
                tokenizer.decode(full, target_only=True).save(output / f"{examples:02d}_reconstructed_target.mid")
                tokenizer.decode(full).save(output / f"{examples:02d}_reconstructed.mid")
                example_sources.append(dict(prefix=f"{examples:02d}", input=str(path), collection=collection))
                examples += 1
            print(f"{files}/{args.num_files} {collection}: activity F1={score.report()['activity_f1']}", flush=True)
    finally:
        if dataset.connection is not None:
            dataset.connection.close()
    if not files:
        raise RuntimeError(f"No usable test files; skips={dict(failures)}")
    metrics, latent_metrics = total.report(), latent.report()
    sampled_f1, zero_f1 = sampled_total.report()["activity_f1"], zero_total.report()["activity_f1"]
    report = dict(checkpoint=str(args.checkpoint.resolve()), step=step, seed=args.seed, files=files, bins=bins,
                  thresholds=dict(activity=args.activity_threshold, onset=args.onset_threshold), skipped=sum(failures.values()),
                  errors=metrics, latent=latent_metrics,
                  checks=dict(sampled_activity_f1=sampled_f1, zero_latent_activity_f1=zero_f1,
                              sampled_onset_f1=sampled_total.report()["onset_f1"]),
                  readiness=readiness(metrics, latent_metrics, sampled_f1, zero_f1, files, bins,
                                      sampled_total.report()["onset_f1"]),
                  collections={k: dict(files=v["files"], activity_f1=v["scores"].report()["activity_f1"],
                                       notes_f1=v["scores"].report()["notes_f1"]) for k, v in collections.items()},
                  skip_reasons=dict(failures.most_common(5)))
    if example_sources:
        report["examples"] = example_sources
    write_report(output / "report.json", report)
    print(json.dumps({"errors": metrics, "readiness": report["readiness"]}))
    print(f"Saved: {output}")
    return report


if __name__ == "__main__":
    main()
