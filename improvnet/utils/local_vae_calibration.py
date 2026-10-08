"""Streaming joint activity/onset threshold selection; no stored piano rolls.

Onsets are gated by activity, exactly as in Reconstruction.tensors(). Each
branch keeps two tiny score histograms: reference onset/inactive-onset cells.
Padding is excluded. Distributed ranks sum counts, never average local F1s.
"""
import math

import torch
import torch.distributed as dist


class ThresholdCalibrator:
    def __init__(self, activity, onset, *, defaults=(.5, .9), min_recall=.99,
                 branches=("clean", "jitter"), device="cpu"):
        self.defaults = tuple(defaults)
        if not branches or len(set(branches)) != len(branches):
            raise ValueError("Calibration needs distinct branches")
        for values in (activity, onset, defaults):
            if not values or any(not math.isfinite(v) or not 0 < v < 1 for v in values):
                raise ValueError("Calibration thresholds must be finite and in (0,1)")
        if len(defaults) != 2 or not math.isfinite(min_recall) or not 0 <= min_recall <= 1:
            raise ValueError("Invalid calibration defaults/recall floor")
        self.activity = sorted(set([*activity, defaults[0]]))
        self.onset = sorted(set([*onset, defaults[1]]))
        self.min_recall, self.branches = min_recall, tuple(branches)
        a, o = len(self.activity) + 1, len(self.onset) + 1
        self.counts = torch.zeros((len(branches), 2, a, o), dtype=torch.int64, device=device)
        self.active_counts = torch.zeros((len(branches), 2, a), dtype=torch.int64, device=device)
        self._activity = torch.tensor(self.activity, device=device, dtype=torch.float32)
        self._onset = torch.tensor(self.onset, device=device, dtype=torch.float32)

    @torch.no_grad()
    def update(self, reconstruction, batch, branch="clean"):
        index = self.branches.index(branch)
        valid = batch["valid_mask"][:, None].expand_as(batch["roll"][:, 0])
        a = reconstruction.activity_logits.detach().float().sigmoid()[valid]
        o = reconstruction.onset_logits.detach().float().sigmoid()[valid]
        if not bool(torch.isfinite(a).all() & torch.isfinite(o).all()):
            raise ValueError("Non-finite calibration probabilities")
        # right=True includes probabilities equal to the chosen threshold (>=).
        ai = torch.bucketize(a.contiguous(), self._activity, right=True)
        oi = torch.bucketize(o.contiguous(), self._onset, right=True)
        active = (batch["roll"][:, 0][valid] > 0).long()
        onset = (batch["roll"][:, 1][valid] > 0).long()
        na, no = len(self.activity) + 1, len(self.onset) + 1
        self.counts[index] += torch.bincount(onset * na * no + ai * no + oi,
                                            minlength=2 * na * no).reshape(2, na, no)
        self.active_counts[index] += torch.bincount(active * na + ai, minlength=2 * na).reshape(2, na)

    @staticmethod
    def _metrics(tp, fp, support):
        # Empty targets cannot certify high reconstruction: handled in select().
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, support)
        f1 = 2 * tp / max(1, tp + fp + support)
        return dict(precision=precision, recall=recall, f1=f1)

    def select(self, *, distributed=True):
        if distributed and dist.is_initialized():
            dist.all_reduce(self.counts)
            dist.all_reduce(self.active_counts)
        # Reverse cumulative sums turn histograms into all >= threshold counts.
        joint = self.counts.flip((-2, -1)).cumsum(-2).cumsum(-1).flip((-2, -1)).cpu()
        active = self.active_counts.flip(-1).cumsum(-1).flip(-1).cpu()
        supports = self.counts[:, 1].sum((-2, -1)).cpu().tolist()
        active_supports = self.active_counts[:, 1].sum(-1).cpu().tolist()
        records = []
        for i, a in enumerate(self.activity):
            for j, o in enumerate(self.onset):
                scores = {}
                for b, name in enumerate(self.branches):
                    scores[name] = dict(
                        onset=self._metrics(int(joint[b, 1, i + 1, j + 1]),
                                            int(joint[b, 0, i + 1, j + 1]), supports[b]),
                        activity=self._metrics(int(active[b, 1, i + 1]),
                                               int(active[b, 0, i + 1]), active_supports[b]))
                supported = all(v > 0 for v in supports + active_supports)
                floor_met = supported and all(scores[b][k]["recall"] >= self.min_recall
                                              for b in self.branches for k in ("activity", "onset"))
                score = min((s["onset"]["f1"] + s["activity"]["f1"]) / 2 for s in scores.values())
                precision = min(s[k]["precision"] for s in scores.values() for k in ("onset", "activity"))
                distance = abs(a - self.defaults[0]) + abs(o - self.defaults[1])
                records.append(dict(activity_threshold=a, onset_threshold=o, recall_floor_met=floor_met,
                                    score=score, scores=scores, rank=(score, precision, -distance)))
        feasible = [r for r in records if r["recall_floor_met"]]
        if not all(v > 0 for v in supports + active_supports):
            selected = next(r for r in records if (r["activity_threshold"], r["onset_threshold"]) == self.defaults)
            selected["status"] = "insufficient_positive_targets"
        else:
            selected = max(feasible or records, key=lambda r: r["rank"])
            selected["status"] = "recall_floor_met" if feasible else "recall_floor_unattainable"
        selected.pop("rank")
        selected.update(min_recall=self.min_recall, branches=list(self.branches),
                        objective="worst_branch_mean_activity_onset_f1")
        return selected


def calibration_scalars(result):
    values = {"calibration/activity_threshold": result["activity_threshold"],
              "calibration/onset_threshold": result["onset_threshold"],
              "calibration/recall_floor_met": float(result["recall_floor_met"]),
              "calibration/score": result["score"]}
    for branch, scores in result["scores"].items():
        for kind, metrics in scores.items():
            values.update({f"calibration/{branch}_{kind}_{key}": value for key, value in metrics.items()})
    return values
