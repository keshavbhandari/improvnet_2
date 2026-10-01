"""Natural-distribution classifier metrics and held-out temperature fitting."""

import math

import torch
import torch.distributed as dist
from torch.nn import functional as F


def class_weights(counts, mode="none"):
    counts = torch.as_tensor(counts, dtype=torch.float32)
    if counts.ndim != 1 or not len(counts) or bool((counts <= 0).any()):
        raise ValueError("Every classifier class must have positive train support")
    if mode not in ("none", "sqrt", "inverse"):
        raise ValueError("Unknown class balancing mode")
    weights = torch.ones_like(counts) if mode == "none" else counts.pow(-.5 if mode == "sqrt" else -1.)
    return weights / weights.mean()


def classification_metrics(confusion, labels, *, focus_genres=("jazz", "blues")):
    """Multiclass scores; group accuracy selects true-label rows only.

    Group F1 averages one-vs-rest genre F1, retaining errors across groups.
    Per-genre accuracy is recall (correct predictions / true genre examples).
    Empty groups have undefined scores rather than misleading zeros.
    """
    confusion = torch.as_tensor(confusion, dtype=torch.float64)
    if confusion.shape != (len(labels), len(labels)) or bool((confusion < 0).any()):
        raise ValueError("Need a nonnegative class-by-class confusion matrix")
    support, predicted = confusion.sum(1), confusion.sum(0)
    precision = confusion.diag() / predicted.clamp_min(1)
    recall = confusion.diag() / support.clamp_min(1)
    f1 = 2 * confusion.diag() / (support + predicted).clamp_min(1)
    present = support > 0
    focus = set(focus_genres)
    groups = {}
    for name, indices in (("all", list(range(len(labels)))),
                          ("jazz_blues", [i for i, label in enumerate(labels) if label in focus]),
                          ("remaining", [i for i, label in enumerate(labels) if label not in focus])):
        total = support[indices].sum()
        supported = [i for i in indices if present[i]]
        groups[name] = {
            "support": int(total),
            "accuracy": float(confusion.diag()[indices].sum() / total) if total > 0 else None,
            "macro_f1": float(f1[supported].mean()) if supported else None,
        }
    return {"accuracy": float(confusion.diag().sum() / support.sum()) if bool(present.any()) else None,
            "macro_f1": float(f1[present].mean()) if bool(present.any()) else None,
            "groups": groups,
            "per_genre": {label: {"support": int(support[i]), "precision": float(precision[i]),
                                  "recall": float(recall[i]), "f1": float(f1[i]),
                                  "accuracy": float(recall[i]) if support[i] > 0 else None} for i, label in enumerate(labels)}}


@torch.inference_mode()
def evaluate(model, loader, labels, device, *, prior_correction=None, max_batches=0, collect_logits=False,
             focus_genres=("jazz", "blues")):
    # Unequal validation shards cannot use DDP forwards (buffer collectives).
    model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    was_training = model.training
    model.eval()
    classes = len(labels)
    confusion = torch.zeros(classes, classes, dtype=torch.long)
    losses, count, all_logits, all_labels = 0., 0, [], []
    correction = torch.zeros(classes, device=device) if prior_correction is None else prior_correction.to(device)
    try:
        for index, batch in enumerate(loader):
            if max_batches and index >= max_batches:
                break
            ids, mask, targets = [batch[key].to(device) for key in ("ids", "attention_mask", "labels")]
            dtype = torch.bfloat16 if device.type != "cuda" or torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                logits = model(ids, mask)
            logits = logits.float() - correction
            losses += float(F.cross_entropy(logits, targets, reduction="sum"))
            count += len(targets)
            predicted = logits.argmax(-1)
            confusion += torch.bincount((targets * classes + predicted).cpu(), minlength=classes**2).view(classes, classes)
            if collect_logits:
                all_logits.append(logits.cpu())
                all_labels.append(targets.cpu())
        if dist.is_initialized():
            totals = torch.tensor([losses, count], dtype=torch.float64, device=device)
            matrix = confusion.to(device)
            dist.all_reduce(totals)
            dist.all_reduce(matrix)
            losses, count = totals.tolist()
            count = int(count)
            confusion = matrix.cpu()
        if not count:
            raise ValueError("No validation examples were evaluated")
        result = {**classification_metrics(confusion, labels, focus_genres=focus_genres), "nll": losses / count, "count": count}
        if collect_logits:
            logits = torch.cat(all_logits) if all_logits else torch.empty(0, classes)
            targets = torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.long)
            if dist.is_initialized():
                shards = [None] * dist.get_world_size()
                dist.all_gather_object(shards, (logits, targets))
                logits = torch.cat([shard[0] for shard in shards])
                targets = torch.cat([shard[1] for shard in shards])
            return result, logits, targets
        return result
    finally:
        model.train(was_training)


def fit_temperature(logits, targets, *, iterations=50):
    # evaluate() returns inference tensors. Clone outside inference mode before
    # the temperature optimizer's autograd pass needs to save logits/labels.
    logits, targets = logits.detach().float().cpu().clone(), targets.detach().long().cpu().clone()
    if logits.ndim != 2 or targets.shape != logits.shape[:1] or not len(targets) or logits.shape[1] < 2:
        raise ValueError("Need nonempty held-out logits [N,C] and labels [N]")
    if not bool(torch.isfinite(logits).all()) or bool(((targets < 0) | (targets >= logits.shape[1])).any()):
        raise ValueError("Invalid calibration logits/labels")
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], max_iter=iterations, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_temperature.clamp(math.log(.05), math.log(20.)).exp(), targets)
        loss.backward()
        return loss
    before = float(F.cross_entropy(logits, targets))
    optimizer.step(closure)
    temperature = float(log_temperature.detach().clamp(math.log(.05), math.log(20.)).exp())
    after = float(F.cross_entropy(logits / temperature, targets))
    if not math.isfinite(after) or after > before:
        temperature, after = 1., before
    return {"temperature": temperature, "nll_before": before, "nll_after": after, "count": len(targets)}
