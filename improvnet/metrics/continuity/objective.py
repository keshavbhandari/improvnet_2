"""Symmetric InfoNCE (cross-entropy), global negatives with autograd gathering."""

import math
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from torch.nn import functional as F


def contrastive_loss(a, b, piece_ids, *, temperature=.1, distributed=False):
    if a.ndim != 2 or a.shape != b.shape or piece_ids.shape != (len(a),) or len(a) < 1:
        raise ValueError("Need matching nonempty embedding batches and piece IDs")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Contrastive temperature must be finite and positive")
    a, b = F.normalize(a.float(), dim=-1), F.normalize(b.float(), dim=-1)
    if distributed and dist.is_initialized():
        # All ranks need equal batches. Differentiable gather sends candidate
        # gradients back to their owning rank; DDP averages model gradients.
        aa, bb = torch.cat(all_gather(a)), torch.cat(all_gather(b))
        ids = [torch.empty_like(piece_ids) for _ in range(dist.get_world_size())]
        dist.all_gather(ids, piece_ids)
        ids = torch.cat(ids)
        offset = dist.get_rank() * len(a)
    else:
        aa, bb, ids, offset = a, b, piece_ids, 0
    targets = torch.arange(len(a), device=a.device) + offset
    same_piece = piece_ids[:, None] == ids[None, :]
    blocked = same_piece.clone()
    blocked[torch.arange(len(a), device=a.device), targets] = False
    negatives = ~same_piece
    usable = negatives.any(1)
    # A duplicate-only batch has no contrastive signal. Fail rather than report
    # zero loss/100% accuracy as success. Every rank participates in the check.
    count = usable.sum()
    total = count.clone()
    if distributed and dist.is_initialized():
        dist.all_reduce(total)
    if not total.item():
        raise ValueError("Contrastive batch needs at least two different pieces")
    ab, ba = a @ bb.T, b @ aa.T
    logits_ab, logits_ba = (ab / temperature).masked_fill(blocked, -torch.inf), (ba / temperature).masked_fill(blocked, -torch.inf)
    row_loss = (F.cross_entropy(logits_ab, targets, reduction="none") + F.cross_entropy(logits_ba, targets, reduction="none")) / 2
    # Scale local loss so DDP's rank average is a global usable-anchor mean.
    world = dist.get_world_size() if distributed and dist.is_initialized() else 1
    loss = row_loss[usable].sum() * world / total
    with torch.no_grad():
        diagonal = torch.arange(len(a), device=a.device)
        correct = ((logits_ab.argmax(1) == targets).float() + (logits_ba.argmax(1) == targets).float()) / 2
        positive = (ab[diagonal, targets] + ba[diagonal, targets]) / 2
        hardest = (ab.masked_fill(~negatives, -torch.inf).max(1).values + ba.masked_fill(~negatives, -torch.inf).max(1).values) / 2
        stats = torch.stack((row_loss[usable].sum(), correct[usable].sum(), positive[usable].sum(),
                             (ab[negatives].sum() + ba[negatives].sum()) / 2,
                             negatives.sum().float(), (positive[usable] - hardest[usable]).sum(), count.float()))
    return loss, stats


def summarize(stats):
    values = stats.detach().cpu().tolist()
    if not values[6] or not values[4]:
        raise ValueError("No contrastive examples/negatives evaluated")
    return {"loss": values[0] / values[6], "retrieval_accuracy": values[1] / values[6],
            "positive_similarity": values[2] / values[6], "negative_similarity": values[3] / values[4],
            "hard_negative_margin": values[5] / values[6], "pairs": int(values[6])}
