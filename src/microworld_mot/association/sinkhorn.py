from __future__ import annotations

import torch
from torch import Tensor


def log_sinkhorn(logits: Tensor, iterations: int = 8, temperature: float = 0.10) -> Tensor:
    """Differentiable approximately doubly-stochastic assignment probabilities."""
    log_prob = logits / max(temperature, 1e-4)
    for _ in range(iterations):
        log_prob = log_prob - torch.logsumexp(log_prob, dim=-1, keepdim=True)
        log_prob = log_prob - torch.logsumexp(log_prob, dim=-2, keepdim=True)
    return log_prob


def sinkhorn_loss(logits: Tensor, target: Tensor, valid_pairs: Tensor | None = None) -> Tensor:
    log_prob = log_sinkhorn(logits)
    if valid_pairs is not None:
        target = target * valid_pairs.to(target.dtype)
    normalizer = target.sum().clamp_min(1.0)
    return -(target * log_prob).sum() / normalizer

