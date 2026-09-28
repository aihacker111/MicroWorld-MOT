from __future__ import annotations

import math

import torch
from torch import Tensor

from .types import WorldPrediction


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(values.dtype).expand_as(values)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def gaussian_mixture_nll(prediction: WorldPrediction, target: Tensor, mask: Tensor) -> Tensor:
    target = target.unsqueeze(-2)
    inverse_variance = prediction.mixture_log_scales.mul(-2).exp()
    component_log_prob = -0.5 * (
        (target - prediction.mixture_means).square() * inverse_variance
        + 2 * prediction.mixture_log_scales
        + math.log(2 * math.pi)
    ).sum(dim=-1)
    log_prob = torch.logsumexp(
        prediction.mixture_logits.log_softmax(dim=-1) + component_log_prob,
        dim=-1,
    )
    return masked_mean(-log_prob, mask)


def box_iou_aligned(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    center_a, size_a = boxes_a[..., :2], boxes_a[..., 2:].exp().clamp(max=2)
    center_b, size_b = boxes_b[..., :2], boxes_b[..., 2:].exp().clamp(max=2)
    min_a, max_a = center_a - size_a / 2, center_a + size_a / 2
    min_b, max_b = center_b - size_b / 2, center_b + size_b / 2
    intersection = (torch.minimum(max_a, max_b) - torch.maximum(min_a, min_b)).clamp_min(0)
    intersection_area = intersection[..., 0] * intersection[..., 1]
    area_a = size_a[..., 0] * size_a[..., 1]
    area_b = size_b[..., 0] * size_b[..., 1]
    return intersection_area / (area_a + area_b - intersection_area).clamp_min(1e-6)

