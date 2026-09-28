from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn


def make_mlp(
    dimensions: Iterable[int],
    *,
    dropout: float = 0.0,
    final_activation: bool = False,
) -> nn.Sequential:
    dims = list(dimensions)
    layers: list[nn.Module] = []
    for index, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:], strict=False)):
        layers.append(nn.Linear(input_dim, output_dim))
        is_last = index == len(dims) - 2
        if not is_last or final_activation:
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def pairwise_iou_cxcylogwh(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Pairwise IoU for normalized cx, cy, log(w), log(h) boxes."""
    center_a, size_a = boxes_a[..., :2], boxes_a[..., 2:].exp().clamp(max=2.0)
    center_b, size_b = boxes_b[..., :2], boxes_b[..., 2:].exp().clamp(max=2.0)
    min_a, max_a = center_a - size_a / 2, center_a + size_a / 2
    min_b, max_b = center_b - size_b / 2, center_b + size_b / 2
    intersection = (torch.minimum(max_a, max_b) - torch.maximum(min_a, min_b)).clamp_min(0)
    intersection_area = intersection[..., 0] * intersection[..., 1]
    area_a = size_a[..., 0] * size_a[..., 1]
    area_b = size_b[..., 0] * size_b[..., 1]
    return intersection_area / (area_a + area_b - intersection_area).clamp_min(1e-6)


class SparseInteractionLayer(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        relation_dim = 9  # delta box, delta velocity and IoU
        pair_dim = 2 * latent_dim + relation_dim
        self.message = make_mlp([pair_dim, hidden_dim, latent_dim], dropout=dropout)
        self.gate = make_mlp([pair_dim, hidden_dim, 1], dropout=dropout)
        self.update = make_mlp([2 * latent_dim, hidden_dim, latent_dim], dropout=dropout)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        hidden: Tensor,
        boxes: Tensor,
        velocity: Tensor,
        neighbor_indices: Tensor,
        neighbor_mask: Tensor,
    ) -> Tensor:
        batch, num_objects, latent_dim = hidden.shape
        neighbors = neighbor_indices.shape[-1]
        batch_indices = torch.arange(batch, device=hidden.device)[:, None, None]
        target = hidden[batch_indices, neighbor_indices]
        target_boxes = boxes[batch_indices, neighbor_indices]
        target_velocity = velocity[batch_indices, neighbor_indices]
        source = hidden.unsqueeze(2).expand(-1, -1, neighbors, -1)
        source_boxes = boxes.unsqueeze(2).expand(-1, -1, neighbors, -1)
        source_velocity = velocity.unsqueeze(2).expand(-1, -1, neighbors, -1)
        delta_box = target_boxes - source_boxes
        delta_velocity = target_velocity - source_velocity
        iou = pairwise_iou_cxcylogwh(source_boxes, target_boxes).unsqueeze(-1)
        pair = torch.cat((source, target, delta_box, delta_velocity, iou), dim=-1)

        messages = self.message(pair)
        gate_logits = self.gate(pair).squeeze(-1).masked_fill(~neighbor_mask, -1e4)
        weights = gate_logits.softmax(dim=-1) * neighbor_mask.to(hidden.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        aggregate = (weights.unsqueeze(-1) * messages).sum(dim=2)
        update = self.update(torch.cat((hidden, aggregate), dim=-1))
        return self.norm(hidden + update).reshape(-1, num_objects, latent_dim)


class SparseInteractionGraph(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        layers: int,
        num_neighbors: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors
        self.layers = nn.ModuleList(
            SparseInteractionLayer(latent_dim, hidden_dim, dropout) for _ in range(layers)
        )

    def _neighbors(self, boxes: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        batch, num_objects, _ = boxes.shape
        centers = boxes[..., :2]
        distances = torch.cdist(centers, centers)
        candidate = valid.unsqueeze(1) & valid.unsqueeze(2)
        eye = torch.eye(num_objects, dtype=torch.bool, device=boxes.device).unsqueeze(0)
        candidate = candidate & ~eye
        distances = distances.masked_fill(~candidate, float("inf"))
        k = min(self.num_neighbors, num_objects - 1)
        values, indices = distances.topk(k=k, dim=-1, largest=False)
        return indices, values.isfinite()

    def forward(self, hidden: Tensor, boxes: Tensor, velocity: Tensor, valid: Tensor) -> Tensor:
        if hidden.shape[1] <= 1 or not self.layers:
            return hidden * valid.unsqueeze(-1)
        neighbor_indices, neighbor_mask = self._neighbors(boxes, valid)
        for layer in self.layers:
            hidden = layer(
                hidden,
                boxes,
                velocity,
                neighbor_indices,
                neighbor_mask,
            )
            hidden = hidden * valid.unsqueeze(-1)
        return hidden


class StructuredTransitionCell(nn.Module):
    def __init__(self, latent_dim: int, camera_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        input_dim = 2 * latent_dim + camera_dim
        self.gate = make_mlp([input_dim, hidden_dim, latent_dim], dropout=dropout)
        self.candidate = make_mlp([input_dim, hidden_dim, latent_dim], dropout=dropout)
        self.acceleration = make_mlp([input_dim, hidden_dim, 4], dropout=dropout)
        self.damping = make_mlp([input_dim, hidden_dim, 4], dropout=dropout)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        memory: Tensor,
        interaction: Tensor,
        camera: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        camera = camera.unsqueeze(1).expand(-1, memory.shape[1], -1)
        inputs = torch.cat((memory, interaction, camera), dim=-1)
        gate = self.gate(inputs).sigmoid()
        candidate = self.candidate(inputs).tanh()
        next_memory = self.norm(gate * memory + (1.0 - gate) * candidate)
        acceleration = 0.25 * self.acceleration(inputs).tanh()
        damping = self.damping(inputs).sigmoid()
        return next_memory, acceleration, damping
