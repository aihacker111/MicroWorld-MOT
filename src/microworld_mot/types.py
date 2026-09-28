from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor


@dataclass
class WorldState:
    """A padded batch of object beliefs.

    Shapes use ``[batch, objects, channels]``. Boxes are normalized
    ``(center_x, center_y, log_width, log_height)``.
    """

    boxes: Tensor
    velocity: Tensor
    appearance: Tensor
    memory: Tensor
    valid: Tensor
    existence: Tensor
    occlusion: Tensor
    log_variance: Tensor

    def to(self, device: torch.device | str) -> WorldState:
        values = {field.name: getattr(self, field.name).to(device) for field in fields(self)}
        return WorldState(**values)

    def detach(self) -> WorldState:
        values = {field.name: getattr(self, field.name).detach() for field in fields(self)}
        return WorldState(**values)

    def clone(self) -> WorldState:
        values = {field.name: getattr(self, field.name).clone() for field in fields(self)}
        return WorldState(**values)

    @property
    def batch_size(self) -> int:
        return int(self.boxes.shape[0])

    @property
    def num_objects(self) -> int:
        return int(self.boxes.shape[1])


@dataclass
class WorldPrediction:
    state: WorldState
    mixture_logits: Tensor
    mixture_means: Tensor
    mixture_log_scales: Tensor
    refresh_logits: Tensor
    acceleration: Tensor
    latent_prediction: Tensor | None = None

    @property
    def expected_boxes(self) -> Tensor:
        weights = self.mixture_logits.softmax(dim=-1).unsqueeze(-1)
        return (weights * self.mixture_means).sum(dim=-2)

    @property
    def uncertainty(self) -> Tensor:
        weights = self.mixture_logits.softmax(dim=-1).unsqueeze(-1)
        mean = self.expected_boxes.unsqueeze(-2)
        aleatoric = self.mixture_log_scales.mul(2).exp()
        epistemic = (self.mixture_means - mean).square()
        return (weights * (aleatoric + epistemic)).sum(dim=(-2, -1))


@dataclass
class Detections:
    boxes: Tensor
    scores: Tensor
    appearance: Tensor
    labels: Tensor | None = None

    def to(self, device: torch.device | str) -> Detections:
        return Detections(
            boxes=self.boxes.to(device),
            scores=self.scores.to(device),
            appearance=self.appearance.to(device),
            labels=None if self.labels is None else self.labels.to(device),
        )


@dataclass
class TrackOutput:
    track_id: int
    box: Tensor
    score: float
    existence: float
    occlusion: float
