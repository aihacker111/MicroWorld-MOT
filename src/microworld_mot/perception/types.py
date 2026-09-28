from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class DetectorOutput:
    boxes_xyxy: Tensor
    scores: Tensor
    labels: Tensor
    features: Tensor
