from __future__ import annotations

import torch

from ..types import WorldPrediction


class UncertaintyScheduler:
    def __init__(self, threshold: float = 0.55, max_gap: int = 4) -> None:
        self.threshold = threshold
        self.max_gap = max_gap
        self.frames_since_detector = max_gap

    def reset(self) -> None:
        self.frames_since_detector = self.max_gap

    def update(self, detector_ran: bool) -> None:
        self.frames_since_detector = 0 if detector_ran else self.frames_since_detector + 1

    def should_refresh(self, prediction: WorldPrediction | None) -> bool:
        if prediction is None or self.frames_since_detector >= self.max_gap:
            return True
        valid = prediction.state.valid
        if not bool(valid.any()):
            return True
        refresh = prediction.refresh_logits.sigmoid()
        uncertainty = prediction.uncertainty
        risk = torch.maximum(refresh, (uncertainty / 0.05).clamp(0, 1))
        return bool(risk[valid].max() > self.threshold)

