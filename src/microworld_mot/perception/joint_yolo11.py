from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..types import WorldPrediction


def _first_conv_in_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return child.in_channels
    raise RuntimeError("Could not infer YOLO detection feature channels")


class JointYOLO11(nn.Module):
    """Trainable YOLO11 bridge with world priors and differentiable ROI features.

    The official Ultralytics detection model owns the box/classification loss.
    A pre-hook on its Detect head injects the causal world prior and captures
    the same FPN tensors for the tracking appearance path.
    """

    def __init__(
        self,
        weights: str,
        observation_dim: int,
        roi_size: int = 3,
        world_prior_strength: float = 0.10,
        *,
        detector_model: nn.Module | None = None,
        feature_channels: list[int] | None = None,
    ) -> None:
        super().__init__()
        if detector_model is None:
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise RuntimeError(
                    "Joint training requires Ultralytics. Install with "
                    "`python -m pip install -e '.[perception]'`."
                ) from error
            detector_model = YOLO(weights).model
        self.detector = detector_model
        self.weights = weights
        if not hasattr(self.detector, "model") or not len(self.detector.model):
            raise TypeError("Expected an Ultralytics DetectionModel-compatible module")
        self.detect_head = self.detector.model[-1]
        if feature_channels is None:
            branches = getattr(self.detect_head, "cv2", None)
            if branches is None:
                raise RuntimeError("Unsupported YOLO head: missing cv2 branches")
            feature_channels = [_first_conv_in_channels(branch) for branch in branches]
        self.observation_dim = observation_dim
        self.roi_size = roi_size
        self.world_gates = nn.Parameter(
            torch.full((len(feature_channels),), float(world_prior_strength))
        )
        self.roi_projections = nn.ModuleList(
            nn.Conv2d(channels, observation_dim, kernel_size=1)
            for channels in feature_channels
        )
        self._features: list[Tensor] = []
        self._world_prior: WorldPrediction | None = None
        self._hook_handle = self.detect_head.register_forward_pre_hook(self._detect_pre_hook)

    def clear_world_prior(self) -> None:
        self._world_prior = None

    def set_world_prior(self, prediction: WorldPrediction) -> None:
        self._world_prior = prediction

    def _render_prior(self, feature: Tensor) -> Tensor:
        if self._world_prior is None:
            return feature.new_zeros(feature.shape[0], 1, feature.shape[-2], feature.shape[-1])
        state = self._world_prior.state
        height, width = feature.shape[-2:]
        y = torch.linspace(0.0, 1.0, height, device=feature.device, dtype=feature.dtype)
        x = torch.linspace(0.0, 1.0, width, device=feature.device, dtype=feature.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        center = state.boxes[..., :2].to(feature.dtype)
        size = state.boxes[..., 2:].exp().clamp(1e-3, 2.0).to(feature.dtype)
        variance = state.log_variance[..., :2].exp().clamp_min(1e-8).to(feature.dtype)
        sigma = 0.35 * size + variance.sqrt()
        dx = (xx[None, None] - center[..., 0, None, None]) / sigma[..., 0, None, None]
        dy = (yy[None, None] - center[..., 1, None, None]) / sigma[..., 1, None, None]
        confidence = (state.existence * state.valid).to(feature.dtype)[..., None, None]
        heat = torch.exp(-0.5 * (dx.square() + dy.square())) * confidence
        return heat.amax(dim=1, keepdim=True)

    def _detect_pre_hook(self, _module: nn.Module, inputs: tuple[Any, ...]):
        if not inputs or not isinstance(inputs[0], list | tuple):
            raise RuntimeError("Unexpected Ultralytics Detect-head input")
        features = list(inputs[0])
        if len(features) != len(self.roi_projections):
            raise RuntimeError(
                f"YOLO returned {len(features)} FPN scales; expected {len(self.roi_projections)}"
            )
        fused = [
            feature + torch.tanh(self.world_gates[index]) * self._render_prior(feature)
            for index, feature in enumerate(features)
        ]
        self._features = fused
        return (fused, *inputs[1:])

    @staticmethod
    def _roi_pool(feature: Tensor, boxes_xywh: Tensor, output_size: int) -> Tensor:
        batch, objects, _ = boxes_xywh.shape
        offsets = torch.linspace(
            -0.5,
            0.5,
            output_size,
            device=feature.device,
            dtype=feature.dtype,
        )
        pooled: list[Tensor] = []
        for batch_index in range(batch):
            boxes = boxes_xywh[batch_index].to(feature.dtype)
            grid_y, grid_x = torch.meshgrid(offsets, offsets, indexing="ij")
            x = boxes[:, 0, None, None] + grid_x * boxes[:, 2, None, None]
            y = boxes[:, 1, None, None] + grid_y * boxes[:, 3, None, None]
            grid = torch.stack((2.0 * x - 1.0, 2.0 * y - 1.0), dim=-1)
            expanded = feature[batch_index : batch_index + 1].expand(objects, -1, -1, -1)
            sampled = F.grid_sample(
                expanded,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            pooled.append(sampled.mean(dim=(-1, -2)))
        return torch.stack(pooled)

    def encode_rois(self, boxes_xywh: Tensor, valid: Tensor) -> Tensor:
        if not self._features:
            raise RuntimeError("YOLO forward must run before encode_rois")
        encoded = [
            self._roi_pool(projection(feature), boxes_xywh, self.roi_size)
            for projection, feature in zip(
                self.roi_projections, self._features, strict=True
            )
        ]
        appearance = torch.stack(encoded, dim=0).mean(dim=0)
        appearance = F.normalize(appearance, dim=-1)
        return appearance * valid.unsqueeze(-1)

    def detector_loss(self, images: Tensor, targets: dict[str, Tensor]) -> Tensor:
        result = self.detector({"img": images, **targets})
        loss = result[0] if isinstance(result, tuple) else result
        if not isinstance(loss, Tensor):
            raise RuntimeError("Ultralytics detector did not return a Tensor loss")
        return loss.mean()

    def extract_features(self, images: Tensor) -> None:
        self.detector(images)
        if not self._features:
            raise RuntimeError("YOLO feature hook did not run")

    def parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
        head_ids = {id(parameter) for parameter in self.detect_head.parameters()}
        detector_head = [
            parameter for parameter in self.detector.parameters() if id(parameter) in head_ids
        ]
        detector_backbone = [
            parameter for parameter in self.detector.parameters() if id(parameter) not in head_ids
        ]
        perception = list(self.roi_projections.parameters()) + [self.world_gates]
        return detector_backbone, detector_head, perception

    def extra_repr(self) -> str:
        return f"weights={self.weights}, observation_dim={self.observation_dim}"
