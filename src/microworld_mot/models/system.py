from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from ..config import ModelConfig
from ..types import WorldPrediction, WorldState
from .components import make_mlp, pairwise_iou_cxcylogwh


class ObservationCorrector(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        input_dim = config.latent_dim + 4 + 4 + config.appearance_dim + 1
        hidden = config.graph_hidden_dim
        self.gain = make_mlp([input_dim, hidden, 8], dropout=config.dropout)
        self.memory_update = make_mlp(
            [input_dim, hidden, config.latent_dim], dropout=config.dropout
        )
        self.appearance_gate = make_mlp([input_dim, hidden, 1], dropout=config.dropout)
        self.norm = nn.LayerNorm(config.latent_dim)

    def forward(
        self,
        predicted: WorldState,
        observation_boxes: Tensor,
        observation_appearance: Tensor,
        observation_scores: Tensor,
        observation_mask: Tensor,
    ) -> WorldState:
        innovation = observation_boxes - predicted.boxes
        inputs = torch.cat(
            (
                predicted.memory,
                innovation,
                predicted.log_variance,
                observation_appearance,
                observation_scores.unsqueeze(-1),
            ),
            dim=-1,
        )
        box_gain, velocity_gain = self.gain(inputs).sigmoid().chunk(2, dim=-1)
        mask = observation_mask.unsqueeze(-1)
        boxes = torch.where(mask, predicted.boxes + box_gain * innovation, predicted.boxes)
        velocity = torch.where(
            mask,
            predicted.velocity + velocity_gain * innovation,
            predicted.velocity,
        )
        appearance_gate = self.appearance_gate(inputs).sigmoid()
        appearance = torch.where(
            mask,
            appearance_gate * predicted.appearance + (1 - appearance_gate) * observation_appearance,
            predicted.appearance,
        )
        memory_delta = self.memory_update(inputs)
        memory = torch.where(mask, self.norm(predicted.memory + memory_delta), predicted.memory)
        existence = torch.where(observation_mask, torch.maximum(predicted.existence, observation_scores), predicted.existence)
        occlusion = torch.where(observation_mask, predicted.occlusion * 0.25, predicted.occlusion)
        measurement_variance = (0.02 * (1.0 - observation_scores) + 1e-4).square().unsqueeze(-1)
        variance = torch.where(mask, predicted.log_variance.exp().minimum(measurement_variance), predicted.log_variance.exp())
        return WorldState(
            boxes=boxes,
            velocity=velocity,
            appearance=appearance,
            memory=memory,
            valid=predicted.valid,
            existence=existence,
            occlusion=occlusion,
            log_variance=variance.clamp_min(1e-8).log(),
        )


class AssociationHead(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        pair_dim = 9
        self.residual_score = make_mlp(
            [pair_dim, config.graph_hidden_dim, config.graph_hidden_dim, 1],
            dropout=config.dropout,
        )

    def forward(
        self,
        prediction: WorldState,
        detection_boxes: Tensor,
        detection_appearance: Tensor,
        detection_scores: Tensor,
        detection_valid: Tensor | None = None,
    ) -> Tensor:
        # prediction: [B, N, *], detections: [B, M, *]
        delta = detection_boxes.unsqueeze(1) - prediction.boxes.unsqueeze(2)
        variance = prediction.log_variance.exp().unsqueeze(2).clamp_min(1e-6)
        mahalanobis = (delta.square() / variance).mean(dim=-1, keepdim=True)
        pred_app = torch.nn.functional.normalize(prediction.appearance, dim=-1)
        det_app = torch.nn.functional.normalize(detection_appearance, dim=-1)
        cosine = torch.einsum("bna,bma->bnm", pred_app, det_app).unsqueeze(-1)
        score = detection_scores.unsqueeze(1).unsqueeze(-1).expand(-1, prediction.num_objects, -1, -1)
        existence = prediction.existence.unsqueeze(-1).unsqueeze(-1).expand_as(score)
        occlusion = prediction.occlusion.unsqueeze(-1).unsqueeze(-1).expand_as(score)
        features = torch.cat((delta, mahalanobis, cosine, score, existence, occlusion), dim=-1)
        learned = self.residual_score(features).squeeze(-1)
        logits = learned - 0.25 * mahalanobis.squeeze(-1) + cosine.squeeze(-1)
        valid_pairs = prediction.valid.unsqueeze(-1)
        if detection_valid is not None:
            valid_pairs = valid_pairs & detection_valid.unsqueeze(1)
        return logits.masked_fill(~valid_pairs, -1e4)


class MicroWorldMOT(nn.Module):
    """Trainable system: world model, observation correction and association."""

    def __init__(self, world_model: nn.Module, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.observation_projection = nn.Sequential(
            nn.Linear(config.observation_dim, config.appearance_dim),
            nn.LayerNorm(config.appearance_dim),
        )
        self.target_observation_projection = copy.deepcopy(self.observation_projection)
        self.target_observation_projection.requires_grad_(False)
        self.world_model = world_model
        self.corrector = ObservationCorrector(config)
        self.association_head = AssociationHead(config)

    def project_appearance(self, observation_features: Tensor) -> Tensor:
        if observation_features.shape[-1] != self.config.observation_dim:
            raise ValueError(
                f"Expected observation features with dim={self.config.observation_dim}, "
                f"got {observation_features.shape[-1]}"
            )
        return torch.nn.functional.normalize(
            self.observation_projection(observation_features), dim=-1
        )

    @torch.no_grad()
    def project_target_appearance(self, observation_features: Tensor) -> Tensor:
        """Encode a stop-gradient JEPA target with the EMA projection."""
        if observation_features.shape[-1] != self.config.observation_dim:
            raise ValueError(
                f"Expected observation features with dim={self.config.observation_dim}, "
                f"got {observation_features.shape[-1]}"
            )
        return torch.nn.functional.normalize(
            self.target_observation_projection(observation_features), dim=-1
        )

    @torch.no_grad()
    def update_target_encoder(self, decay: float) -> None:
        """EMA update used to keep Object-JEPA targets stable."""
        if not 0.0 <= decay <= 1.0:
            raise ValueError("EMA decay must be in [0, 1]")
        for target, online in zip(
            self.target_observation_projection.parameters(),
            self.observation_projection.parameters(),
            strict=True,
        ):
            target.mul_(decay).add_(online, alpha=1.0 - decay)

    def initialize_state(
        self, boxes: Tensor, appearance: Tensor, valid: Tensor | None = None
    ) -> WorldState:
        return self.world_model.initialize_state(
            boxes, self.project_appearance(appearance), valid
        )

    def predict(
        self,
        state: WorldState,
        camera_motion: Tensor | None = None,
        delta_time: Tensor | float | None = None,
    ) -> WorldPrediction:
        return self.world_model(state, camera_motion, delta_time)

    def association_logits(
        self,
        prediction: WorldState,
        detection_boxes: Tensor,
        detection_appearance: Tensor,
        detection_scores: Tensor,
        detection_valid: Tensor | None = None,
    ) -> Tensor:
        return self.association_head(
            prediction,
            detection_boxes,
            self.project_appearance(detection_appearance),
            detection_scores,
            detection_valid,
        )

    def correct(
        self,
        prediction: WorldState,
        observation_boxes: Tensor,
        observation_appearance: Tensor,
        observation_scores: Tensor,
        observation_mask: Tensor,
    ) -> WorldState:
        return self.corrector(
            prediction,
            observation_boxes,
            self.project_appearance(observation_appearance),
            observation_scores,
            observation_mask,
        )

    @staticmethod
    def future_energy(predictions: list[WorldPrediction]) -> Tensor:
        if not predictions:
            raise ValueError("future_energy requires at least one prediction")
        energies: list[Tensor] = []
        previous_velocity: Tensor | None = None
        for prediction in predictions:
            state = prediction.state
            uncertainty = prediction.uncertainty.mean(dim=-1)
            acceleration = prediction.acceleration.square().mean(dim=(-1, -2))
            existence_penalty = (1.0 - state.existence).mean(dim=-1)
            overlap = pairwise_iou_cxcylogwh(
                state.boxes.unsqueeze(2), state.boxes.unsqueeze(1)
            )
            objects = state.num_objects
            eye = torch.eye(objects, device=overlap.device, dtype=torch.bool).unsqueeze(0)
            valid_pair = state.valid.unsqueeze(1) & state.valid.unsqueeze(2) & ~eye
            overlap_penalty = (overlap * valid_pair).sum(dim=(-1, -2)) / valid_pair.sum(dim=(-1, -2)).clamp_min(1)
            velocity_change = state.boxes.new_zeros(state.batch_size)
            if previous_velocity is not None:
                velocity_change = (state.velocity - previous_velocity).square().mean(dim=(-1, -2))
            previous_velocity = state.velocity
            energies.append(
                uncertainty + 0.1 * acceleration + 0.1 * existence_penalty + 0.2 * overlap_penalty + 0.1 * velocity_change
            )
        return torch.stack(energies, dim=0).mean(dim=0)
