from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..config import ModelConfig
from ..types import WorldPrediction, WorldState
from .components import SparseInteractionGraph, StructuredTransitionCell, make_mlp


class ObjectWorldModel(nn.Module):
    """Tiny object-level probabilistic world model.

    The model predicts object beliefs, not pixels. Its structured kinematic path
    gives a stable constant-velocity prior while the learned residual models
    interactions, occlusion and non-linear motion.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        state_dim = 4 + 4 + config.appearance_dim + 1 + 1 + 4
        self.state_encoder = make_mlp(
            [state_dim, config.graph_hidden_dim, config.latent_dim], dropout=config.dropout
        )
        self.input_norm = nn.LayerNorm(config.latent_dim)
        self.graph = SparseInteractionGraph(
            latent_dim=config.latent_dim,
            hidden_dim=config.graph_hidden_dim,
            layers=config.graph_layers,
            num_neighbors=config.num_neighbors,
            dropout=config.dropout,
        )
        self.transition = StructuredTransitionCell(
            latent_dim=config.latent_dim,
            camera_dim=config.camera_dim,
            hidden_dim=config.graph_hidden_dim,
            dropout=config.dropout,
        )
        modes = config.motion_modes
        self.mixture_head = make_mlp(
            [config.latent_dim, config.graph_hidden_dim, modes * 9], dropout=config.dropout
        )
        self.existence_head = nn.Linear(config.latent_dim, 1)
        self.occlusion_head = nn.Linear(config.latent_dim, 1)
        self.refresh_head = make_mlp(
            [config.latent_dim + 5, config.graph_hidden_dim, 1], dropout=config.dropout
        )
        self.latent_prediction_head = make_mlp(
            [config.latent_dim, config.graph_hidden_dim, config.appearance_dim],
            dropout=config.dropout,
        )

    def initialize_state(self, boxes: Tensor, appearance: Tensor, valid: Tensor | None = None) -> WorldState:
        if valid is None:
            valid = torch.ones(boxes.shape[:2], dtype=torch.bool, device=boxes.device)
        batch, objects, _ = boxes.shape
        velocity = torch.zeros_like(boxes)
        existence = valid.to(boxes.dtype)
        occlusion = torch.zeros_like(existence)
        log_variance = torch.full_like(boxes, math.log(0.01**2))
        encoded = self._encode(
            boxes, velocity, appearance, existence, occlusion, log_variance
        )
        memory = self.input_norm(encoded) * valid.unsqueeze(-1)
        return WorldState(
            boxes=boxes,
            velocity=velocity,
            appearance=appearance,
            memory=memory,
            valid=valid,
            existence=existence,
            occlusion=occlusion,
            log_variance=log_variance,
        )

    def _encode(
        self,
        boxes: Tensor,
        velocity: Tensor,
        appearance: Tensor,
        existence: Tensor,
        occlusion: Tensor,
        log_variance: Tensor,
    ) -> Tensor:
        inputs = torch.cat(
            (
                boxes,
                velocity,
                appearance,
                existence.unsqueeze(-1),
                occlusion.unsqueeze(-1),
                log_variance,
            ),
            dim=-1,
        )
        return self.state_encoder(inputs)

    def forward(
        self,
        state: WorldState,
        camera_motion: Tensor | None = None,
        delta_time: Tensor | float | None = None,
    ) -> WorldPrediction:
        batch = state.batch_size
        if camera_motion is None:
            camera_motion = state.boxes.new_zeros(batch, self.config.camera_dim)
        if delta_time is None:
            delta_time = state.boxes.new_ones(batch)
        elif not isinstance(delta_time, Tensor):
            delta_time = state.boxes.new_full((batch,), float(delta_time))
        delta_time = delta_time.reshape(batch, 1, 1)

        encoded = self._encode(
            state.boxes,
            state.velocity,
            state.appearance,
            state.existence,
            state.occlusion,
            state.log_variance,
        )
        memory = self.input_norm(state.memory + encoded)
        interaction = self.graph(memory, state.boxes, state.velocity, state.valid)
        next_memory, acceleration, damping = self.transition(memory, interaction, camera_motion)

        next_velocity = damping * state.velocity + delta_time * acceleration
        physical_box = state.boxes + delta_time * next_velocity + 0.5 * delta_time.square() * acceleration

        batch, objects, _ = state.boxes.shape
        modes = self.config.motion_modes
        raw_mixture = self.mixture_head(next_memory).reshape(batch, objects, modes, 9)
        residual = 0.10 * raw_mixture[..., :4].tanh()
        mixture_means = physical_box.unsqueeze(-2) + residual
        mixture_log_scales = raw_mixture[..., 4:8].clamp(-7.0, 1.0)
        mixture_logits = raw_mixture[..., 8]
        weights = mixture_logits.softmax(dim=-1).unsqueeze(-1)
        expected_box = (weights * mixture_means).sum(dim=-2)
        variance = (
            weights
            * (
                mixture_log_scales.mul(2).exp()
                + (mixture_means - expected_box.unsqueeze(-2)).square()
            )
        ).sum(dim=-2)

        existence = self.existence_head(next_memory).squeeze(-1).sigmoid()
        occlusion = self.occlusion_head(next_memory).squeeze(-1).sigmoid()
        refresh_inputs = torch.cat((next_memory, variance.mean(dim=-1, keepdim=True), occlusion.unsqueeze(-1), existence.unsqueeze(-1), acceleration.norm(dim=-1, keepdim=True), delta_time.expand(-1, objects, 1)), dim=-1)
        refresh_logits = self.refresh_head(refresh_inputs).squeeze(-1)
        latent_prediction = torch.nn.functional.normalize(
            self.latent_prediction_head(next_memory), dim=-1
        )

        valid_float = state.valid.unsqueeze(-1)
        next_state = WorldState(
            boxes=expected_box * valid_float,
            velocity=next_velocity * valid_float,
            appearance=state.appearance * valid_float,
            memory=next_memory * valid_float,
            valid=state.valid,
            existence=existence * state.valid,
            occlusion=occlusion * state.valid,
            log_variance=variance.clamp_min(1e-8).log() * valid_float,
        )
        return WorldPrediction(
            state=next_state,
            mixture_logits=mixture_logits,
            mixture_means=mixture_means,
            mixture_log_scales=mixture_log_scales,
            refresh_logits=refresh_logits,
            acceleration=acceleration,
            latent_prediction=latent_prediction * valid_float,
        )

    predict = forward

    def rollout(
        self,
        state: WorldState,
        steps: int,
        camera_motion: Tensor | None = None,
    ) -> list[WorldPrediction]:
        outputs: list[WorldPrediction] = []
        current = state
        for _ in range(steps):
            prediction = self(current, camera_motion=camera_motion)
            outputs.append(prediction)
            current = prediction.state
        return outputs
