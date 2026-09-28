from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..config import ModelConfig
from ..types import WorldPrediction, WorldState
from .components import make_mlp


class _BaselineBase(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    def initialize_state(
        self, boxes: Tensor, appearance: Tensor, valid: Tensor | None = None
    ) -> WorldState:
        if valid is None:
            valid = torch.ones(boxes.shape[:2], dtype=torch.bool, device=boxes.device)
        memory = boxes.new_zeros(*boxes.shape[:2], self.config.latent_dim)
        return WorldState(
            boxes=boxes,
            velocity=torch.zeros_like(boxes),
            appearance=appearance,
            memory=memory,
            valid=valid,
            existence=valid.to(boxes.dtype),
            occlusion=torch.zeros_like(valid, dtype=boxes.dtype),
            log_variance=torch.full_like(boxes, math.log(0.01**2)),
        )

    @staticmethod
    def _delta_time(state: WorldState, delta_time: Tensor | float | None) -> Tensor:
        if delta_time is None:
            return state.boxes.new_ones(state.batch_size, 1, 1)
        if isinstance(delta_time, Tensor):
            return delta_time.to(state.boxes).reshape(state.batch_size, 1, 1)
        return state.boxes.new_full((state.batch_size, 1, 1), float(delta_time))

    def _prediction(
        self, state: WorldState, boxes: Tensor, velocity: Tensor, log_scale: Tensor
    ) -> WorldPrediction:
        next_state = WorldState(
            boxes=boxes,
            velocity=velocity,
            appearance=state.appearance,
            memory=state.memory,
            valid=state.valid,
            existence=state.existence,
            occlusion=state.occlusion,
            log_variance=2 * log_scale,
        )
        return WorldPrediction(
            state=next_state,
            mixture_logits=boxes.new_zeros(*boxes.shape[:2], 1),
            mixture_means=boxes.unsqueeze(-2),
            mixture_log_scales=log_scale.unsqueeze(-2),
            refresh_logits=boxes.new_zeros(boxes.shape[:2]),
            acceleration=torch.zeros_like(boxes),
        )


class ConstantVelocityModel(_BaselineBase):
    def forward(
        self,
        state: WorldState,
        camera_motion: Tensor | None = None,
        delta_time: Tensor | float | None = None,
    ) -> WorldPrediction:
        del camera_motion
        dt = self._delta_time(state, delta_time)
        boxes = state.boxes + dt * state.velocity
        log_scale = 0.5 * state.log_variance + math.log(1.05)
        return self._prediction(state, boxes, state.velocity, log_scale)

    predict = forward


class KalmanWorldModel(ConstantVelocityModel):
    """Parameter-free diagonal covariance predict step used as a fair baseline."""

    def forward(
        self,
        state: WorldState,
        camera_motion: Tensor | None = None,
        delta_time: Tensor | float | None = None,
    ) -> WorldPrediction:
        prediction = super().forward(state, camera_motion, delta_time)
        process_noise = prediction.state.boxes.new_tensor([1e-4, 1e-4, 5e-5, 5e-5])
        variance = state.log_variance.exp() + process_noise
        prediction.state.log_variance = variance.log()
        prediction.mixture_log_scales = (0.5 * variance.log()).unsqueeze(-2)
        return prediction


class MLPDeltaModel(_BaselineBase):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        input_dim = 8 + config.appearance_dim
        self.network = make_mlp([input_dim, config.latent_dim, config.latent_dim, 8])

    def forward(
        self,
        state: WorldState,
        camera_motion: Tensor | None = None,
        delta_time: Tensor | float | None = None,
    ) -> WorldPrediction:
        del camera_motion
        dt = self._delta_time(state, delta_time)
        output = self.network(torch.cat((state.boxes, state.velocity, state.appearance), dim=-1))
        delta, scale = output.chunk(2, dim=-1)
        acceleration = 0.1 * delta.tanh()
        velocity = state.velocity + dt * acceleration
        boxes = state.boxes + dt * state.velocity + 0.5 * dt.square() * acceleration
        return self._prediction(state, boxes, velocity, scale.clamp(-7, 1))

    predict = forward


class GRUWorldModel(_BaselineBase):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.encoder = nn.Linear(8 + config.appearance_dim, config.latent_dim)
        self.cell = nn.GRUCell(config.latent_dim, config.latent_dim)
        self.head = nn.Linear(config.latent_dim, 8)

    def forward(
        self,
        state: WorldState,
        camera_motion: Tensor | None = None,
        delta_time: Tensor | float | None = None,
    ) -> WorldPrediction:
        del camera_motion
        dt = self._delta_time(state, delta_time)
        features = torch.cat((state.boxes, state.velocity, state.appearance), dim=-1)
        batch, objects, _ = features.shape
        memory = self.cell(
            self.encoder(features).reshape(batch * objects, -1),
            state.memory.reshape(batch * objects, -1),
        ).reshape(batch, objects, -1)
        delta, scale = self.head(memory).chunk(2, dim=-1)
        acceleration = 0.1 * delta.tanh()
        velocity = state.velocity + dt * acceleration
        boxes = state.boxes + dt * state.velocity + 0.5 * dt.square() * acceleration
        prediction = self._prediction(state, boxes, velocity, scale.clamp(-7, 1))
        prediction.state.memory = memory
        return prediction

    predict = forward


class TinySSMWorldModel(_BaselineBase):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.input_projection = nn.Linear(8 + config.appearance_dim, config.latent_dim)
        self.log_a = nn.Parameter(torch.full((config.latent_dim,), -2.0))
        self.b = nn.Parameter(torch.ones(config.latent_dim))
        self.output = nn.Linear(config.latent_dim, 8)

    def forward(
        self,
        state: WorldState,
        camera_motion: Tensor | None = None,
        delta_time: Tensor | float | None = None,
    ) -> WorldPrediction:
        del camera_motion
        dt = self._delta_time(state, delta_time)
        features = torch.cat((state.boxes, state.velocity, state.appearance), dim=-1)
        inputs = self.input_projection(features)
        a = -self.log_a.exp()
        discrete_a = (dt * a).exp()
        memory = discrete_a * state.memory + dt * self.b * inputs
        delta, scale = self.output(memory).chunk(2, dim=-1)
        acceleration = 0.1 * delta.tanh()
        velocity = state.velocity + dt * acceleration
        boxes = state.boxes + dt * state.velocity + 0.5 * dt.square() * acceleration
        prediction = self._prediction(state, boxes, velocity, scale.clamp(-7, 1))
        prediction.state.memory = memory
        return prediction

    predict = forward
