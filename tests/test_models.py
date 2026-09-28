import pytest
import torch

from microworld_mot.config import ModelConfig
from microworld_mot.models import build_system


@pytest.mark.parametrize(
    "name",
    [
        "constant_velocity",
        "kalman",
        "mlp_delta",
        "gru_tiny",
        "ssm_tiny",
        "microworld_cf_tiny",
    ],
)
def test_model_contract(name: str) -> None:
    config = ModelConfig(
        name=name,
        latent_dim=16,
        observation_dim=8,
        appearance_dim=8,
        graph_hidden_dim=24,
        graph_layers=1,
        num_neighbors=2,
        motion_modes=2,
    )
    system = build_system(config)
    observation_dim = system.config.observation_dim
    boxes = torch.randn(2, 4, 4)
    appearance = torch.randn(2, 4, observation_dim)
    state = system.initialize_state(boxes, appearance)
    prediction = system.predict(state)
    assert prediction.expected_boxes.shape == (2, 4, 4)
    assert prediction.state.memory.shape[-1] == system.config.latent_dim
    assert torch.isfinite(prediction.expected_boxes).all()


def test_world_model_backpropagates() -> None:
    config = ModelConfig(
        name="microworld_cf",
        latent_dim=16,
        observation_dim=8,
        appearance_dim=8,
        graph_hidden_dim=24,
        graph_layers=1,
        num_neighbors=2,
        motion_modes=2,
    )
    system = build_system(config)
    state = system.initialize_state(torch.randn(1, 3, 4), torch.randn(1, 3, 8))
    prediction = system.predict(state)
    prediction.expected_boxes.square().mean().backward()
    assert any(parameter.grad is not None for parameter in system.world_model.parameters())
