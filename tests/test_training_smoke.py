import torch
from torch.utils.data import DataLoader

from microworld_mot.config import ExperimentConfig, ModelConfig
from microworld_mot.data import SyntheticMOTDataset
from microworld_mot.models import build_system
from microworld_mot.training.engine import sequence_objective


def test_single_run_objective_backward() -> None:
    config = ExperimentConfig()
    config.model = ModelConfig(
        name="microworld_cf",
        latent_dim=16,
        observation_dim=8,
        appearance_dim=8,
        graph_hidden_dim=24,
        graph_layers=1,
        num_neighbors=2,
        motion_modes=2,
        rollout_horizon=2,
    )
    dataset = SyntheticMOTDataset(
        samples=2,
        sequence_length=5,
        max_tracks=4,
        max_detections=6,
        observation_dim=8,
        observation_dropout=0.2,
        box_noise=0.01,
    )
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    system = build_system(config.model)
    loss, metrics = sequence_objective(system, batch, config, torch.device("cpu"), epoch=0)
    assert torch.isfinite(loss)
    assert "association" in metrics
    assert "object_jepa" in metrics
    assert "object_jepa_rollout" in metrics
    assert metrics["object_jepa"] > 0
    loss.backward()
    assert any(parameter.grad is not None for parameter in system.parameters())
    assert any(
        parameter.grad is not None
        for parameter in system.world_model.latent_prediction_head.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in system.target_observation_projection.parameters()
    )


def test_object_jepa_target_projection_uses_ema() -> None:
    config = ModelConfig(
        name="microworld_jepa",
        latent_dim=16,
        observation_dim=8,
        appearance_dim=8,
        graph_hidden_dim=24,
    )
    system = build_system(config)
    online = next(system.observation_projection.parameters())
    target = next(system.target_observation_projection.parameters())
    before = target.detach().clone()
    with torch.no_grad():
        online.add_(2.0)
    system.update_target_encoder(0.75)
    assert torch.allclose(target, before + 0.5)
