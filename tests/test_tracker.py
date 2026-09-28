import torch

from microworld_mot.config import ModelConfig, TrackerConfig
from microworld_mot.models import build_system
from microworld_mot.tracking import MicroWorldTracker
from microworld_mot.types import Detections


def test_tracker_outputs_world_prediction_when_detector_is_skipped() -> None:
    model_config = ModelConfig(
        name="microworld_cf",
        latent_dim=16,
        observation_dim=8,
        appearance_dim=8,
        graph_hidden_dim=24,
        graph_layers=1,
        num_neighbors=2,
        motion_modes=2,
    )
    tracker_config = TrackerConfig(
        confirmation_hits=1,
        existence_threshold=0.0,
        max_age=5,
    )
    tracker = MicroWorldTracker(build_system(model_config), tracker_config)
    detections = Detections(
        boxes=torch.tensor([[0.3, 0.4, -2.5, -2.0], [0.7, 0.5, -2.6, -2.1]]),
        scores=torch.tensor([0.9, 0.9]),
        appearance=torch.randn(2, 8),
    )
    first = tracker.step(detections)
    predicted = tracker.step(None)
    assert len(first) == 2
    assert [track.track_id for track in predicted] == [1, 2]
