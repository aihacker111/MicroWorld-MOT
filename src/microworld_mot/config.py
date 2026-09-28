from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    name: str = "microworld_jepa_tiny"
    latent_dim: int = 64
    observation_dim: int = 64
    appearance_dim: int = 32
    graph_hidden_dim: int = 96
    graph_layers: int = 2
    num_neighbors: int = 6
    motion_modes: int = 3
    rollout_horizon: int = 3
    max_hypotheses: int = 4
    camera_dim: int = 6
    dropout: float = 0.05


@dataclass
class DataConfig:
    kind: str = "synthetic"
    path: str = ""
    train_paths: list[str] = field(default_factory=list)
    val_paths: list[str] = field(default_factory=list)
    sequence_length: int = 16
    max_tracks: int = 12
    max_detections: int = 20
    train_samples: int = 2048
    val_samples: int = 256
    observation_dropout: float = 0.20
    box_noise: float = 0.01
    temporal_dropout: float = 0.0
    temporal_dropout_max_span: int = 4
    false_positive_ratio: float = 0.0
    score_noise: float = 0.0
    shuffle_detections: bool = True
    num_workers: int = 0
    balance_datasets: bool = True
    raw_format: str = "mot"
    image_size: int = 640
    stride: int = 4
    temporal_intervals: list[int] = field(default_factory=lambda: [1, 2])
    min_track_frames: int = 2
    gt_classes: list[int] = field(default_factory=lambda: [1])


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    seed: int = 7
    device: str = "auto"
    amp: bool = True
    output_dir: str = "outputs/microworld_jepa_tiny"
    resume: str = ""
    counterfactual_warmup_epochs: int = 5
    object_mask_ratio: float = 0.35
    target_ema_decay: float = 0.996
    gradient_accumulation: int = 1
    detach_interval: int = 4
    log_interval: int = 20


@dataclass
class JointConfig:
    enabled: bool = False
    detector_weights: str = "yolo11n.pt"
    detector_backbone_lr: float = 1e-5
    detector_head_lr: float = 5e-5
    perception_lr: float = 1e-4
    world_lr: float = 3e-4
    detector_loss_weight: float = 1.0
    world_loss_weight: float = 1.0
    association_temperature: float = 0.10
    sinkhorn_iterations: int = 8
    roi_size: int = 3
    world_prior_strength: float = 0.10


@dataclass
class LossConfig:
    state: float = 1.0
    rollout: float = 0.5
    association: float = 1.0
    counterfactual: float = 0.5
    identity: float = 0.25
    existence: float = 0.25
    occlusion: float = 0.25
    uncertainty: float = 0.10
    refresh: float = 0.10
    object_jepa: float = 0.50
    object_jepa_rollout: float = 0.25
    counterfactual_margin: float = 0.05


@dataclass
class TrackerConfig:
    association_threshold: float = 2.5
    ambiguity_margin: float = 0.25
    birth_threshold: float = 0.35
    existence_threshold: float = 0.20
    confirmation_hits: int = 2
    max_age: int = 30
    max_detector_gap: int = 4
    refresh_threshold: float = 0.55
    use_counterfactual: bool = True


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    joint: JointConfig = field(default_factory=JointConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(**data.get("model", {})),
        data=DataConfig(**data.get("data", {})),
        train=TrainConfig(**data.get("train", {})),
        loss=LossConfig(**data.get("loss", {})),
        tracker=TrackerConfig(**data.get("tracker", {})),
        joint=JointConfig(**data.get("joint", {})),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return config_from_dict(json.load(handle))


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(config.as_dict(), handle, indent=2)
