from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import nn

from microworld_mot.config import DataConfig, ExperimentConfig, ModelConfig
from microworld_mot.data.raw_clips import RawMOTClipDataset
from microworld_mot.models import build_system
from microworld_mot.perception import JointYOLO11
from microworld_mot.training.joint_engine import joint_sequence_objective
from microworld_mot.types import WorldPrediction, WorldState


class _FakeDetect(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.cv2 = nn.ModuleList([nn.Sequential(nn.Conv2d(channels, 4, 1))])
        self.output = nn.Conv2d(channels, 1, 1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.output(features[0])


class _FakeDetectionModel(nn.Module):
    def __init__(self, channels: int = 8) -> None:
        super().__init__()
        self.model = nn.ModuleList(
            [nn.Conv2d(3, channels, 3, padding=1), _FakeDetect(channels)]
        )

    def forward(self, inputs):
        if isinstance(inputs, dict):
            images = inputs["img"]
        else:
            images = inputs
        features = [torch.relu(self.model[0](images))]
        output = self.model[-1](features)
        if isinstance(inputs, dict):
            target_term = inputs["bboxes"].sum() * 0.0
            return output.square().mean() + target_term
        return output


def _prediction(batch: int, objects: int) -> WorldPrediction:
    boxes = torch.tensor([0.5, 0.5, -1.5, -1.5]).view(1, 1, 4)
    boxes = boxes.expand(batch, objects, -1).clone().requires_grad_()
    state = WorldState(
        boxes=boxes,
        velocity=torch.zeros_like(boxes),
        appearance=torch.zeros(batch, objects, 8),
        memory=torch.zeros(batch, objects, 8),
        valid=torch.ones(batch, objects, dtype=torch.bool),
        existence=torch.ones(batch, objects),
        occlusion=torch.zeros(batch, objects),
        log_variance=torch.full_like(boxes, -4.0),
    )
    return WorldPrediction(
        state=state,
        mixture_logits=torch.zeros(batch, objects, 1),
        mixture_means=boxes.unsqueeze(-2),
        mixture_log_scales=torch.zeros(batch, objects, 1, 4),
        refresh_logits=torch.zeros(batch, objects),
        acceleration=torch.zeros_like(boxes),
    )


def test_joint_yolo_bridge_backpropagates_through_prior_and_rois() -> None:
    detector = _FakeDetectionModel()
    bridge = JointYOLO11(
        "fake.pt",
        observation_dim=16,
        detector_model=detector,
        feature_channels=[8],
    )
    bridge.set_world_prior(_prediction(1, 2))
    images = torch.rand(1, 3, 16, 16)
    targets = {
        "batch_idx": torch.tensor([0]),
        "cls": torch.zeros(1, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
    }
    detector_loss = bridge.detector_loss(images, targets)
    roi_features = bridge.encode_rois(
        torch.tensor([[[0.5, 0.5, 0.3, 0.3], [0.25, 0.25, 0.2, 0.2]]]),
        torch.ones(1, 2, dtype=torch.bool),
    )
    (detector_loss + roi_features.square().mean()).backward()
    assert roi_features.shape == (1, 2, 16)
    assert bridge.world_gates.grad is not None
    assert detector.model[0].weight.grad is not None
    assert bridge.roi_projections[0].weight.grad is not None


def test_raw_mot_dataset_loads_joint_clip(tmp_path: Path) -> None:
    sequence = tmp_path / "SEQ-01"
    (sequence / "img1").mkdir(parents=True)
    (sequence / "gt").mkdir()
    (sequence / "seqinfo.ini").write_text(
        "[Sequence]\nseqLength=5\nimWidth=32\nimHeight=24\n",
        encoding="utf-8",
    )
    lines: list[str] = []
    for frame in range(1, 6):
        Image.new("RGB", (32, 24), color=(frame * 20, 0, 0)).save(
            sequence / "img1" / f"{frame:06d}.jpg"
        )
        lines.extend(
            [
                f"{frame},1,{frame},2,8,10,1,1,1",
                f"{frame},2,12,{frame},7,9,1,1,0.8",
            ]
        )
    (sequence / "gt" / "gt.txt").write_text("\n".join(lines), encoding="utf-8")
    dataset = RawMOTClipDataset(
        tmp_path,
        data_format="mot",
        sequence_length=3,
        stride=1,
        temporal_intervals=[1],
        max_tracks=4,
        max_detections=8,
        image_size=32,
        min_track_frames=2,
        gt_classes={1},
        seed=7,
    )
    sample = dataset[0]
    assert sample["images"].shape == (3, 3, 32, 32)
    assert sample["slot_valid"].sum().item() == 2
    assert sample["detector_target_valid"].sum().item() == 6
    assert torch.equal(sample["assignment"][:, :2], torch.tensor([[0, 1]]).expand(3, 2))


def test_joint_objective_updates_detector_and_world_model() -> None:
    model_config = ModelConfig(
        name="microworld_jepa",
        latent_dim=16,
        observation_dim=16,
        appearance_dim=8,
        graph_hidden_dim=16,
        graph_layers=1,
        num_neighbors=1,
        motion_modes=2,
        rollout_horizon=2,
        camera_dim=6,
        dropout=0.0,
    )
    config = ExperimentConfig(model=model_config, data=DataConfig(kind="joint_raw"))
    config.joint.enabled = True
    config.train.detach_interval = 2
    system = build_system(model_config)
    detector = _FakeDetectionModel()
    perception = JointYOLO11(
        "fake.pt",
        observation_dim=16,
        detector_model=detector,
        feature_channels=[8],
    )
    batch, time, tracks, detections = 1, 3, 2, 3
    xywh = torch.tensor(
        [[0.30, 0.30, 0.20, 0.20], [0.70, 0.65, 0.18, 0.22], [0.0, 0.0, 0.0, 0.0]]
    )
    state_boxes = xywh.clone()
    state_boxes[:, 2:] = state_boxes[:, 2:].clamp_min(1e-6).log()
    raw_batch = {
        "images": torch.rand(batch, time, 3, 16, 16),
        "boxes": state_boxes[:tracks].view(1, 1, tracks, 4).expand(batch, time, -1, -1).clone(),
        "slot_valid": torch.ones(batch, tracks, dtype=torch.bool),
        "existence": torch.ones(batch, time, tracks, dtype=torch.bool),
        "visibility": torch.ones(batch, time, tracks, dtype=torch.bool),
        "detection_boxes": state_boxes.view(1, 1, detections, 4).expand(batch, time, -1, -1).clone(),
        "detection_xywh": xywh.view(1, 1, detections, 4).expand(batch, time, -1, -1).clone(),
        "detection_scores": torch.tensor([1.0, 0.9, 0.0]).view(1, 1, -1).expand(batch, time, -1).clone(),
        "detection_valid": torch.tensor([True, True, False]).view(1, 1, -1).expand(batch, time, -1).clone(),
        "detector_target_valid": torch.tensor([True, True, False]).view(1, 1, -1).expand(batch, time, -1).clone(),
        "detector_labels": torch.zeros(batch, time, detections, dtype=torch.long),
        "detection_track_ids": torch.tensor([1, 2, -1]).view(1, 1, -1).expand(batch, time, -1).clone(),
        "assignment": torch.tensor([0, 1]).view(1, 1, tracks).expand(batch, time, -1).clone(),
        "camera_motion": torch.zeros(batch, time, 6),
        "delta_time": torch.ones(batch, time),
    }
    loss, metrics = joint_sequence_objective(
        system, perception, raw_batch, config, torch.device("cpu"), epoch=0
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["detector"].item() > 0
    assert any(parameter.grad is not None for parameter in system.world_model.parameters())
    assert detector.model[0].weight.grad is not None
