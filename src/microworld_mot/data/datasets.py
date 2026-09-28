from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset, random_split

from ..config import DataConfig, ModelConfig


class SyntheticMOTDataset(Dataset[dict[str, Tensor]]):
    """Deterministic synthetic trajectories for development and sanity checks."""

    def __init__(
        self,
        samples: int,
        sequence_length: int,
        max_tracks: int,
        max_detections: int,
        observation_dim: int,
        observation_dropout: float,
        box_noise: float,
        seed: int = 0,
    ) -> None:
        self.samples = samples
        self.sequence_length = sequence_length
        self.max_tracks = max_tracks
        if max_detections < max_tracks:
            raise ValueError("max_detections must be >= max_tracks")
        self.max_detections = max_detections
        self.observation_dim = observation_dim
        self.observation_dropout = observation_dropout
        self.box_noise = box_noise
        self.seed = seed

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        time, tracks = self.sequence_length, self.max_tracks
        count = int(torch.randint(max(2, tracks // 3), tracks + 1, (1,), generator=generator))
        slot_valid = torch.arange(tracks) < count

        center = torch.rand(count, 2, generator=generator) * 0.70 + 0.15
        size = torch.rand(count, 2, generator=generator) * 0.08 + 0.03
        velocity = (torch.rand(count, 2, generator=generator) - 0.5) * 0.045
        acceleration = (torch.rand(count, 2, generator=generator) - 0.5) * 0.003
        size_velocity = (torch.rand(count, 2, generator=generator) - 0.5) * 0.005

        boxes = torch.zeros(time, tracks, 4)
        existence = torch.zeros(time, tracks, dtype=torch.bool)
        visibility = torch.zeros(time, tracks, dtype=torch.bool)
        appearance = torch.randn(tracks, self.observation_dim, generator=generator)
        appearance = torch.nn.functional.normalize(appearance, dim=-1)
        appearance[~slot_valid] = 0

        death_time = torch.full((count,), time)
        can_die = torch.rand(count, generator=generator) < 0.25
        if time >= 6:
            sampled_death = torch.randint(time // 2, time, (count,), generator=generator)
            death_time = torch.where(can_die, sampled_death, death_time)

        occlusion_start = torch.randint(1, max(2, time - 2), (count,), generator=generator)
        occlusion_length = torch.randint(
            0, max(1, min(5, time // 2)), (count,), generator=generator
        )
        for frame in range(time):
            if frame > 0:
                velocity = velocity + acceleration
                center = center + velocity
                for axis in range(2):
                    low = center[:, axis] < 0.03
                    high = center[:, axis] > 0.97
                    velocity[low | high, axis] *= -1
                    center[:, axis].clamp_(0.03, 0.97)
                size = (size + size_velocity).clamp(0.02, 0.20)
            boxes[frame, :count, :2] = center
            boxes[frame, :count, 2:] = size.log()
            alive = frame < death_time
            occluded = (frame >= occlusion_start) & (frame < occlusion_start + occlusion_length)
            existence[frame, :count] = alive
            visibility[frame, :count] = alive & ~occluded

        observed = visibility & (
            torch.rand(time, tracks, generator=generator) > self.observation_dropout
        )
        observed[0] = existence[0] & slot_valid
        detections = self.max_detections
        detection_boxes = torch.zeros(time, detections, 4)
        detection_features = torch.zeros(time, detections, self.observation_dim)
        detection_scores = torch.zeros(time, detections)
        detection_valid = torch.zeros(time, detections, dtype=torch.bool)
        assignment = torch.full((time, tracks), -1, dtype=torch.long)
        for frame in range(time):
            track_indices = observed[frame].nonzero(as_tuple=False).flatten()
            true_count = min(len(track_indices), detections)
            track_indices = track_indices[:true_count]
            capacity = detections - true_count
            false_count = int(torch.randint(0, min(4, capacity + 1), (1,), generator=generator))
            total = true_count + false_count
            if total == 0:
                continue
            frame_boxes = boxes[frame, track_indices].clone()
            noise = torch.randn(frame_boxes.shape, generator=generator) * self.box_noise
            noise[..., 2:] *= 0.5
            frame_boxes += noise
            frame_features = appearance[track_indices].clone()
            frame_features += 0.03 * torch.randn(frame_features.shape, generator=generator)
            frame_scores = 0.7 + 0.3 * torch.rand(true_count, generator=generator)
            source_tracks = track_indices.clone()
            if false_count:
                false_boxes = torch.rand(false_count, 4, generator=generator)
                false_boxes[:, 2:] = (0.02 + 0.15 * false_boxes[:, 2:]).log()
                false_features = torch.randn(false_count, self.observation_dim, generator=generator)
                false_scores = 0.15 + 0.55 * torch.rand(false_count, generator=generator)
                frame_boxes = torch.cat((frame_boxes, false_boxes), dim=0)
                frame_features = torch.cat((frame_features, false_features), dim=0)
                frame_scores = torch.cat((frame_scores, false_scores), dim=0)
                source_tracks = torch.cat(
                    (source_tracks, torch.full((false_count,), -1, dtype=torch.long))
                )
            permutation = torch.randperm(total, generator=generator)
            frame_boxes = frame_boxes[permutation]
            frame_features = torch.nn.functional.normalize(frame_features[permutation], dim=-1)
            frame_scores = frame_scores[permutation]
            source_tracks = source_tracks[permutation]
            detection_boxes[frame, :total] = frame_boxes
            detection_features[frame, :total] = frame_features
            detection_scores[frame, :total] = frame_scores
            detection_valid[frame, :total] = True
            for detection_index, track_index in enumerate(source_tracks.tolist()):
                if track_index >= 0:
                    assignment[frame, track_index] = detection_index

        camera = torch.zeros(time, 6)
        camera[:, :2] = 0.002 * torch.randn(time, 2, generator=generator)
        return {
            "boxes": boxes,
            "slot_valid": slot_valid,
            "existence": existence,
            "visibility": visibility,
            "detection_boxes": detection_boxes,
            "detection_features": detection_features,
            "detection_scores": detection_scores,
            "detection_valid": detection_valid,
            "assignment": assignment,
            "appearance": appearance,
            "camera_motion": camera,
            "delta_time": torch.ones(time),
        }


class CachedSequenceDataset(Dataset[dict[str, Tensor]]):
    """Loads one compressed NPZ per fixed-length sequence."""

    REQUIRED_KEYS = {
        "boxes",
        "slot_valid",
        "existence",
        "visibility",
        "detection_boxes",
        "detection_features",
        "detection_scores",
        "detection_valid",
        "assignment",
        "appearance",
        "camera_motion",
    }

    def __init__(
        self,
        root: str | Path,
        detection_dropout: float = 0.0,
        box_noise: float = 0.0,
        temporal_dropout: float = 0.0,
        temporal_dropout_max_span: int = 4,
        false_positive_ratio: float = 0.0,
        score_noise: float = 0.0,
        shuffle_detections: bool = False,
    ) -> None:
        self.files = sorted(Path(root).glob("*.npz"))
        self.detection_dropout = detection_dropout
        self.box_noise = box_noise
        self.temporal_dropout = temporal_dropout
        self.temporal_dropout_max_span = temporal_dropout_max_span
        self.false_positive_ratio = false_positive_ratio
        self.score_noise = score_noise
        self.shuffle_detections = shuffle_detections
        for name, value in {
            "detection_dropout": detection_dropout,
            "temporal_dropout": temporal_dropout,
            "false_positive_ratio": false_positive_ratio,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if box_noise < 0 or score_noise < 0:
            raise ValueError("box_noise and score_noise must be non-negative")
        if temporal_dropout_max_span < 1:
            raise ValueError("temporal_dropout_max_span must be positive")
        if not self.files:
            raise FileNotFoundError(f"No .npz sequences found in {root}")

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _repair_assignment(output: dict[str, Tensor]) -> None:
        assignment = output["assignment"]
        safe = assignment.clamp_min(0)
        assigned_valid = torch.gather(output["detection_valid"], 1, safe)
        output["assignment"] = torch.where(
            (assignment >= 0) & assigned_valid,
            assignment,
            torch.full_like(assignment, -1),
        )

    def _apply_temporal_dropout(self, output: dict[str, Tensor]) -> None:
        if self.temporal_dropout <= 0:
            return
        time, tracks = output["assignment"].shape
        valid_tracks = output["slot_valid"].nonzero(as_tuple=False).flatten().tolist()
        for track in valid_tracks:
            if torch.rand(()) >= self.temporal_dropout:
                continue
            span = int(torch.randint(1, min(time, self.temporal_dropout_max_span) + 1, (1,)))
            start = int(torch.randint(0, time - span + 1, (1,)))
            for frame in range(start, start + span):
                detection = int(output["assignment"][frame, track])
                if detection >= 0:
                    output["detection_valid"][frame, detection] = False
                    output["assignment"][frame, track] = -1

    def _insert_false_positives(self, output: dict[str, Tensor]) -> None:
        if self.false_positive_ratio <= 0:
            return
        for frame in range(output["detection_valid"].shape[0]):
            valid = output["detection_valid"][frame].nonzero(as_tuple=False).flatten()
            invalid = (~output["detection_valid"][frame]).nonzero(as_tuple=False).flatten()
            if not len(valid) or not len(invalid):
                continue
            expected = len(valid) * self.false_positive_ratio
            count = int(expected)
            count += int(torch.rand(()) < expected - count)
            count = min(len(invalid), count)
            if count == 0:
                continue
            targets = invalid[torch.randperm(len(invalid))[:count]]
            sources = valid[torch.randint(0, len(valid), (count,))]
            boxes = output["detection_boxes"][frame, sources].clone()
            boxes[:, :2] += 0.05 * torch.randn_like(boxes[:, :2])
            boxes[:, :2].clamp_(0.0, 1.0)
            boxes[:, 2:] += 0.15 * torch.randn_like(boxes[:, 2:])
            features = output["detection_features"][frame, sources].clone()
            features += 0.05 * torch.randn_like(features)
            features = torch.nn.functional.normalize(features, dim=-1)
            scores = output["detection_scores"][frame, sources]
            scores = scores * (0.2 + 0.6 * torch.rand_like(scores))
            output["detection_boxes"][frame, targets] = boxes
            output["detection_features"][frame, targets] = features
            output["detection_scores"][frame, targets] = scores
            output["detection_valid"][frame, targets] = True

    @staticmethod
    def _shuffle_detection_slots(output: dict[str, Tensor]) -> None:
        detections = output["detection_valid"].shape[1]
        for frame in range(output["detection_valid"].shape[0]):
            permutation = torch.randperm(detections)
            inverse = torch.empty_like(permutation)
            inverse[permutation] = torch.arange(detections)
            for key in (
                "detection_boxes",
                "detection_features",
                "detection_scores",
                "detection_valid",
            ):
                output[key][frame] = output[key][frame, permutation]
            assigned = output["assignment"][frame]
            mask = assigned >= 0
            output["assignment"][frame, mask] = inverse[assigned[mask]]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        with np.load(self.files[index]) as sample:
            missing = self.REQUIRED_KEYS - set(sample.files)
            if missing:
                raise KeyError(f"{self.files[index]} is missing keys: {sorted(missing)}")
            output: dict[str, Tensor] = {}
            for key in self.REQUIRED_KEYS:
                value = torch.from_numpy(sample[key])
                if key in {"slot_valid", "existence", "visibility", "detection_valid"}:
                    value = value.bool()
                elif key == "assignment":
                    value = value.long()
                else:
                    value = value.float()
                output[key] = value
            if "delta_time" in sample.files:
                output["delta_time"] = torch.from_numpy(sample["delta_time"]).float()
            else:
                output["delta_time"] = torch.ones(output["boxes"].shape[0])
            if self.detection_dropout > 0:
                dropped = (
                    torch.rand_like(output["detection_scores"]) < self.detection_dropout
                ) & output["detection_valid"]
                output["detection_valid"] &= ~dropped
                self._repair_assignment(output)
            self._apply_temporal_dropout(output)
            if self.box_noise > 0:
                noise = torch.randn_like(output["detection_boxes"]) * self.box_noise
                noise[..., 2:] *= 0.5
                output["detection_boxes"] += noise * output["detection_valid"].unsqueeze(-1)
            if self.score_noise > 0:
                noise = torch.randn_like(output["detection_scores"]) * self.score_noise
                output["detection_scores"] = (
                    output["detection_scores"] + noise * output["detection_valid"]
                ).clamp(0.0, 1.0)
            self._insert_false_positives(output)
            if self.shuffle_detections:
                self._shuffle_detection_slots(output)
            return output


def build_datasets(
    data_config: DataConfig,
    model_config: ModelConfig,
    seed: int,
) -> tuple[Dataset[dict[str, Tensor]], Dataset[dict[str, Tensor]]]:
    if data_config.kind == "synthetic":
        common: dict[str, Any] = {
            "sequence_length": data_config.sequence_length,
            "max_tracks": data_config.max_tracks,
            "max_detections": data_config.max_detections,
            "observation_dim": model_config.observation_dim,
            "observation_dropout": data_config.observation_dropout,
            "box_noise": data_config.box_noise,
        }
        return (
            SyntheticMOTDataset(data_config.train_samples, seed=seed, **common),
            SyntheticMOTDataset(data_config.val_samples, seed=seed + 1_000_000, **common),
        )
    if data_config.kind == "cached":
        if data_config.train_paths:
            train_sets = [
                CachedSequenceDataset(
                    path,
                    detection_dropout=data_config.observation_dropout,
                    box_noise=data_config.box_noise,
                    temporal_dropout=data_config.temporal_dropout,
                    temporal_dropout_max_span=data_config.temporal_dropout_max_span,
                    false_positive_ratio=data_config.false_positive_ratio,
                    score_noise=data_config.score_noise,
                    shuffle_detections=data_config.shuffle_detections,
                )
                for path in data_config.train_paths
            ]
            train: Dataset[dict[str, Tensor]] = (
                train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
            )
            if not data_config.val_paths:
                raise ValueError("data.val_paths is required when data.train_paths is used")
            val_sets = [CachedSequenceDataset(path) for path in data_config.val_paths]
            validation: Dataset[dict[str, Tensor]] = (
                val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)
            )
            return train, validation
        full = CachedSequenceDataset(data_config.path)
        val_count = max(1, int(round(0.1 * len(full))))
        train_count = len(full) - val_count
        if train_count < 1:
            raise ValueError("Cached dataset needs at least two sequences")
        return tuple(
            random_split(
                full, [train_count, val_count], generator=torch.Generator().manual_seed(seed)
            )
        )  # type: ignore[return-value]
    raise ValueError(f"Unknown data kind: {data_config.kind}")
