from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset

from ..config import DataConfig
from .sequences import SequenceRecord, scan_motchallenge, scan_visdrone


def _coverage_groups(track_ids: list[int], max_tracks: int) -> list[list[int]]:
    groups = [
        track_ids[start : start + max_tracks] for start in range(0, len(track_ids), max_tracks)
    ]
    if len(groups) > 1 and len(groups[-1]) == 1:
        groups[-1].insert(0, track_ids[0])
    return groups


def _normalized_box(box_xywh: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    xywh = box_xywh.astype(np.float32).copy()
    xywh[0] = (xywh[0] + 0.5 * xywh[2]) / width
    xywh[1] = (xywh[1] + 0.5 * xywh[3]) / height
    xywh[2] = max(xywh[2] / width, 1e-6)
    xywh[3] = max(xywh[3] / height, 1e-6)
    state = xywh.copy()
    state[2:] = np.log(state[2:])
    return xywh, state


@dataclass(frozen=True)
class RawClipIndex:
    sequence_index: int
    frame_numbers: tuple[int, ...]
    track_ids: tuple[int, ...]


class RawMOTClipDataset(Dataset[dict[str, Tensor]]):
    """Load raw MOT frames and GT for simultaneous detector/world training."""

    def __init__(
        self,
        root: str | Path,
        *,
        data_format: str,
        sequence_length: int,
        stride: int,
        temporal_intervals: list[int],
        max_tracks: int,
        max_detections: int,
        image_size: int,
        min_track_frames: int,
        gt_classes: set[int],
        seed: int,
    ) -> None:
        if sequence_length < 2 or stride < 1 or image_size < 32:
            raise ValueError("Invalid sequence_length, stride or image_size")
        if not temporal_intervals or any(interval < 1 for interval in temporal_intervals):
            raise ValueError("temporal_intervals must contain positive integers")
        self.max_tracks = max_tracks
        self.max_detections = max_detections
        self.image_size = image_size
        if data_format == "mot":
            self.sequences = scan_motchallenge(root, gt_classes)
        elif data_format == "visdrone":
            self.sequences = scan_visdrone(root, gt_classes)
        else:
            raise ValueError("data_format must be 'mot' or 'visdrone'")
        self.samples: list[RawClipIndex] = []
        intervals = sorted(set(temporal_intervals))
        for sequence_index, sequence in enumerate(self.sequences):
            stable_seed = seed + zlib.crc32(sequence.name.encode("utf-8"))
            rng = np.random.default_rng(stable_seed)
            latest_start = sequence.length - sequence_length + 1
            for start in range(1, latest_start + 1, stride):
                feasible = [
                    interval
                    for interval in intervals
                    if start + (sequence_length - 1) * interval <= sequence.length
                ]
                if not feasible:
                    continue
                interval = feasible[int(rng.integers(0, len(feasible)))]
                frame_numbers = tuple(
                    start + offset * interval for offset in range(sequence_length)
                )
                counts: dict[int, int] = {}
                visible: dict[int, int] = {}
                for frame_number in frame_numbers:
                    frame = sequence.ground_truth.get(frame_number)
                    if frame is None:
                        continue
                    for item, track_id_value in enumerate(frame.track_ids):
                        track_id = int(track_id_value)
                        counts[track_id] = counts.get(track_id, 0) + 1
                        visible[track_id] = visible.get(track_id, 0) + int(
                            frame.visibility[item] > 0.10
                        )
                candidates = [
                    track_id for track_id, count in counts.items() if count >= min_track_frames
                ]
                candidates.sort(
                    key=lambda track_id: (-visible[track_id], -counts[track_id], track_id)
                )
                if len(candidates) < 2:
                    continue
                for group in _coverage_groups(candidates, max_tracks):
                    self.samples.append(
                        RawClipIndex(sequence_index, frame_numbers, tuple(group))
                    )
        if not self.samples:
            raise FileNotFoundError(f"No valid raw MOT clips found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> Tensor:
        from PIL import Image

        with Image.open(path) as source:
            image = source.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample = self.samples[index]
        sequence: SequenceRecord = self.sequences[sample.sequence_index]
        time = len(sample.frame_numbers)
        tracks = self.max_tracks
        detections = self.max_detections
        images = torch.empty(time, 3, self.image_size, self.image_size)
        boxes = torch.zeros(time, tracks, 4)
        existence = torch.zeros(time, tracks, dtype=torch.bool)
        visibility = torch.zeros(time, tracks, dtype=torch.bool)
        slot_valid = torch.zeros(tracks, dtype=torch.bool)
        slot_valid[: len(sample.track_ids)] = True
        detection_boxes = torch.zeros(time, detections, 4)
        detection_xywh = torch.zeros(time, detections, 4)
        detection_scores = torch.zeros(time, detections)
        detection_valid = torch.zeros(time, detections, dtype=torch.bool)
        detector_target_valid = torch.zeros(time, detections, dtype=torch.bool)
        detector_labels = torch.zeros(time, detections, dtype=torch.long)
        detection_track_ids = torch.full((time, detections), -1, dtype=torch.long)
        assignment = torch.full((time, tracks), -1, dtype=torch.long)
        track_to_slot = {track_id: slot for slot, track_id in enumerate(sample.track_ids)}

        for offset, frame_number in enumerate(sample.frame_numbers):
            images[offset] = self._load_image(sequence.image_paths[frame_number - 1])
            frame = sequence.ground_truth.get(frame_number)
            if frame is None:
                continue
            order = sorted(
                range(len(frame.track_ids)),
                key=lambda item: (-float(frame.visibility[item]), int(frame.track_ids[item])),
            )[:detections]
            id_to_detection: dict[int, int] = {}
            for detection_index, item in enumerate(order):
                xywh, state_box = _normalized_box(
                    frame.boxes_xywh[item], sequence.width, sequence.height
                )
                track_id = int(frame.track_ids[item])
                detection_xywh[offset, detection_index] = torch.from_numpy(xywh)
                detection_boxes[offset, detection_index] = torch.from_numpy(state_box)
                detection_scores[offset, detection_index] = float(
                    np.clip(frame.visibility[item], 0.05, 1.0)
                )
                detector_target_valid[offset, detection_index] = True
                detection_valid[offset, detection_index] = bool(
                    frame.visibility[item] > 0.10
                )
                detection_track_ids[offset, detection_index] = track_id
                id_to_detection[track_id] = detection_index

            lookup = {int(track_id): item for item, track_id in enumerate(frame.track_ids)}
            for track_id, slot in track_to_slot.items():
                if track_id not in lookup:
                    continue
                item = lookup[track_id]
                _, state_box = _normalized_box(
                    frame.boxes_xywh[item], sequence.width, sequence.height
                )
                boxes[offset, slot] = torch.from_numpy(state_box)
                existence[offset, slot] = True
                visibility[offset, slot] = bool(frame.visibility[item] > 0.10)
                detection_index = id_to_detection.get(track_id, -1)
                if detection_index >= 0 and detection_valid[offset, detection_index]:
                    assignment[offset, slot] = detection_index

        for slot in range(len(sample.track_ids)):
            for offset in range(1, time):
                if not existence[offset, slot]:
                    boxes[offset, slot] = boxes[offset - 1, slot]
        delta_time = torch.ones(time)
        if time > 1:
            delta_time[1:] = torch.tensor(np.diff(sample.frame_numbers), dtype=torch.float32)
        return {
            "images": images,
            "boxes": boxes,
            "slot_valid": slot_valid,
            "existence": existence,
            "visibility": visibility,
            "detection_boxes": detection_boxes,
            "detection_xywh": detection_xywh,
            "detection_scores": detection_scores,
            "detection_valid": detection_valid,
            "detector_target_valid": detector_target_valid,
            "detector_labels": detector_labels,
            "detection_track_ids": detection_track_ids,
            "assignment": assignment,
            "camera_motion": torch.zeros(time, 6),
            "delta_time": delta_time,
        }


def build_raw_datasets(data_config: DataConfig, seed: int) -> tuple[Dataset, Dataset]:
    if not data_config.train_paths or not data_config.val_paths:
        raise ValueError("joint_raw requires data.train_paths and data.val_paths")
    common = {
        "data_format": data_config.raw_format,
        "sequence_length": data_config.sequence_length,
        "stride": data_config.stride,
        "max_tracks": data_config.max_tracks,
        "max_detections": data_config.max_detections,
        "image_size": data_config.image_size,
        "min_track_frames": data_config.min_track_frames,
        "gt_classes": set(data_config.gt_classes),
    }
    train_sets = [
        RawMOTClipDataset(
            root,
            temporal_intervals=data_config.temporal_intervals,
            seed=seed + index,
            **common,
        )
        for index, root in enumerate(data_config.train_paths)
    ]
    val_sets = [
        RawMOTClipDataset(
            root,
            temporal_intervals=[1],
            seed=seed + 1_000_000 + index,
            **common,
        )
        for index, root in enumerate(data_config.val_paths)
    ]
    train = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
    validation = val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)
    return train, validation
