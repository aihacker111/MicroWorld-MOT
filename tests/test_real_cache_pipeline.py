from pathlib import Path

import numpy as np
import torch

from microworld_mot.data.cache_builder import build_sequence_clips, cache_sequence_detections
from microworld_mot.data.datasets import CachedSequenceDataset
from microworld_mot.data.sequences import FrameObjects, SequenceRecord
from microworld_mot.perception import DetectorOutput
from tools.audit_dataset_clips import audit


class _BatchPerception:
    feature_dim = 4
    score_threshold = 0.15

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def detect_batch(self, image_paths):
        self.batch_sizes.append(len(image_paths))
        return [
            DetectorOutput(
                boxes_xyxy=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
                scores=torch.tensor([0.9]),
                labels=torch.tensor([1]),
                features=torch.ones(1, self.feature_dim),
            )
            for _ in image_paths
        ]


def test_detection_cache_batches_missing_frames(tmp_path: Path) -> None:
    sequence = SequenceRecord(
        name="batch_cache",
        image_paths=[tmp_path / f"{index}.jpg" for index in range(5)],
        width=100,
        height=100,
        ground_truth={},
    )
    perception = _BatchPerception()
    destination = cache_sequence_detections(
        sequence,
        tmp_path / "cache",
        perception,  # type: ignore[arg-type]
        "yolo11n",
        inference_batch_size=2,
    )

    assert perception.batch_sizes == [2, 2, 1]
    assert len(list(destination.glob("*.npz"))) == 5


def test_real_detection_cache_builds_unaligned_clip(tmp_path: Path) -> None:
    frame = FrameObjects(
        track_ids=np.asarray([10, 20]),
        boxes_xywh=np.asarray([[10, 10, 20, 30], [60, 15, 15, 25]], dtype=np.float32),
        visibility=np.ones(2, dtype=np.float32),
        labels=np.ones(2, dtype=np.int64),
    )
    sequence = SequenceRecord(
        name="sequence",
        image_paths=[tmp_path / f"{index}.jpg" for index in range(3)],
        width=100,
        height=100,
        ground_truth={1: frame, 2: frame, 3: frame},
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    # Reverse the detector order to ensure assignments cannot rely on slot order.
    detector_boxes = np.asarray([[60, 15, 75, 40], [10, 10, 30, 40]], dtype=np.float32)
    for frame_index in range(1, 4):
        np.savez_compressed(
            cache / f"{frame_index:06d}.npz",
            boxes_xyxy=detector_boxes,
            scores=np.asarray([0.9, 0.8], dtype=np.float32),
            labels=np.ones(2, dtype=np.int64),
            features=np.eye(2, 4, dtype=np.float32),
        )
    output = tmp_path / "clips"
    count = build_sequence_clips(
        sequence=sequence,
        frame_cache_dir=cache,
        output_dir=output,
        sequence_length=3,
        stride=1,
        max_tracks=2,
        max_detections=3,
        feature_dim=4,
        iou_threshold=0.5,
        dataset_name="test",
    )
    assert count == 1
    sample = CachedSequenceDataset(output)[0]
    assert sample["detection_boxes"].shape == (3, 3, 4)
    assert sample["assignment"][0].tolist() == [1, 0]
    assert sample["detection_valid"][0].sum() == 2
    report = audit([output])
    assert report["protocol_versions"] == {2: 1}
    assert report["detector_match_rate_visible"] == 1.0


def test_gt_first_sampling_keeps_tracks_missed_in_first_frame(tmp_path: Path) -> None:
    frame = FrameObjects(
        track_ids=np.asarray([10, 20]),
        boxes_xywh=np.asarray([[10, 10, 20, 30], [60, 15, 15, 25]], dtype=np.float32),
        visibility=np.ones(2, dtype=np.float32),
        labels=np.ones(2, dtype=np.int64),
    )
    sequence = SequenceRecord(
        name="missed_start",
        image_paths=[tmp_path / f"{index}.jpg" for index in range(3)],
        width=100,
        height=100,
        ground_truth={1: frame, 2: frame, 3: frame},
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    boxes = np.asarray([[10, 10, 30, 40], [60, 15, 75, 40]], dtype=np.float32)
    for frame_index in range(1, 4):
        frame_boxes = np.empty((0, 4), dtype=np.float32) if frame_index == 1 else boxes
        np.savez_compressed(
            cache / f"{frame_index:06d}.npz",
            boxes_xyxy=frame_boxes,
            scores=np.ones(len(frame_boxes), dtype=np.float32),
            labels=np.ones(len(frame_boxes), dtype=np.int64),
            features=np.ones((len(frame_boxes), 4), dtype=np.float32),
        )

    output = tmp_path / "clips"
    count = build_sequence_clips(
        sequence=sequence,
        frame_cache_dir=cache,
        output_dir=output,
        sequence_length=3,
        stride=1,
        max_tracks=2,
        max_detections=3,
        feature_dim=4,
        iou_threshold=0.5,
        dataset_name="test",
    )

    assert count == 1
    sample = CachedSequenceDataset(output)[0]
    assert sample["slot_valid"].tolist() == [True, True]
    assert sample["assignment"][0].tolist() == [-1, -1]
    assert sample["existence"][0].tolist() == [True, True]
    assert bool((sample["boxes"][0].abs().sum(dim=-1) > 0).all())


def test_crowded_windows_create_coverage_groups(tmp_path: Path) -> None:
    track_ids = np.arange(1, 6)
    boxes_xywh = np.asarray([[5 + 15 * index, 10, 10, 20] for index in range(5)], dtype=np.float32)
    frame = FrameObjects(
        track_ids=track_ids,
        boxes_xywh=boxes_xywh,
        visibility=np.ones(5, dtype=np.float32),
        labels=np.ones(5, dtype=np.int64),
    )
    sequence = SequenceRecord(
        name="crowded",
        image_paths=[tmp_path / f"{index}.jpg" for index in range(3)],
        width=100,
        height=100,
        ground_truth={1: frame, 2: frame, 3: frame},
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    boxes_xyxy = boxes_xywh.copy()
    boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]
    for frame_index in range(1, 4):
        np.savez_compressed(
            cache / f"{frame_index:06d}.npz",
            boxes_xyxy=boxes_xyxy,
            scores=np.ones(5, dtype=np.float32),
            labels=np.ones(5, dtype=np.int64),
            features=np.ones((5, 4), dtype=np.float32),
        )
    output = tmp_path / "clips"

    count = build_sequence_clips(
        sequence=sequence,
        frame_cache_dir=cache,
        output_dir=output,
        sequence_length=3,
        stride=1,
        max_tracks=2,
        max_detections=5,
        feature_dim=4,
        iou_threshold=0.5,
        dataset_name="test",
    )

    assert count == 3
    covered: set[int] = set()
    for path in output.glob("*.npz"):
        with np.load(path) as clip:
            covered.update(int(value) for value in clip["track_ids"] if value >= 0)
    assert covered == set(track_ids.tolist())


def test_temporal_interval_and_online_detection_augmentation(tmp_path: Path) -> None:
    frame = FrameObjects(
        track_ids=np.asarray([10, 20]),
        boxes_xywh=np.asarray([[10, 10, 20, 30], [60, 15, 15, 25]], dtype=np.float32),
        visibility=np.ones(2, dtype=np.float32),
        labels=np.ones(2, dtype=np.int64),
    )
    sequence = SequenceRecord(
        name="interval",
        image_paths=[tmp_path / f"{index}.jpg" for index in range(5)],
        width=100,
        height=100,
        ground_truth={index: frame for index in range(1, 6)},
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    boxes = np.asarray([[10, 10, 30, 40], [60, 15, 75, 40]], dtype=np.float32)
    for frame_index in range(1, 6):
        np.savez_compressed(
            cache / f"{frame_index:06d}.npz",
            boxes_xyxy=boxes,
            scores=np.asarray([0.9, 0.8], dtype=np.float32),
            labels=np.ones(2, dtype=np.int64),
            features=np.eye(2, 4, dtype=np.float32),
        )
    output = tmp_path / "clips"
    count = build_sequence_clips(
        sequence=sequence,
        frame_cache_dir=cache,
        output_dir=output,
        sequence_length=3,
        stride=1,
        max_tracks=2,
        max_detections=3,
        feature_dim=4,
        iou_threshold=0.5,
        dataset_name="test",
        temporal_intervals=(2,),
    )

    assert count == 1
    path = next(output.glob("*.npz"))
    with np.load(path) as clip:
        assert clip["frame_numbers"].tolist() == [1, 3, 5]
        assert clip["delta_time"].tolist() == [1.0, 2.0, 2.0]
        assert int(clip["protocol_version"]) == 2
        assert int(clip["sequence_length"]) == 3
        assert int(clip["max_tracks"]) == 2

    torch.manual_seed(3)
    augmented = CachedSequenceDataset(
        output,
        temporal_dropout=1.0,
        temporal_dropout_max_span=2,
        false_positive_ratio=0.5,
        shuffle_detections=True,
    )[0]
    assert bool((augmented["assignment"] == -1).any())
    assert bool((augmented["detection_valid"].sum(dim=1) >= 1).all())
    assert bool((augmented["detection_valid"].sum(dim=1) == 3).any())
    assert augmented["delta_time"].tolist() == [1.0, 2.0, 2.0]
