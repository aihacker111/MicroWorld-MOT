from pathlib import Path

import numpy as np

from microworld_mot.data.cache_builder import build_sequence_clips
from microworld_mot.data.datasets import CachedSequenceDataset
from microworld_mot.data.sequences import FrameObjects, SequenceRecord


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
