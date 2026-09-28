from __future__ import annotations

import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..association.hungarian import linear_assignment
from ..perception import DetectorOutput, YOLO11Perception
from .sequences import FrameObjects, SequenceRecord


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    output = boxes.astype(np.float32).copy()
    output[:, 2] += output[:, 0]
    output[:, 3] += output[:, 1]
    return output


def normalize_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    center_x = (boxes[:, 0] + boxes[:, 2]) / (2 * width)
    center_y = (boxes[:, 1] + boxes[:, 3]) / (2 * height)
    box_width = np.maximum((boxes[:, 2] - boxes[:, 0]) / width, 1e-6)
    box_height = np.maximum((boxes[:, 3] - boxes[:, 1]) / height, 1e-6)
    return np.stack((center_x, center_y, np.log(box_width), np.log(box_height)), axis=-1).astype(
        np.float32
    )


def pairwise_iou_numpy(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if not len(boxes_a) or not len(boxes_b):
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    minimum = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    maximum = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = np.maximum(maximum - minimum, 0)
    intersection_area = intersection[..., 0] * intersection[..., 1]
    area_a = np.prod(np.maximum(boxes_a[:, 2:] - boxes_a[:, :2], 0), axis=-1)
    area_b = np.prod(np.maximum(boxes_b[:, 2:] - boxes_b[:, :2], 0), axis=-1)
    return intersection_area / np.maximum(
        area_a[:, None] + area_b[None, :] - intersection_area, 1e-6
    )


def match_ground_truth(
    ground_truth_xyxy: np.ndarray,
    detections_xyxy: np.ndarray,
    iou_threshold: float,
) -> list[tuple[int, int]]:
    iou = pairwise_iou_numpy(ground_truth_xyxy, detections_xyxy)
    matches, _, _ = linear_assignment(1.0 - iou, threshold=1.0 - iou_threshold)
    return matches


def estimate_camera_motion(sequence: SequenceRecord) -> np.ndarray:
    """Estimate normalized affine camera motion with ORB and RANSAC."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python is required for camera-motion estimation") from error
    tokens = np.zeros((sequence.length, 6), dtype=np.float32)
    detector = cv2.ORB_create(nfeatures=1500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    previous = cv2.imread(str(sequence.image_paths[0]), cv2.IMREAD_GRAYSCALE)
    previous_keypoints, previous_descriptors = detector.detectAndCompute(previous, None)
    for index, image_path in enumerate(sequence.image_paths[1:], start=1):
        current = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        current_keypoints, current_descriptors = detector.detectAndCompute(current, None)
        if previous_descriptors is not None and current_descriptors is not None:
            matches = matcher.match(previous_descriptors, current_descriptors)
            matches = sorted(matches, key=lambda match: match.distance)[:300]
            if len(matches) >= 6:
                source = np.float32([previous_keypoints[match.queryIdx].pt for match in matches])
                target = np.float32([current_keypoints[match.trainIdx].pt for match in matches])
                affine, _ = cv2.estimateAffinePartial2D(
                    source,
                    target,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0,
                )
                if affine is not None:
                    tokens[index] = np.asarray(
                        [
                            affine[0, 0] - 1,
                            affine[0, 1],
                            affine[0, 2] / sequence.width,
                            affine[1, 0],
                            affine[1, 1] - 1,
                            affine[1, 2] / sequence.height,
                        ],
                        dtype=np.float32,
                    )
        previous = current
        previous_keypoints = current_keypoints
        previous_descriptors = current_descriptors
    return tokens


def cache_sequence_detections(
    sequence: SequenceRecord,
    cache_root: str | Path,
    perception: YOLO11Perception,
    detector_source: str,
    overwrite: bool = False,
) -> Path:
    destination = Path(cache_root) / sequence.name
    destination.mkdir(parents=True, exist_ok=True)
    for frame, image_path in enumerate(sequence.image_paths, start=1):
        cache_path = destination / f"{frame:06d}.npz"
        if cache_path.exists() and not overwrite:
            with np.load(cache_path) as cached:
                cached_feature_dim = cached["features"].shape[-1]
            if cached_feature_dim == perception.feature_dim:
                continue
            raise ValueError(
                f"Stale detector cache at {cache_path}: expected feature_dim="
                f"{perception.feature_dim}, got {cached_feature_dim}. Rerun with "
                "--overwrite-cache after changing the perception backbone."
            )
        if detector_source == "public":
            rows = sequence.public_detections.get(frame, np.empty((0, 5), dtype=np.float32))
            rows = rows[rows[:, 4] >= perception.score_threshold]
            boxes_xyxy = xywh_to_xyxy(rows[:, :4]) if len(rows) else rows[:, :4]
            scores = rows[:, 4] if len(rows) else np.empty(0, dtype=np.float32)
            features = perception.encode_boxes(
                image_path, torch.from_numpy(boxes_xyxy).float()
            ).numpy()
            labels = np.ones(len(rows), dtype=np.int64)
        else:
            output = perception.detect(image_path)
            boxes_xyxy = output.boxes_xyxy.numpy().astype(np.float32)
            scores = output.scores.numpy().astype(np.float32)
            labels = output.labels.numpy().astype(np.int64)
            features = output.features.numpy().astype(np.float32)
        np.savez_compressed(
            cache_path,
            boxes_xyxy=boxes_xyxy,
            scores=scores,
            labels=labels,
            features=features,
        )
    return destination


def _load_frame_cache(cache_path: Path) -> DetectorOutput:
    with np.load(cache_path) as frame:
        return DetectorOutput(
            boxes_xyxy=torch.from_numpy(frame["boxes_xyxy"]).float(),
            scores=torch.from_numpy(frame["scores"]).float(),
            labels=torch.from_numpy(frame["labels"]).long(),
            features=torch.from_numpy(frame["features"]).float(),
        )


def _frame_lookup(frame: FrameObjects | None) -> dict[int, int]:
    if frame is None:
        return {}
    return {int(track_id): index for index, track_id in enumerate(frame.track_ids)}


def _coverage_groups(track_ids: list[int], max_tracks: int) -> list[list[int]]:
    """Partition crowded clips without permanently dropping tail identities."""
    groups = [
        track_ids[start : start + max_tracks] for start in range(0, len(track_ids), max_tracks)
    ]
    if len(groups) > 1 and len(groups[-1]) == 1:
        # A single-object tail is poor interaction supervision. Add one context
        # identity while retaining the previously uncovered identity.
        groups[-1].insert(0, track_ids[0])
    return groups


def _aggregate_camera_motion(
    camera_motion: np.ndarray | None,
    frame_numbers: list[int],
) -> np.ndarray:
    output = np.zeros((len(frame_numbers), 6), dtype=np.float32)
    if camera_motion is None:
        return output
    for offset in range(1, len(frame_numbers)):
        previous = frame_numbers[offset - 1]
        current = frame_numbers[offset]
        output[offset] = camera_motion[previous:current].sum(axis=0)
    return output


def build_sequence_clips(
    sequence: SequenceRecord,
    frame_cache_dir: str | Path,
    output_dir: str | Path,
    sequence_length: int,
    stride: int,
    max_tracks: int,
    max_detections: int,
    feature_dim: int,
    iou_threshold: float,
    dataset_name: str,
    camera_motion: np.ndarray | None = None,
    temporal_intervals: tuple[int, ...] = (1,),
    sampling_seed: int = 0,
    min_track_frames: int = 2,
    detector_source: str = "unknown",
    score_threshold: float = -1.0,
) -> int:
    """Build detector-conditioned clips using GT-first, coverage-preserving sampling.

    Unlike the legacy pipeline, an identity does not need to match a detection
    in the first frame. Detector misses remain explicit ``assignment=-1``
    targets. Crowded windows are partitioned into coverage groups so identities
    beyond ``max_tracks`` are not silently discarded.
    """
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if stride < 1 or max_tracks < 2 or max_detections < 1:
        raise ValueError("stride/max_tracks/max_detections must be positive")
    if min_track_frames < 1:
        raise ValueError("min_track_frames must be positive")
    temporal_intervals = tuple(sorted(set(temporal_intervals)))
    if not temporal_intervals or any(interval < 1 for interval in temporal_intervals):
        raise ValueError("temporal_intervals must contain positive integers")

    frame_cache_dir = Path(frame_cache_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    stable_name_seed = zlib.crc32(sequence.name.encode("utf-8"))
    rng = np.random.default_rng(sampling_seed + stable_name_seed)
    latest_start = sequence.length - sequence_length + 1
    for start in range(1, latest_start + 1, stride):
        feasible = [
            interval
            for interval in temporal_intervals
            if start + (sequence_length - 1) * interval <= sequence.length
        ]
        if not feasible:
            continue
        interval = feasible[int(rng.integers(0, len(feasible)))]
        frame_numbers = [start + offset * interval for offset in range(sequence_length)]

        longevity: dict[int, int] = defaultdict(int)
        visible_frames: dict[int, int] = defaultdict(int)
        for frame_number in frame_numbers:
            frame_gt = sequence.ground_truth.get(frame_number)
            if frame_gt is None:
                continue
            for index, track_id_value in enumerate(frame_gt.track_ids):
                track_id = int(track_id_value)
                longevity[track_id] += 1
                visible_frames[track_id] += int(frame_gt.visibility[index] > 0.10)
        candidate_ids = [
            track_id for track_id, count in longevity.items() if count >= min_track_frames
        ]
        candidate_ids.sort(
            key=lambda track_id: (
                -visible_frames[track_id],
                -longevity[track_id],
                track_id,
            )
        )
        if len(candidate_ids) < 2:
            continue

        for group_index, track_ids in enumerate(_coverage_groups(candidate_ids, max_tracks)):
            tracks = max_tracks
            detections = max_detections
            boxes = np.zeros((sequence_length, tracks, 4), dtype=np.float32)
            existence = np.zeros((sequence_length, tracks), dtype=bool)
            visibility = np.zeros((sequence_length, tracks), dtype=bool)
            slot_valid = np.zeros(tracks, dtype=bool)
            slot_valid[: len(track_ids)] = True
            padded_track_ids = np.full(tracks, -1, dtype=np.int64)
            padded_track_ids[: len(track_ids)] = track_ids
            detection_boxes = np.zeros((sequence_length, detections, 4), dtype=np.float32)
            detection_features = np.zeros(
                (sequence_length, detections, feature_dim), dtype=np.float32
            )
            detection_scores = np.zeros((sequence_length, detections), dtype=np.float32)
            detection_valid = np.zeros((sequence_length, detections), dtype=bool)
            assignment = np.full((sequence_length, tracks), -1, dtype=np.int64)
            appearance_sum = np.zeros((tracks, feature_dim), dtype=np.float32)
            appearance_count = np.zeros(tracks, dtype=np.float32)
            id_to_slot = {track_id: slot for slot, track_id in enumerate(track_ids)}

            for offset, frame_number in enumerate(frame_numbers):
                frame_gt = sequence.ground_truth.get(frame_number)
                lookup = _frame_lookup(frame_gt)
                active_slots: list[int] = []
                gt_boxes_xyxy: list[np.ndarray] = []
                if frame_gt is not None:
                    for track_id, slot in id_to_slot.items():
                        if track_id not in lookup:
                            continue
                        gt_index = lookup[track_id]
                        box_xywh = frame_gt.boxes_xywh[gt_index : gt_index + 1]
                        box_xyxy = xywh_to_xyxy(box_xywh)[0]
                        boxes[offset, slot] = normalize_xyxy(
                            box_xyxy[None], sequence.width, sequence.height
                        )[0]
                        existence[offset, slot] = True
                        visibility[offset, slot] = frame_gt.visibility[gt_index] > 0.10
                        active_slots.append(slot)
                        gt_boxes_xyxy.append(box_xyxy)

                frame_cache = _load_frame_cache(frame_cache_dir / f"{frame_number:06d}.npz")
                order = frame_cache.scores.argsort(descending=True)[:detections]
                selected_boxes = frame_cache.boxes_xyxy[order].numpy()
                selected_scores = frame_cache.scores[order].numpy()
                selected_features = frame_cache.features[order].numpy()
                count = len(order)
                if selected_features.shape[-1] != feature_dim:
                    raise ValueError(
                        f"Cached feature dim {selected_features.shape[-1]} "
                        f"!= requested {feature_dim}"
                    )
                if count:
                    detection_boxes[offset, :count] = normalize_xyxy(
                        selected_boxes, sequence.width, sequence.height
                    )
                    detection_features[offset, :count] = selected_features
                    detection_scores[offset, :count] = selected_scores
                    detection_valid[offset, :count] = True
                if active_slots and count:
                    matches = match_ground_truth(
                        np.asarray(gt_boxes_xyxy), selected_boxes, iou_threshold
                    )
                    for local_gt, detection_index in matches:
                        slot = active_slots[local_gt]
                        assignment[offset, slot] = detection_index
                        appearance_sum[slot] += selected_features[detection_index]
                        appearance_count[slot] += 1

            # Continue the last known state through missed/dead frames; masks
            # decide which targets contribute to losses.
            for slot in range(len(track_ids)):
                for offset in range(1, sequence_length):
                    if not existence[offset, slot]:
                        boxes[offset, slot] = boxes[offset - 1, slot]
            appearance = appearance_sum / np.maximum(appearance_count[:, None], 1)
            delta_time = np.ones(sequence_length, dtype=np.float32)
            delta_time[1:] = np.diff(frame_numbers)
            clip_path = output_dir / (
                f"{dataset_name}_{sequence.name}_{start:06d}_i{interval}_g{group_index:02d}.npz"
            )
            np.savez_compressed(
                clip_path,
                boxes=boxes,
                slot_valid=slot_valid,
                existence=existence,
                visibility=visibility,
                detection_boxes=detection_boxes,
                detection_features=detection_features,
                detection_scores=detection_scores,
                detection_valid=detection_valid,
                assignment=assignment,
                appearance=appearance,
                camera_motion=_aggregate_camera_motion(camera_motion, frame_numbers),
                delta_time=delta_time,
                track_ids=padded_track_ids,
                frame_numbers=np.asarray(frame_numbers, dtype=np.int64),
                dataset=np.asarray(dataset_name),
                sequence=np.asarray(sequence.name),
                start_frame=np.asarray(start),
                temporal_interval=np.asarray(interval),
                sequence_length=np.asarray(sequence_length),
                max_tracks=np.asarray(max_tracks),
                max_detections=np.asarray(max_detections),
                min_track_frames=np.asarray(min_track_frames),
                iou_threshold=np.asarray(iou_threshold, dtype=np.float32),
                score_threshold=np.asarray(score_threshold, dtype=np.float32),
                detector_source=np.asarray(detector_source),
                sampling_seed=np.asarray(sampling_seed),
                protocol_version=np.asarray(2),
            )
            written += 1
    return written
