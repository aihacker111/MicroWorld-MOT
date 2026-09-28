from __future__ import annotations

import argparse
from pathlib import Path

from microworld_mot.data.cache_builder import (
    build_sequence_clips,
    cache_sequence_detections,
    estimate_camera_motion,
)
from microworld_mot.data.sequences import scan_motchallenge, scan_visdrone
from microworld_mot.perception import YOLO11Perception


def _parse_classes(raw: str) -> set[int] | None:
    if not raw:
        return None
    return {int(value) for value in raw.split(",")}


def _deduplicate_mot17(sequences):
    selected = {}
    for sequence in sequences:
        parts = sequence.name.split("-")
        canonical = "-".join(parts[:2]) if sequence.name.startswith("MOT17-") else sequence.name
        if canonical not in selected:
            sequence.name = canonical
            selected[canonical] = sequence
    return list(selected.values())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache a frozen detector and build real MicroWorld-MOT training clips"
    )
    parser.add_argument("--format", choices=["mot", "visdrone"], required=True)
    parser.add_argument("--root", required=True, help="One train/val split root")
    parser.add_argument("--dataset-name", required=True, help="mot17, mot16, dancetrack, visdrone")
    parser.add_argument("--output", default="", help="Output directory containing clip .npz files")
    parser.add_argument("--cache", required=True, help="Resumable per-frame detector cache")
    parser.add_argument(
        "--detector",
        choices=["yolo11n", "public"],
        default="yolo11n",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-threshold", type=float, default=0.15)
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-tracks", type=int, default=64)
    parser.add_argument("--max-detections", type=int, default=128)
    parser.add_argument("--gt-classes", default="", help="Comma-separated source GT classes")
    parser.add_argument("--coco-labels", default="", help="Comma-separated COCO detector labels")
    parser.add_argument(
        "--include-sequences",
        default="",
        help="Optional comma-separated exact sequence names for a leak-free split",
    )
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Cache detections for test data without requiring GT or building clips",
    )
    parser.add_argument("--estimate-camera-motion", action="store_true")
    parser.add_argument(
        "--keep-mot17-detector-duplicates",
        action="store_true",
        help="Keep DPM/FRCNN/SDP copies even when running a new detector",
    )
    args = parser.parse_args()

    gt_classes = _parse_classes(args.gt_classes)
    if args.format == "mot":
        sequences = scan_motchallenge(
            args.root, gt_classes or {1}, require_ground_truth=not args.cache_only
        )
    else:
        sequences = scan_visdrone(
            args.root, gt_classes, require_ground_truth=not args.cache_only
        )
    if (
        args.dataset_name.lower() == "mot17"
        and args.detector != "public"
        and not args.keep_mot17_detector_duplicates
    ):
        sequences = _deduplicate_mot17(sequences)
    if args.include_sequences:
        allowed_sequences = set(args.include_sequences.split(","))
        sequences = [sequence for sequence in sequences if sequence.name in allowed_sequences]
        if not sequences:
            raise ValueError("--include-sequences did not match any sequence")

    if args.coco_labels:
        coco_labels = _parse_classes(args.coco_labels)
    elif args.format == "mot":
        coco_labels = {1}  # person
    else:
        coco_labels = {1, 2, 3, 4, 6, 8}  # people and common road vehicles
    perception = YOLO11Perception(
        weights="yolo11n.pt",
        device=args.device,
        score_threshold=args.score_threshold,
        allowed_coco_labels=coco_labels,
    )
    if not args.cache_only and not args.output:
        raise ValueError("--output is required unless --cache-only is set")
    output_root = Path(args.output) if args.output else None
    cache_root = Path(args.cache)
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    total = 0
    for index, sequence in enumerate(sequences, start=1):
        print(f"[{index}/{len(sequences)}] cache {sequence.name}", flush=True)
        frame_cache = cache_sequence_detections(
            sequence,
            cache_root,
            perception,
            args.detector,
            overwrite=args.overwrite_cache,
        )
        if args.cache_only:
            continue
        camera_motion = estimate_camera_motion(sequence) if args.estimate_camera_motion else None
        clips = build_sequence_clips(
            sequence=sequence,
            frame_cache_dir=frame_cache,
            output_dir=output_root,
            sequence_length=args.sequence_length,
            stride=args.stride,
            max_tracks=args.max_tracks,
            max_detections=args.max_detections,
            feature_dim=perception.feature_dim,
            iou_threshold=args.iou_threshold,
            dataset_name=args.dataset_name,
            camera_motion=camera_motion,
        )
        total += clips
        print(f"[{index}/{len(sequences)}] {sequence.name}: {clips} clips", flush=True)
    if args.cache_only:
        print(f"cache_complete={cache_root.resolve()}")
    else:
        print(f"total_clips={total} output={output_root.resolve()}")


if __name__ == "__main__":
    main()
