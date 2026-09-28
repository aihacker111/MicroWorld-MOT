# ruff: noqa: E402 -- make direct `python tools/...` execution work without installation.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from microworld_mot.config import config_from_dict
from microworld_mot.data.cache_builder import estimate_camera_motion, normalize_xyxy
from microworld_mot.data.sequences import scan_motchallenge, scan_visdrone
from microworld_mot.models import build_system
from microworld_mot.models.registry import resolved_config
from microworld_mot.tracking import MicroWorldTracker
from microworld_mot.tracking.tracker import cxcylogwh_to_xyxy
from microworld_mot.types import Detections


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
        description="Run MicroWorld-MOT on a cached MOTChallenge or VisDrone split"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--format", choices=["mot", "visdrone"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--cache", required=True, help="Per-frame cache made by prepare_real_data.py"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-threshold", type=float, default=0.15)
    parser.add_argument("--adaptive-detector", action="store_true")
    parser.add_argument("--estimate-camera-motion", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = config_from_dict(checkpoint["config"])
    config.model = resolved_config(config.model)
    system = build_system(config.model)
    system.load_state_dict(checkpoint["model"])
    sequences = (
        scan_motchallenge(args.root, {1}, require_ground_truth=False)
        if args.format == "mot"
        else scan_visdrone(args.root, require_ground_truth=False)
    )
    if any(sequence.name.startswith("MOT17-") for sequence in sequences):
        sequences = _deduplicate_mot17(sequences)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    for sequence_index, sequence in enumerate(sequences, start=1):
        tracker = MicroWorldTracker(system, config.tracker, device)
        rows: list[list[float]] = []
        sequence_cache = Path(args.cache) / sequence.name
        camera_tokens = (
            estimate_camera_motion(sequence)
            if args.estimate_camera_motion
            else np.zeros((sequence.length, 6), dtype=np.float32)
        )
        for frame in range(1, sequence.length + 1):
            run_detector = not args.adaptive_detector or tracker.should_run_detector()
            if run_detector:
                cache_path = sequence_cache / f"{frame:06d}.npz"
                if not cache_path.exists():
                    raise FileNotFoundError(f"Missing detector cache: {cache_path}")
                with np.load(cache_path) as cached:
                    keep = cached["scores"] >= args.score_threshold
                    boxes_xyxy = cached["boxes_xyxy"][keep]
                    boxes = torch.from_numpy(
                        normalize_xyxy(boxes_xyxy, sequence.width, sequence.height)
                    ).float()
                    scores = torch.from_numpy(cached["scores"][keep]).float()
                    features = torch.from_numpy(cached["features"][keep]).float()
                detections = Detections(boxes=boxes, scores=scores, appearance=features)
            else:
                detections = None
            camera_motion = torch.from_numpy(camera_tokens[frame - 1]).to(device).unsqueeze(0)
            for track in tracker.step(detections, camera_motion=camera_motion):
                xyxy = cxcylogwh_to_xyxy(track.box.unsqueeze(0), (sequence.height, sequence.width))[
                    0
                ]
                left, top, right, bottom = [float(value) for value in xyxy]
                rows.append(
                    [
                        frame,
                        track.track_id,
                        left,
                        top,
                        right - left,
                        bottom - top,
                        track.score,
                        -1,
                        -1,
                        -1,
                    ]
                )
        destination = output_root / f"{sequence.name}.txt"
        if rows:
            np.savetxt(destination, np.asarray(rows), delimiter=",", fmt="%.6f")
        else:
            destination.write_text("", encoding="utf-8")
        print(
            f"[{sequence_index}/{len(sequences)}] {sequence.name}: "
            f"{len(rows)} rows -> {destination}"
        )


if __name__ == "__main__":
    main()
