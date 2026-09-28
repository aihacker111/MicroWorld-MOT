from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class FrameObjects:
    track_ids: np.ndarray
    boxes_xywh: np.ndarray
    visibility: np.ndarray
    labels: np.ndarray


@dataclass
class SequenceRecord:
    name: str
    image_paths: list[Path]
    width: int
    height: int
    ground_truth: dict[int, FrameObjects]
    public_detections: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.image_paths)


def _load_csv(path: Path) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, 10), dtype=np.float32)
    return np.loadtxt(path, delimiter=",", ndmin=2, dtype=np.float32)


def _mot_frame_objects(rows: np.ndarray, classes: set[int] | None) -> dict[int, FrameObjects]:
    output: dict[int, FrameObjects] = {}
    if not len(rows):
        return output
    keep = rows[:, 6] > 0
    if rows.shape[1] > 7 and classes:
        keep &= np.isin(rows[:, 7].astype(int), list(classes))
    rows = rows[keep]
    for frame in np.unique(rows[:, 0].astype(int)):
        frame_rows = rows[rows[:, 0].astype(int) == frame]
        labels = (
            frame_rows[:, 7].astype(np.int64)
            if frame_rows.shape[1] > 7
            else np.ones(len(frame_rows), dtype=np.int64)
        )
        visibility = (
            frame_rows[:, 8]
            if frame_rows.shape[1] > 8
            else np.ones(len(frame_rows), dtype=np.float32)
        )
        output[frame] = FrameObjects(
            track_ids=frame_rows[:, 1].astype(np.int64),
            boxes_xywh=frame_rows[:, 2:6].astype(np.float32),
            visibility=visibility.astype(np.float32),
            labels=labels,
        )
    return output


def scan_motchallenge(
    root: str | Path,
    classes: set[int] | None = None,
    require_ground_truth: bool = True,
) -> list[SequenceRecord]:
    """Read MOT16, MOT17 or DanceTrack directories.

    ``root`` must directly contain sequence folders. DanceTrack and MOTChallenge
    share the same frame/id/tlwh annotation layout.
    """
    root = Path(root)
    sequences: list[SequenceRecord] = []
    for sequence in sorted(path for path in root.iterdir() if path.is_dir()):
        image_dir = sequence / "img1"
        gt_path = sequence / "gt" / "gt.txt"
        if not image_dir.is_dir() or (require_ground_truth and not gt_path.exists()):
            continue
        image_paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not image_paths:
            continue
        info = configparser.ConfigParser()
        info.read(sequence / "seqinfo.ini")
        if "Sequence" in info:
            width = int(info["Sequence"].get("imWidth", 0))
            height = int(info["Sequence"].get("imHeight", 0))
        else:
            from PIL import Image

            with Image.open(image_paths[0]) as image:
                width, height = image.size
        ground_truth = _mot_frame_objects(_load_csv(gt_path), classes)
        public: dict[int, np.ndarray] = {}
        detection_rows = _load_csv(sequence / "det" / "det.txt")
        for frame in np.unique(detection_rows[:, 0].astype(int)) if len(detection_rows) else []:
            frame_rows = detection_rows[detection_rows[:, 0].astype(int) == frame]
            public[int(frame)] = frame_rows[:, 2:7].astype(np.float32)
        sequences.append(
            SequenceRecord(
                name=sequence.name,
                image_paths=image_paths,
                width=width,
                height=height,
                ground_truth=ground_truth,
                public_detections=public,
            )
        )
    if not sequences:
        raise FileNotFoundError(f"No MOTChallenge sequences found under {root}")
    return sequences


def scan_visdrone(
    root: str | Path,
    classes: set[int] | None = None,
    require_ground_truth: bool = True,
) -> list[SequenceRecord]:
    """Read a VisDrone MOT split containing ``sequences`` and ``annotations``."""
    root = Path(root)
    sequence_root = root / "sequences"
    annotation_root = root / "annotations"
    if not sequence_root.is_dir() or not annotation_root.is_dir():
        raise FileNotFoundError(
            f"Expected {sequence_root} and {annotation_root}; pass one VisDrone split root"
        )
    allowed = classes or set(range(1, 11))
    sequences: list[SequenceRecord] = []
    for image_dir in sorted(path for path in sequence_root.iterdir() if path.is_dir()):
        image_paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        annotation_path = annotation_root / f"{image_dir.name}.txt"
        if not image_paths or (require_ground_truth and not annotation_path.exists()):
            continue
        from PIL import Image

        with Image.open(image_paths[0]) as image:
            width, height = image.size
        rows = _load_csv(annotation_path)
        # VisDrone-MOT: frame, id, left, top, width, height, score, class,
        # truncation, occlusion. Ignore regions/classes outside 1..10 are removed.
        keep = (rows[:, 6] > 0) & np.isin(rows[:, 7].astype(int), list(allowed))
        rows = rows[keep]
        ground_truth: dict[int, FrameObjects] = {}
        for frame in np.unique(rows[:, 0].astype(int)):
            frame_rows = rows[rows[:, 0].astype(int) == frame]
            occlusion = frame_rows[:, 9] if frame_rows.shape[1] > 9 else 0
            visibility = 1.0 - np.asarray(occlusion, dtype=np.float32) / 2.0
            ground_truth[int(frame)] = FrameObjects(
                track_ids=frame_rows[:, 1].astype(np.int64),
                boxes_xywh=frame_rows[:, 2:6].astype(np.float32),
                visibility=np.clip(visibility, 0, 1),
                labels=frame_rows[:, 7].astype(np.int64),
            )
        sequences.append(
            SequenceRecord(
                name=image_dir.name,
                image_paths=image_paths,
                width=width,
                height=height,
                ground_truth=ground_truth,
            )
        )
    if not sequences:
        raise FileNotFoundError(f"No VisDrone MOT sequences found under {root}")
    return sequences
