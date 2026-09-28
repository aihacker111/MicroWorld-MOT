from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from .types import DetectorOutput


class YOLO11Perception:
    """Frozen YOLO11n detector and crop-embedding backend.

    One Ultralytics YOLO11n checkpoint provides both COCO detections and the
    appearance vectors. Crop embeddings come from YOLO11n's penultimate layer,
    so this adapter does not carry a separate MobileNet/ReID backbone.

    Public labels intentionally use one-based COCO IDs at the project boundary
    (person=1). They are converted to Ultralytics' zero-based class IDs
    internally.
    """

    FEATURE_DIM = 256

    def __init__(
        self,
        weights: str = "yolo11n.pt",
        device: str = "cpu",
        score_threshold: float = 0.20,
        allowed_coco_labels: set[int] | None = None,
        image_size: int = 640,
        embedding_size: int = 128,
        crop_batch_size: int = 64,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is required for YOLO11 perception. "
                "Install the project with `python -m pip install -e '.[perception]'`."
            ) from error

        if Path(weights).name != "yolo11n.pt":
            raise ValueError(
                "This configuration uses the native 256-D YOLO11n embedding; "
                "weights must be yolo11n.pt."
            )
        if allowed_coco_labels and any(label < 1 for label in allowed_coco_labels):
            raise ValueError("COCO labels must be one-based (person=1)")

        self.model = YOLO(weights)
        self.device = device
        self.score_threshold = score_threshold
        self.allowed_coco_labels = allowed_coco_labels
        self.image_size = image_size
        self.embedding_size = embedding_size
        self.crop_batch_size = crop_batch_size
        self.feature_dim = self.FEATURE_DIM

    @staticmethod
    def _read_image(path: str | Path):
        from PIL import Image

        with Image.open(path) as source:
            return source.convert("RGB").copy()

    @torch.inference_mode()
    def _encode_crops(self, image, boxes: Tensor) -> Tensor:
        width, height = image.size
        crops = []
        for box in boxes.detach().cpu():
            x1 = max(0, min(int(torch.floor(box[0]).item()), width - 1))
            y1 = max(0, min(int(torch.floor(box[1]).item()), height - 1))
            x2 = max(x1 + 1, min(int(torch.ceil(box[2]).item()), width))
            y2 = max(y1 + 1, min(int(torch.ceil(box[3]).item()), height))
            crops.append(image.crop((x1, y1, x2, y2)))

        if not crops:
            return torch.empty((0, self.feature_dim), dtype=torch.float32)

        outputs: list[Tensor] = []
        for start in range(0, len(crops), self.crop_batch_size):
            batch = crops[start : start + self.crop_batch_size]
            embeddings = self.model.embed(
                source=batch,
                imgsz=self.embedding_size,
                device=self.device,
                verbose=False,
            )
            batch_features = torch.stack(
                [torch.as_tensor(vector).detach().float().reshape(-1) for vector in embeddings]
            )
            if batch_features.shape[-1] != self.feature_dim:
                raise RuntimeError(
                    "Unexpected YOLO11n embedding width: "
                    f"expected {self.feature_dim}, got {batch_features.shape[-1]}"
                )
            outputs.append(batch_features.cpu())
        return torch.nn.functional.normalize(torch.cat(outputs), dim=-1)

    @torch.inference_mode()
    def detect(self, image_path: str | Path) -> DetectorOutput:
        classes = None
        if self.allowed_coco_labels:
            classes = sorted(label - 1 for label in self.allowed_coco_labels)
        result = self.model.predict(
            source=str(image_path),
            conf=self.score_threshold,
            classes=classes,
            imgsz=self.image_size,
            device=self.device,
            embed=None,
            verbose=False,
        )[0]
        boxes = result.boxes.xyxy.detach().cpu().float()
        scores = result.boxes.conf.detach().cpu().float()
        labels = result.boxes.cls.detach().cpu().long() + 1
        features = self._encode_crops(self._read_image(image_path), boxes)
        return DetectorOutput(
            boxes_xyxy=boxes,
            scores=scores,
            labels=labels,
            features=features,
        )

    @torch.inference_mode()
    def encode_boxes(self, image_path: str | Path, boxes_xyxy: Tensor) -> Tensor:
        return self._encode_crops(self._read_image(image_path), boxes_xyxy)
