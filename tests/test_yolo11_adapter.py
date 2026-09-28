from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import torch
from PIL import Image

from microworld_mot.perception import YOLO11Perception


class _FakeYOLO:
    def __init__(self, weights: str) -> None:
        assert weights == "yolo11n.pt"

    def predict(self, **kwargs):
        assert kwargs["classes"] == [0]
        boxes = SimpleNamespace(
            xyxy=torch.tensor([[2.0, 3.0, 18.0, 27.0]]),
            conf=torch.tensor([0.9]),
            cls=torch.tensor([0.0]),
        )
        return [SimpleNamespace(boxes=boxes)]

    def embed(self, source, **kwargs):
        assert kwargs["imgsz"] == 128
        vector = torch.arange(1, 257, dtype=torch.float32)
        return [vector.clone() for _ in source]


def test_yolo11_is_single_detector_and_embedding_backend(tmp_path, monkeypatch) -> None:
    ultralytics = ModuleType("ultralytics")
    ultralytics.YOLO = _FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)

    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (32, 32), color=(127, 127, 127)).save(image_path)
    perception = YOLO11Perception(allowed_coco_labels={1})
    output = perception.detect(image_path)

    assert output.boxes_xyxy.shape == (1, 4)
    assert output.labels.tolist() == [1]
    assert output.features.shape == (1, 256)
    assert torch.allclose(output.features.norm(dim=-1), torch.ones(1))
