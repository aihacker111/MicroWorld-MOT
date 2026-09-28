from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile frozen YOLO11 perception")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--detections", type=int, default=40)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
        from ultralytics.utils.torch_utils import get_flops
    except ImportError as error:
        raise RuntimeError("Install the project with the perception extra") from error

    model = YOLO(args.weights).model.eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    detector_gflops = float(get_flops(model, args.image_size))
    crop_upper_gflops = float(get_flops(model, args.embedding_size))
    print(
        json.dumps(
            {
                "weights": args.weights,
                "parameters": parameters,
                "detector_image_size": args.image_size,
                "detector_gflops": detector_gflops,
                "embedding_size": args.embedding_size,
                "embedding_gflops_per_crop_upper_bound": crop_upper_gflops,
                "detections": args.detections,
                "detector_plus_embeddings_gflops_upper_bound": detector_gflops
                + args.detections * crop_upper_gflops,
                "note": (
                    "The same weights are reused. The crop number affects compute, not parameter "
                    "count; embedding exits before Detect, so full-model crop GFLOPs are an upper bound."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
