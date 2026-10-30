"""Export the best PyTorch checkpoint to an ONNX bundle for web inference."""

import argparse
import json
from pathlib import Path

import torch

from src.models.model import build_model


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the fruit model to ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/resnet18/best_model.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/resnet18/web_bundle"),
    )
    parser.add_argument("--opset-version", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    classes = checkpoint["classes"]
    image_size = int(checkpoint["image_size"])
    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output_dir / "fruit_classifier.onnx"
    metadata_path = args.output_dir / "metadata.json"
    dummy_input = torch.randn(1, 3, image_size, image_size)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=args.opset_version,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes={
                "images": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            dynamo=False,
        )

    metadata = {
        "model_name": "fruit_classifier",
        "architecture": "resnet18",
        "input": {
            "name": "images",
            "layout": "NCHW",
            "color_order": "RGB",
            "image_size": image_size,
            "dtype": "float32",
            "normalize_mean": MEAN,
            "normalize_std": STD,
        },
        "output": {
            "name": "logits",
            "activation": "softmax",
            "description": "Apply softmax to logits in the web client.",
        },
        "classes": classes,
        "class_to_index": {name: index for index, name in enumerate(classes)},
        "opset_version": args.opset_version,
        "source_checkpoint": str(args.checkpoint),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Exported model: {onnx_path}")
    print(f"Exported metadata: {metadata_path}")


if __name__ == "__main__":
    main()
