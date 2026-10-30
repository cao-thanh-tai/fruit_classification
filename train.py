"""Configurable experiment entry point for training the fruit classifier."""

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from src.models.train_model import train_model


CONFIG_PATH = Path("config/config.yaml")


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fruit classifier.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    training_config = config["training"]

    def override(name: str, value: Any) -> Any:
        return value if value is not None else training_config[name]

    output_dir = args.output_dir or Path(training_config.get("output_dir", "models/resnet18"))
    experiment_config = {
        "data": data_config,
        "training": {
            **training_config,
            "epochs": override("epochs", args.epochs),
            "batch_size": override("batch_size", args.batch_size),
            "learning_rate": override("learning_rate", args.learning_rate),
            "num_workers": override("num_workers", args.num_workers),
            "pretrained": bool(training_config["pretrained"]) and not args.no_pretrained,
            "mixed_precision": bool(training_config["mixed_precision"])
            and not args.no_mixed_precision,
            "resume": bool(training_config["resume"]) or args.resume,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(experiment_config, indent=2),
        encoding="utf-8",
    )

    train_model(
        train_dir=Path(data_config["train_dir"]),
        test_dir=Path(data_config["test_dir"]),
        output_dir=output_dir,
        image_size=training_config["image_size"],
        validation_ratio=training_config["validation_ratio"],
        batch_size=experiment_config["training"]["batch_size"],
        epochs=experiment_config["training"]["epochs"],
        learning_rate=experiment_config["training"]["learning_rate"],
        seed=training_config["seed"],
        num_workers=experiment_config["training"]["num_workers"],
        pretrained=experiment_config["training"]["pretrained"],
        weight_decay=training_config["weight_decay"],
        scheduler_factor=training_config["scheduler_factor"],
        scheduler_patience=training_config["scheduler_patience"],
        early_stopping_patience=training_config["early_stopping_patience"],
        mixed_precision=experiment_config["training"]["mixed_precision"],
        resume=experiment_config["training"]["resume"],
    )


if __name__ == "__main__":
    main()
