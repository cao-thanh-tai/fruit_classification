# Fruit Classification

## Training

The training pipeline uses ResNet18 transfer learning. It creates a validation
split from `data/raw/train`, keeps `data/raw/test` for final evaluation, and
saves artifacts under `models/resnet18/`. Defaults are loaded from
`config/config.yaml`.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run training from the project root. Chỉnh các tham số mặc định ở đầu file
`train.py` hoặc truyền bằng command line:

```powershell
python train.py
```

Useful overrides:

```powershell
python train.py --epochs 20 --batch-size 32 --learning-rate 0.001
```

For a resumable run, set `resume: true` in the config or use:

```powershell
python train.py --resume
```

The pipeline also supports early stopping, `ReduceLROnPlateau`, optional CUDA
mixed precision, and configurable data-loader workers.

The command saves:

- `models/resnet18/best_model.pt`
- `models/resnet18/last_model.pt`
- `models/resnet18/history.csv`
- `models/resnet18/metrics.json`
- `models/resnet18/config.json`

Use `--no-pretrained` when ImageNet weights are not available locally.

## Documentation

- [Project context](PROJECT_CONTEXT.md)
- [Experiment log](reports/EXPERIMENT_LOG.md)

## Export for web inference

After training, export the best checkpoint:

```powershell
python export_onnx.py
```

The export is written to `models/resnet18/web_bundle/`. The self-contained
handoff package for the web application is copied to `web/model/` and contains
the ONNX model plus `metadata.json`. The preprocessing contract is documented
in [web/README.md](web/README.md).