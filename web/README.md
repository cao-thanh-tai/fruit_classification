# Web integration bundle

This folder is the self-contained handoff package for the web application.
Read [WEB_CONTEXT.md](WEB_CONTEXT.md) before integrating the model.

## Files

- `model/fruit_classifier.onnx`: ResNet18 model. Input is normalized NCHW
  `float32`.
- `model/metadata.json`: class order and preprocessing contract.

The web application only needs to copy the `model/` folder and install
`onnxruntime-web`:

```bash
npm install onnxruntime-web
```

The browser should use ONNX Runtime Web and perform this preprocessing:

1. Load the image and convert it to RGB.
2. Resize it to `metadata.input.image_size` (`224 x 224`).
3. Convert pixels from `[0, 255]` to `[0, 1]`.
4. Normalize each channel with `normalize_mean` and `normalize_std`.
5. Arrange values as RGB `NCHW`: `[1, 3, 224, 224]`.
6. Run the model and apply softmax to the `logits` output.
7. Map the highest output index through `metadata.classes`.

The class order must not be reconstructed alphabetically by the web app; use
the order stored in `model/metadata.json`.

## Suggested inference paths

```text
web/
├── README.md
└── model/
    ├── fruit_classifier.onnx
    └── metadata.json
```

The model output is logits. Apply softmax in the web application, choose the
highest probability, and map that index through `metadata.classes`.
