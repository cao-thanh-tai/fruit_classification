# Fruit Classification - Web Context

## Mục tiêu

Xây dựng giao diện web cho phép người dùng tải ảnh trái cây và nhận:

- Loại trái cây được dự đoán.
- Confidence tương ứng.

Đây là bài toán phân loại ảnh với 10 class:

`apple`, `avocado`, `banana`, `cherry`, `kiwi`, `mango`, `orange`,
`pinenapple`, `strawberries`, `watermelon`.

## Model

Model ONNX nằm tại:

```text
model/fruit_classifier.onnx
```

Model được huấn luyện bằng ResNet18 transfer learning. Web chỉ cần dùng
`onnxruntime-web`; không cần PyTorch hoặc checkpoint `.pt`.

## Input contract

Metadata đầy đủ nằm trong `model/metadata.json`.

- Input name: `images`
- Input type: `float32`
- Layout: `NCHW`
- Shape: `[1, 3, 224, 224]`
- Color order: RGB
- Pixel range trước normalize: `[0, 1]`
- Mean: `[0.485, 0.456, 0.406]`
- Std: `[0.229, 0.224, 0.225]`

Preprocessing:

1. Đọc ảnh và chuyển sang RGB.
2. Resize về `224 x 224`.
3. Chuyển pixel từ `[0, 255]` sang `[0, 1]`.
4. Normalize từng channel bằng mean/std ở trên.
5. Chuyển dữ liệu sang thứ tự `NCHW`.

## Output contract

- Output name: `logits`
- Shape: `[1, 10]`
- Model trả về logits, chưa phải probability.
- Web phải áp dụng softmax.
- Lấy index có probability cao nhất.
- Map index qua `metadata.classes`; không tự sắp xếp lại class.

## Web dependency

```bash
npm install onnxruntime-web
```

## Expected UX

- Hiển thị ảnh người dùng đã chọn.
- Hiển thị class dự đoán và confidence.
- Báo lỗi nếu file không phải ảnh hoặc không thể đọc ảnh.
- Có thể cảnh báo khi confidence thấp.

## Important constraints

- Giữ đúng RGB, resize, normalize và class order.
- Không dùng `RandomHorizontalFlip` hoặc `RandomRotation` khi predict.
- Không đưa file `.pt`, source code training hoặc dataset vào web bundle.
- Giữ spelling `pinenapple` đúng như metadata.
