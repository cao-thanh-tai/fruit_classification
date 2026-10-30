# Experiment Log

Tài liệu này ghi lại các lần phân tích, huấn luyện và các quyết định kỹ thuật.
Mỗi lần thay đổi model, learning rate, transform hoặc hyperparameter đáng kể
thì thêm một entry mới; không xóa các entry cũ.

## Quy ước

- Mỗi experiment có một ID tăng dần: `EXP-000`, `EXP-001`, ...
- Ghi cả thử nghiệm thất bại để tránh lặp lại cùng một hướng.
- Chỉ ghi kết quả đã quan sát được; thông tin chưa thử phải ghi là `chưa có`.
- Không dùng test set để chọn hyperparameter.

## Experiment index

| ID | Ngày | Loại | Model | Kết quả chính | Trạng thái |
|---|---|---|---|---|---|
| `EXP-000` | 2026-10-30 | EDA baseline | Chưa train | 10 class, 25 ảnh có alpha channel | completed |
| `EXP-001` | 2026-10-30 | Training pipeline | ResNet18 transfer learning | Chưa chạy | planned |

---

## EXP-000 - EDA baseline

- **Nguồn:** `notebooks/eda.ipynb`
- **Mục tiêu:** kiểm tra cấu trúc dataset, phân bố số lượng ảnh và format
  kênh màu trước khi huấn luyện.
- **Trạng thái:** completed
- **Model:** chưa train

### Dataset

Dataset có 10 thư mục lớp trong cả `train` và `test`:

`apple`, `avocado`, `banana`, `cherry`, `kiwi`, `mango`, `orange`,
`pinenapple`, `strawberries`, `watermelon`.

| Class | Train | Test |
|---|---:|---:|
| `apple` | 230 | 89 |
| `avocado` | 230 | 105 |
| `banana` | 230 | 105 |
| `cherry` | 230 | 105 |
| `kiwi` | 230 | 105 |
| `mango` | 231 | 105 |
| `orange` | 227 | 97 |
| `pinenapple` | 230 | 105 |
| `strawberries` | 230 | 103 |
| `watermelon` | 230 | 105 |
| **Tổng** | **2,298** | **1,024** |

Phân bố train gần cân bằng. Test có chênh lệch nhẹ, chủ yếu do `apple` có
89 ảnh và `orange` có 97 ảnh.

### Kết quả kiểm tra ảnh

- Notebook đặt kích thước ảnh dự kiến là `IMG_SIZE = 224`.
- Phát hiện **25 ảnh không phải RGB chuẩn**.
- Toàn bộ 25 ảnh được báo lỗi vì có **alpha channel**.
- EDA không ghi nhận ảnh grayscale hoặc ảnh nhỏ hơn 32 pixel trong output hiện
  tại.

### Quyết định từ EDA

1. Giữ nguyên 10 tên lớp theo dataset, bao gồm spelling `pinenapple`.
2. Pipeline đọc ảnh phải chuẩn hóa về 3 kênh RGB, xử lý alpha channel trước khi
   đưa vào model.
3. Dùng kích thước `224 x 224` làm baseline đầu tiên vì đây là giá trị đã đặt
   trong notebook.
4. Chưa quyết định model, optimizer, learning rate, transform augmentation,
   batch size hoặc số epoch; các lựa chọn này phải được ghi ở experiment train
   tiếp theo.

### Hạn chế của EDA hiện tại

- Chưa có kết quả huấn luyện hoặc metric đánh giá.
- Chưa có validation split được ghi nhận.
- Chưa có thống kê về kích thước gốc, background, duplicate hoặc leakage.
- Kiểm tra ảnh hiện tại tập trung vào shape/kênh màu; cần xác nhận pipeline
  chuyển đổi RGB hoạt động đúng.

---

## EXP-001 - ResNet18 training baseline

- **Mục tiêu:** tạo pipeline train chuẩn trên file Python, không dùng test set
  trong quá trình chọn model.
- **Trạng thái:** planned
- **Code:** `src/models/train_model.py`
- **Model:** ResNet18, pretrained ImageNet, thay classifier cuối thành 10 lớp.
- **Input image size:** `224 x 224`
- **Train/validation split:** 80/20 từ `data/raw/train`
- **Test:** giữ nguyên `data/raw/test` để đánh giá cuối.
- **Train transform:** resize, random horizontal flip, random rotation 10 độ,
  tensor conversion và ImageNet normalization.
- **Validation/test transform:** resize, tensor conversion và ImageNet
  normalization; không dùng augmentation ngẫu nhiên.
- **Loss:** CrossEntropyLoss, chưa dùng class weight vì train gần cân bằng.
- **Optimizer:** Adam
- **Learning rate:** `0.001`
- **Batch size:** `32`
- **Epochs:** `20`
- **Checkpoint:** lưu model có validation accuracy tốt nhất.
- **Artifacts:** `best_model.pt`, `history.csv`, `metrics.json`.
- **Kết quả:** chưa chạy.
- **Production safeguards:** config được lưu cùng artifact; có scheduler,
  early stopping, resume từ `last_model.pt`, logging và mixed precision trên
  CUDA.
- **Quyết định tiếp theo:** chạy baseline, ghi metric và confusion matrix vào
  entry này trước khi thay đổi model hoặc hyperparameter.

---

## Experiment template

### EXP-XXX - Tên experiment

- **Ngày:**
- **Mục tiêu:**
- **Trạng thái:** planned / running / completed / failed
- **Code hoặc notebook:**
- **Seed:**

#### Dữ liệu

- Dataset/split:
- Số ảnh train/validation/test:
- Có class imbalance không:
- Có data leakage hoặc vấn đề dữ liệu không:

#### Model

- Kiến trúc:
- Pretrained hay train from scratch:
- Input image size:
- Số parameters, nếu biết:

#### Transform và preprocessing

- Train transform:
- Validation/test transform:
- Normalize:
- Augmentation:
- Xử lý alpha channel:
- Lý do chọn:

#### Training configuration

- Loss function:
- Optimizer:
- Learning rate:
- Scheduler:
- Batch size:
- Số epoch:
- Early stopping:
- Hardware/runtime:

#### Kết quả

- Best epoch:
- Train loss/metric:
- Validation loss/metric:
- Test accuracy:
- Precision/recall/F1:
- Kết quả theo từng class:
- Confusion matrix:
- Thời gian train:

#### Phân tích

- Điều gì hoạt động tốt:
- Điều gì không hoạt động:
- Dấu hiệu overfitting/underfitting:
- Class thường bị nhầm:
- Hạn chế của kết quả:

#### Quyết định tiếp theo

- Giữ nguyên:
- Thay đổi:
- Lý do:

---

## Chosen configuration

Chỉ cập nhật phần này sau khi đã có kết quả đủ tin cậy.

- **Model:** ResNet18 pretrained (planned, `EXP-001`)
- **Image size:** `224 x 224` (EDA baseline)
- **Train transform:** resize, horizontal flip, rotation, ImageNet normalize
- **Validation/test transform:** resize, ImageNet normalize
- **Optimizer:** Adam
- **Learning rate:** `0.001`
- **Scheduler:** chưa dùng
- **Batch size:** `32`
- **Epochs:** `20`
- **Xử lý alpha channel:** `ImageFolder` chuyển ảnh qua PIL RGB loader
- **Lý do chọn:** cấu hình baseline trước khi chạy experiment
- **Experiment tham chiếu:** `EXP-001` (planned)
