# Fruit Classification - Project Context

Tài liệu này mô tả bài toán, dữ liệu và các ràng buộc của project trước khi
triển khai model. Chỉ cập nhật khi hiểu biết nền tảng hoặc phạm vi bài toán
thay đổi.

## 1. Mục tiêu

- Xây dựng model phân loại ảnh trái cây.
- Input: một ảnh trái cây.
- Output: tên loại trái cây và xác suất dự đoán.
- Loại bài toán: multi-class image classification.

## 2. Các lớp cần phân loại

Hiện tại dataset có 10 lớp:

- `apple`
- `avocado`
- `banana`
- `cherry`
- `kiwi`
- `mango`
- `orange`
- `pinenapple`
- `strawberries`
- `watermelon`

> Giữ nguyên tên lớp theo dataset, bao gồm cả spelling `pinenapple`, trừ khi
> có quyết định thay đổi được ghi trong `reports/EXPERIMENT_LOG.md`.

## 3. Cấu trúc dữ liệu

- `data/raw/train/<class_name>/`: ảnh dùng để huấn luyện.
- `data/raw/test/<class_name>/`: ảnh dùng để đánh giá.
- `data/predict/`: ảnh dùng để thử dự đoán riêng lẻ.
- `data/real/`: ảnh thực tế dùng để kiểm tra khả năng tổng quát hóa, nếu có.
- `data/processed/`: dữ liệu sau tiền xử lý, nếu pipeline tạo ra.

Không dùng test set để huấn luyện hoặc chọn hyperparameter.

## 4. Kết luận EDA

Các kết luận dưới đây được trích từ `notebooks/eda.ipynb` và ghi chi tiết
trong `reports/EXPERIMENT_LOG.md` ở `EXP-000`:

- Dataset có 10 lớp.
- Train có tổng cộng 2.298 ảnh; mỗi lớp có 227–231 ảnh.
- Test có tổng cộng 1.024 ảnh; mỗi lớp có 89–105 ảnh.
- Phân bố train gần cân bằng; test chênh lệch nhẹ.
- Có 25 ảnh có alpha channel, cần chuẩn hóa về RGB trước khi train.
- Notebook đặt kích thước baseline là `224 x 224`.
- Chưa có kết luận từ EDA về background, duplicate hoặc data leakage.

## 5. Quy ước tiền xử lý dự kiến

- Đọc ảnh theo đúng format mà thư viện model yêu cầu.
- Resize và normalize phải được ghi rõ trong `reports/EXPERIMENT_LOG.md` khi chốt.
- Augmentation chỉ áp dụng cho training set.
- Validation set, nếu được tạo, phải tách ra trước khi augmentation.

## 6. Tiêu chí đánh giá

Các metric cần xem xét:

- Accuracy.
- Precision, recall và F1-score theo từng lớp.
- Confusion matrix.
- Kết quả trên ảnh trong `data/predict/` hoặc `data/real/`, nếu phù hợp.

Không kết luận model tốt chỉ dựa trên accuracy nếu dữ liệu bị mất cân bằng.

## 7. Pipeline dự kiến

1. Kiểm tra và phân tích dữ liệu.
2. Chuẩn bị train/validation/test.
3. Tiền xử lý và augmentation.
4. Huấn luyện model.
5. Đánh giá trên validation và test.
6. Lưu model và metadata cần thiết.
7. Chạy thử dự đoán trên ảnh mới.

## 8. Trạng thái hiện tại

- Đã có: notebook EDA và cấu trúc project cơ bản.
- Đang làm: xác định pipeline huấn luyện.
- Chưa chốt: model, image size, transform, learning rate, batch size và số
  epoch.

## 9. Quy tắc cập nhật

- Thông tin mô tả bài toán và dữ liệu được cập nhật trong file này.
- Lựa chọn, kết quả và lý do của từng lần train được ghi trong
  `reports/EXPERIMENT_LOG.md`.
- Không ghi kết quả thử nghiệm tạm thời vào đây.
