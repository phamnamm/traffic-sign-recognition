# 🚦 Traffic Sign Recognition — Nhận diện biển báo giao thông bằng CNN

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%20%2F%20Keras-FF6F00?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

Mô hình **Convolutional Neural Network (CNN)** phân loại **43 loại biển báo giao thông** (bộ biển báo theo chuẩn GTSRB — German Traffic Sign Recognition Benchmark), kèm ứng dụng **nhận diện thời gian thực qua webcam** bằng OpenCV.

| Chỉ số | Kết quả |
|---|---|
| **Test accuracy** | **98.61 %** |
| Validation accuracy (epoch 10) | 98.44 % |
| Test loss | 0.0536 |
| Tổng số tham số | 378,023 (~1.44 MB) |

---

## 📑 Mục lục
- [Tính năng](#-tính-năng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Dữ liệu](#-dữ-liệu)
- [Pipeline xử lý](#-pipeline-xử-lý)
- [Kiến trúc mô hình](#-kiến-trúc-mô-hình)
- [Kết quả huấn luyện](#-kết-quả-huấn-luyện)
- [Cài đặt & Sử dụng](#️-cài-đặt--sử-dụng)
- [Hạn chế & Hướng phát triển](#-hạn-chế--hướng-phát-triển)

---

## ✨ Tính năng
- Đọc và gán nhãn tự động ảnh từ 43 thư mục lớp.
- Tiền xử lý ảnh: chuyển **grayscale**, **cân bằng histogram** (chuẩn hoá ánh sáng), **chuẩn hoá về [0, 1]**.
- **Data augmentation** (dịch, zoom, xoay, shear) để tăng khả năng tổng quát hoá.
- Huấn luyện CNN 4 lớp tích chập với Dropout chống overfitting.
- Trực quan hoá phân bố dữ liệu, mẫu ảnh từng lớp, đường cong loss/accuracy.
- **Nhận diện real-time từ webcam**, hiển thị tên biển báo và xác suất dự đoán.

## 📁 Cấu trúc dự án

```
traffic-sign-recognition/
├── TrafficSign_Main.ipynb   # Nạp dữ liệu, tiền xử lý, huấn luyện & đánh giá mô hình
├── TrafficSign_Test.ipynb   # Nhận diện biển báo real-time qua webcam
├── myData/                  # (tự chuẩn bị) 43 thư mục ảnh: 0/, 1/, ..., 42/
├── labels.csv               # (tự chuẩn bị) ánh xạ ClassId -> Name
└── model_trained.p          # (sinh ra sau khi huấn luyện) mô hình đã train
```

> ⚠️ Thư mục `myData/`, file `labels.csv` và `model_trained.p` **không được đưa lên repo** do dung lượng lớn.

## 🗂️ Dữ liệu

- **Nguồn:** GTSRB — 43 lớp biển báo (giới hạn tốc độ, cấm vượt, nhường đường, dừng lại, cảnh báo nguy hiểm, biển chỉ hướng...).
- **Kích thước ảnh:** 32 × 32 × 3.
- **Tổng số ảnh:** 34,799, chia như sau:

| Tập | Số ảnh | Tỷ lệ |
|---|---|---|
| Train | 22,271 | 64 % |
| Validation | 5,568 | 16 % |
| Test | 6,960 | 20 % |

**Định dạng yêu cầu:**
- `myData/<class_id>/*.png|jpg` với `class_id` từ `0` đến `42`.
- `labels.csv` gồm 2 cột `ClassId,Name`.

> Dữ liệu **mất cân bằng** giữa các lớp (ít nhất ~106 ảnh, nhiều nhất ~1,269 ảnh/lớp trong tập train) — notebook có biểu đồ phân bố để minh hoạ.

## 🔄 Pipeline xử lý

```
Ảnh gốc (32×32 BGR)
   └─► Grayscale ─► Histogram Equalization ─► Normalize (/255)
         └─► Reshape (32×32×1)
               └─► Data Augmentation (ImageDataGenerator)
                     • width/height shift 10%   • zoom 20%
                     • shear 0.1                 • rotation ±10°
                     └─► CNN ─► Softmax (43 lớp)
```

## 🧠 Kiến trúc mô hình

| Layer | Output Shape | Params |
|---|---|---|
| Conv2D (60 filters, 5×5, ReLU) | (28, 28, 60) | 1,560 |
| Conv2D (60 filters, 5×5, ReLU) | (24, 24, 60) | 90,060 |
| MaxPooling2D (2×2) | (12, 12, 60) | 0 |
| Conv2D (30 filters, 3×3, ReLU) | (10, 10, 30) | 16,230 |
| Conv2D (30 filters, 3×3, ReLU) | (8, 8, 30) | 8,130 |
| MaxPooling2D (2×2) | (4, 4, 30) | 0 |
| Dropout (0.5) | (4, 4, 30) | 0 |
| Flatten | (480) | 0 |
| Dense (500, ReLU) | (500) | 240,500 |
| Dropout (0.5) | (500) | 0 |
| Dense (43, Softmax) | (43) | 21,543 |

**Siêu tham số:** Optimizer `Adam (lr = 0.001)` · Loss `categorical_crossentropy` · Batch size `50` · Epochs `10` · Steps/epoch `2000`.

## 📈 Kết quả huấn luyện

| Epoch | Train acc | Val acc | Val loss |
|---|---|---|---|
| 1 | 0.2699 | 0.7983 | 0.7154 |
| 3 | 0.7378 | 0.9402 | 0.1964 |
| 5 | 0.8347 | 0.9752 | 0.1024 |
| 8 | 0.8801 | 0.9795 | 0.0770 |
| 10 | 0.8983 | **0.9844** | **0.0568** |

**Đánh giá trên tập test:** accuracy **98.61 %**, loss **0.0536**.

> Train accuracy thấp hơn validation accuracy là bình thường: dữ liệu train được augmentation và áp dụng Dropout, trong khi validation/test dùng ảnh gốc.

## ⚙️ Cài đặt & Sử dụng

### 1. Cài đặt môi trường

```bash
git clone https://github.com/phamnamm/traffic-sign-recognition.git
cd traffic-sign-recognition

python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate

pip install tensorflow opencv-python scikit-learn pandas numpy matplotlib jupyter
```

### 2. Chuẩn bị dữ liệu
Đặt thư mục `myData/` (43 thư mục con) và file `labels.csv` vào thư mục gốc dự án.

### 3. Huấn luyện mô hình
Mở và chạy toàn bộ `TrafficSign_Main.ipynb`. Sau khi chạy xong, mô hình được lưu thành `model_trained.p`.

### 4. Nhận diện real-time qua webcam
Mở và chạy `TrafficSign_Test.ipynb`. Đưa biển báo vào trước camera — kết quả chỉ hiển thị khi xác suất > **75 %** (`threshold = 0.75`).

## 🚧 Hạn chế & Hướng phát triển
- Mô hình được lưu bằng `pickle`; nên chuyển sang định dạng chuẩn của Keras (`model.save("model.keras")`) để tương thích tốt hơn giữa các phiên bản.
- Dữ liệu mất cân bằng giữa các lớp — có thể áp dụng class weighting hoặc oversampling.
- Mô hình phân loại **toàn khung hình**, chưa có bước phát hiện vị trí biển báo (object detection như YOLO/SSD).
- Bổ sung `requirements.txt`, confusion matrix và báo cáo precision/recall theo từng lớp.

---

## 👤 Tác giả
**Phạm Nam** — [@phamnamm](https://github.com/phamnamm)
