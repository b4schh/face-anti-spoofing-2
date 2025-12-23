# Face Anti-Spoofing GUI Application

Ứng dụng GUI để phát hiện giả mạo khuôn mặt (Face Anti-Spoofing) với hai chức năng chính:
- **Webcam real-time**: Phát hiện trực tiếp qua webcam
- **Upload ảnh**: Phân tích ảnh tĩnh

## Yêu cầu

### Dependencies
```bash
pip install -r requirements_app.txt
```

Các thư viện cần thiết:
- `torch` (PyTorch)
- `torchvision`
- `opencv-python` (cv2)
- `mediapipe`
- `Pillow` (PIL)
- `numpy`

## Cách chạy

### 1. Từ thư mục `app`
```bash
cd app
python app_gui.py
```

### 2. Từ thư mục gốc
```bash
python app/app_gui.py
```

### 3. Trên Windows (PowerShell)
```powershell
& python app/app_gui.py
```

## Cấu hình Model Path

### Mặc định
Ứng dụng tự động tìm model theo thứ tự ưu tiên:
1. `../test_model/best_model.pth` (folder test_model ở thư mục gốc)
2. `./best_model.pth` (trong folder app)
3. `./model/best_model.pth` (trong folder app/model)

### Sau khi train xong model mới

**Cách 1: Đặt model vào folder `test_model` (Khuyến nghị)**
```bash
# Copy model vào folder test_model
cp your_trained_model.pth ../test_model/best_model.pth
```

**Cách 2: Sửa code để thêm path mới**

Mở file `app_gui.py` và chỉnh sửa phần đầu file (dòng 22-27):

```python
checkpoint_paths = [
    os.path.join(parent_dir, "test_model", "best_model.pth"),
    os.path.join(script_dir, "best_model.pth"),
    os.path.join(script_dir, "model", "best_model.pth"),
    # Thêm path mới của bạn ở đây:
    "D:/path/to/your/new_model.pth",  # Path tuyệt đối
    os.path.join(parent_dir, "models", "custom_model.pth"),  # Path tương đối
]
```

**Cách 3: Đổi tên và thay thế**
```bash
# Backup model cũ (nếu cần)
mv test_model/best_model.pth test_model/best_model_old.pth

# Copy model mới
cp your_trained_model.pth test_model/best_model.pth
```

## Cấu trúc thư mục

```
face-anti-spoofing-2/
├── app/
│   ├── app_gui.py          # File chính
│   ├── requirements_app.txt
│   └── README.md           # File này
├── test_model/
│   └── best_model.pth      # Model checkpoint (ưu tiên tìm ở đây)
└── model.ipynb             # Notebook training
```

## Sử dụng

### Tab 1: Webcam
1. Click **▶ Start Webcam**
2. Khuôn mặt sẽ được detect và phân loại real-time
3. Click **⏹ Stop Webcam** để dừng
4. Click **📸 Screenshot** để lưu ảnh hiện tại

### Tab 2: Upload Image
1. Click **📁 Choose Image**
2. Chọn ảnh từ máy tính
3. Kết quả hiển thị:
   - **LIVE**: Khuôn mặt thật
   - **SPOOF**: Khuôn mặt giả

## Cấu hình nâng cao

### Thay đổi kích thước ảnh input
```python
# Dòng 201-202
cfg = SimpleNamespace()
cfg.img_size = 256  # Thay đổi thành 224, 299, v.v.
```

### Thay đổi ngưỡng phát hiện
```python
# Dòng 466 (webcam) và 665 (upload)
with self.mp_face.FaceDetection(
    model_selection=0, 
    min_detection_confidence=0.7  # Tăng/giảm ngưỡng (0.0-1.0)
) as face_detector:
```

### Thay đổi ngưỡng phân loại LIVE/SPOOF
```python
# Dòng 503, 691
is_live = prob >= 0.5  # Thay đổi ngưỡng (mặc định 0.5)
```

## Xử lý lỗi thường gặp

### Lỗi: "Không tìm thấy model checkpoint!"
**Nguyên nhân**: Không tìm thấy file model

**Giải pháp**:
```bash
# Kiểm tra file có tồn tại
ls test_model/best_model.pth

# Nếu không có, copy từ nơi khác
cp /path/to/your/model.pth test_model/best_model.pth
```

### Lỗi: "AttributeError: 'NoneType' object has no attribute 'FaceDetection'"
**Nguyên nhân**: Model không load được nên MediaPipe cũng không khởi tạo

**Giải pháp**: Đảm bảo model path đúng và file tồn tại

### Lỗi: Unicode path với cv2.imread()
**Nguyên nhân**: OpenCV không đọc được đường dẫn có ký tự Unicode (tiếng Việt)

**Giải pháp**: Code đã tự động xử lý bằng `cv2.imdecode()`, hoặc di chuyển ảnh sang đường dẫn không chứa ký tự đặc biệt

## Model Information

- **Architecture**: SSAN-R (Spatial-Style Adaptive Network)
- **Backbone**: ResNet-18
- **Input size**: 256x256
- **Output**: Binary classification (Live/Spoof)
- **Device**: Tự động detect CUDA/CPU

## License

MIT License

## Author

- **File**: `app_gui.py`
- **Description**: Face Anti-Spoofing Detection GUI
- **Version**: 1.0
