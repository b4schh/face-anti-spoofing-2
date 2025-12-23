# Face Anti-Spoofing Web Application

Web application đơn giản để phát hiện giả mạo khuôn mặt real-time sử dụng webcam và REST API.

## Tổng quan

Web app này sử dụng:
- **Frontend**: HTML, CSS, JavaScript (Vanilla JS)
- **Face Detection**: MediaPipe Face Detection (CDN)
- **Backend API**: Face Anti-Spoofing REST API server
- **Architecture**: Client-side face detection + Server-side inference

## Yêu cầu

### 1. API Server phải đang chạy
```bash
# Chạy API server trước
cd api
python api_server.py
```

API server mặc định chạy tại: `http://localhost:8000`

### 2. Web Browser hỗ trợ
- Chrome/Edge/Brave (khuyến nghị)
- Firefox
- Safari (có thể có giới hạn)

### 3. Webcam
- Webcam hoặc camera laptop
- Quyền truy cập camera từ browser

## Cách chạy

### Cách 1: Sử dụng Live Server (VSCode)

1. Cài đặt extension "Live Server" trong VSCode
2. Mở file `index.html`
3. Click chuột phải và chọn "Open with Live Server"
4. Browser sẽ tự động mở tại `http://127.0.0.1:5500/webapp/`

### Cách 2: Sử dụng Python HTTP Server

```bash
# Từ thư mục gốc
cd face-anti-spoofing-2

# Python 3
python -m http.server 8080

# Hoặc Python 2
python -m SimpleHTTPServer 8080
```

Truy cập: `http://localhost:8080/webapp/`

### Cách 3: Sử dụng Node.js http-server

```bash
# Cài đặt http-server (lần đầu)
npm install -g http-server

# Chạy server
cd face-anti-spoofing-2
http-server -p 8080

# Hoặc với CORS enabled
http-server -p 8080 --cors
```

Truy cập: `http://localhost:8080/webapp/`

### Cách 4: Mở trực tiếp file HTML

**Lưu ý**: Cách này có thể gặp lỗi CORS khi gọi API

1. Mở file `index.html` trực tiếp trong browser
2. Hoặc kéo thả file vào browser

## Cách sử dụng

### Bước 1: Khởi động API Server
```bash
cd api
python api_server.py
```

Đợi đến khi thấy:
```
Model loaded! Device: cuda:0
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Bước 2: Mở Web App
Sử dụng một trong các cách trên để mở web app

### Bước 3: Kiểm tra API Status
- Góc trên cùng sẽ hiển thị trạng thái API:
  - **Dot màu xanh** + "API Online": Kết nối thành công
  - **Dot màu đỏ** + "API Offline": Không kết nối được

Nếu offline, kiểm tra:
- API server có đang chạy không?
- URL API đúng chưa? (mặc định `http://localhost:8000`)

### Bước 4: Bắt đầu phát hiện
1. Click nút **"Start Camera"**
2. Cho phép quyền truy cập camera khi browser yêu cầu
3. Đưa mặt vào trước camera
4. App sẽ:
   - Detect khuôn mặt (box màu xanh hoặc đỏ)
   - Gửi ảnh đến API để phân tích
   - Hiển thị kết quả: **LIVE** (thật) hoặc **SPOOF** (giả)

### Bước 5: Xem kết quả
- **Detection Results**: Kết quả hiện tại với confidence score
- **Detected Face**: Preview của khuôn mặt được detect
- **Detection History**: Lịch sử các lần phát hiện

### Bước 6: Dừng
Click nút **"Stop Camera"** để dừng

## Cấu hình

### Thay đổi API URL
Nếu API server chạy ở địa chỉ khác:

1. Trong phần **Settings** ở web app
2. Thay đổi "API URL" (ví dụ: `http://192.168.1.10:8000`)
3. Hoặc sửa trực tiếp trong code `app.js` (dòng 38):

```javascript
// app.js
this.apiUrl = document.getElementById('apiUrl');
// Thay đổi default value trong index.html hoặc
this.apiUrl.value = 'http://your-api-server:8000';
```

### Thay đổi tần suất gửi API
Trong phần **Settings**:
- **Send Rate**: Thời gian giữa các lần gửi (ms)
  - 1000ms = gửi mỗi giây
  - 500ms = gửi mỗi 0.5 giây
  - Càng thấp = real-time hơn nhưng tốn băng thông

### Thay đổi ngưỡng hiển thị
- **Confidence Threshold**: Ngưỡng để hiển thị kết quả
  - 0.5 = hiển thị kết quả có confidence >= 50%
  - Tăng lên nếu muốn kết quả chắc chắn hơn

## Cấu trúc file

```
webapp/
├── index.html          # Giao diện chính
├── app.js              # Logic xử lý
├── style.css           # Styling
└── README.md           # File này
```

## Tính năng

### Đã triển khai
- Real-time face detection với MediaPipe
- Gửi ảnh đến API server để inference
- Hiển thị kết quả LIVE/SPOOF với confidence
- FPS counter và statistics
- Detection history
- Face preview
- API status monitoring
- Cấu hình linh hoạt (API URL, send rate)

### Có thể mở rộng
- Upload ảnh từ file
- Record video
- Multiple face detection
- Batch processing
- Export results
- Custom threshold per user

## API Integration

Web app gọi API endpoint:
```
POST http://localhost:8000/predict/upload
```

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: image file (blob từ canvas)

**Response:**
```json
{
  "success": true,
  "message": "Face detected",
  "data": {
    "prediction": "LIVE",
    "is_live": true,
    "confidence": 0.9234,
    "bbox": {...}
  }
}
```

Chi tiết xem [API README](../api/API_README.md)

## Flow hoạt động

1. **Client-side Face Detection**:
   - MediaPipe detect face từ webcam stream
   - Vẽ bounding box trên canvas
   - Extract face region

2. **API Call**:
   - Convert face region thành blob
   - Gửi POST request đến API server
   - Tuân theo send rate để tránh spam

3. **Server-side Inference**:
   - API server nhận ảnh
   - Preprocess và đưa vào model
   - Trả về kết quả LIVE/SPOOF

4. **Display Results**:
   - Update UI với kết quả
   - Thay đổi màu bounding box (xanh = LIVE, đỏ = SPOOF)
   - Lưu vào history

## Xử lý lỗi thường gặp

### Lỗi: "API Offline"
**Nguyên nhân**: API server không chạy hoặc không kết nối được

**Giải pháp**:
```bash
# Kiểm tra API server có chạy không
curl http://localhost:8000/health

# Nếu không có response, start API server
cd api
python api_server.py
```

### Lỗi: "Cannot access webcam"
**Nguyên nhân**: Browser không có quyền truy cập camera

**Giải pháp**:
1. Click vào icon khóa/camera trên thanh địa chỉ
2. Cho phép quyền Camera
3. Reload trang
4. Nếu vẫn lỗi, kiểm tra:
   - Camera có bị app khác sử dụng không?
   - Camera có hoạt động không? (test trong app khác)

### Lỗi: CORS (Cross-Origin)
**Nguyên nhân**: Mở file HTML trực tiếp (file://) và gọi API (http://)

**Giải pháp**:
- Sử dụng HTTP server (không mở trực tiếp file)
- Hoặc config API server enable CORS (đã có sẵn trong code)

### Lỗi: "No face detected"
**Nguyên nhân**: Camera không nhìn thấy khuôn mặt rõ

**Giải pháp**:
- Di chuyển gần camera hơn
- Đảm bảo ánh sáng đủ
- Nhìn thẳng vào camera
- Kiểm tra camera có bị che không

### Lỗi: Slow/Lag
**Nguyên nhân**: 
- Send rate quá thấp (gửi API quá nhanh)
- API server chạy trên CPU (chậm)
- Kết nối mạng chậm

**Giải pháp**:
- Tăng send rate lên (ví dụ 1500ms hoặc 2000ms)
- Sử dụng GPU cho API server
- Chạy API server trên cùng máy

## Performance Tips

### 1. Giảm API calls
```javascript
// Trong app.js, thay đổi default send rate
// Dòng 209 trong hàm onResults
const sendInterval = 1500; // Tăng từ 1000 lên 1500ms
```

### 2. Giảm resolution
```javascript
// Trong app.js, dòng 96
this.stream = await navigator.mediaDevices.getUserMedia({
    video: { 
        width: 640,   // Giảm từ 1280
        height: 480   // Giảm từ 720
    }
});
```

### 3. Local API Server
Chạy API server trên cùng máy để giảm latency

### 4. Batch Processing
Nếu cần xử lý nhiều frame, có thể sửa code để gửi batch requests

## Security Considerations

### Khi deploy production:

1. **HTTPS**: Sử dụng HTTPS cho cả web app và API
2. **API Authentication**: Thêm API key hoặc token
3. **Rate Limiting**: Giới hạn số request từ mỗi client
4. **Input Validation**: Kiểm tra file size và format
5. **CSP Headers**: Content Security Policy để tránh XSS

Ví dụ thêm API key:
```javascript
// app.js
async sendToAPI(imageBlob) {
    const formData = new FormData();
    formData.append('file', imageBlob, 'face.jpg');
    
    const response = await fetch(`${this.apiUrl.value}/predict/upload`, {
        method: 'POST',
        body: formData,
        headers: {
            'X-API-Key': 'your-secret-api-key'  // Thêm header
        }
    });
    // ...
}
```

## Browser Compatibility

| Browser | Support | Notes |
|---------|---------|-------|
| Chrome 90+ | Full | Khuyến nghị |
| Edge 90+ | Full | Chromium-based |
| Firefox 88+ | Full | - |
| Safari 14+ | Partial | Có thể có giới hạn với MediaPipe |
| Opera 76+ | Full | Chromium-based |
| Mobile Chrome | Full | Android only |
| Mobile Safari | Partial | iOS có giới hạn |

## Troubleshooting

### Debug Mode
Mở DevTools (F12) để xem console logs:
```javascript
// App tự động log các events:
// - Face detection results
// - API calls and responses
// - Errors
```

### Test API riêng
```bash
# Test API với curl
curl -X POST "http://localhost:8000/predict/upload" \
  -F "file=@test_image.jpg"
```

### Check MediaPipe
Nếu face detection không hoạt động, kiểm tra:
1. CDN có load được không? (xem Network tab)
2. Browser có support MediaPipe không?

## License

MIT License

## Author

- **Files**: `index.html`, `app.js`, `style.css`
- **Description**: Face Anti-Spoofing Web Application
- **Version**: 1.0.0

## Links

- [API Documentation](../api/API_README.md)
- [GUI App Documentation](../app/README.md)
- [MediaPipe Face Detection](https://google.github.io/mediapipe/solutions/face_detection)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
