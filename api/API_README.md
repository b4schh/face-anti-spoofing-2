# Face Anti-Spoofing REST API

REST API server sử dụng FastAPI để phát hiện giả mạo khuôn mặt (Face Anti-Spoofing Detection).

## Yêu cầu

### Dependencies
```bash
pip install -r requirements_api.txt
```

Các thư viện cần thiết:
- `fastapi`
- `uvicorn[standard]`
- `torch` (PyTorch)
- `torchvision`
- `opencv-python` (cv2)
- `mediapipe`
- `numpy`
- `python-multipart` (cho file upload)
- `pydantic`

## Cách chạy

### 1. Từ thư mục api
```bash
cd api
python api_server.py
```

### 2. Từ thư mục gốc
```bash
python api/api_server.py
```

### 3. Sử dụng uvicorn trực tiếp
```bash
uvicorn api.api_server:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Trên Windows (PowerShell)
```powershell
& python api/api_server.py
```

Server sẽ chạy tại:
- API Base: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health

## Cấu hình Model Path

### Mặc định
API tự động tìm model theo thứ tự ưu tiên:
1. `../test_model/best_model.pth` (folder test_model ở thư mục gốc)
2. `./best_model.pth` (trong folder api)
3. `./model/best_model.pth` (trong folder api/model)

### Sau khi train xong model mới

**Cách 1: Đặt model vào folder test_model (Khuyến nghị)**
```bash
# Copy model vào folder test_model
cp your_trained_model.pth ../test_model/best_model.pth

# Restart server
```

**Cách 2: Sửa code để thêm path mới**

Mở file `api_server.py` và chỉnh sửa phần đầu file (dòng 22-26):

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

# Restart server
```

## API Endpoints

### GET /
Thông tin về API

**Response:**
```json
{
  "name": "Face Anti-Spoofing API",
  "version": "1.0.0",
  "status": "running",
  "device": "cuda:0",
  "endpoints": {
    "POST /predict/upload": "Upload image file",
    "POST /predict/base64": "Send base64 image",
    "GET /health": "Health check"
  }
}
```

### GET /health
Kiểm tra trạng thái server

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0"
}
```

### POST /predict/upload
Upload file ảnh để phát hiện

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response Success (có face):**
```json
{
  "success": true,
  "message": "Face detected",
  "data": {
    "prediction": "LIVE",
    "is_live": true,
    "confidence": 0.9234,
    "bbox": {
      "x1": 120,
      "y1": 80,
      "x2": 320,
      "y2": 280
    }
  }
}
```

**Response Success (không có face):**
```json
{
  "success": false,
  "message": "No face detected",
  "data": null
}
```

### POST /predict/base64
Gửi ảnh dạng base64

**Request:**
- Method: `POST`
- Content-Type: `application/json`
- Body:
```json
{
  "image": "base64_encoded_string_here"
}
```

**Response:** Giống với `/predict/upload`

## Cách sử dụng

### 1. Sử dụng curl

**Upload file:**
```bash
curl -X POST "http://localhost:8000/predict/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

**Base64:**
```bash
# Encode ảnh thành base64
BASE64_IMG=$(base64 -w 0 image.jpg)

# Gửi request
curl -X POST "http://localhost:8000/predict/base64" \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$BASE64_IMG\"}"
```

### 2. Sử dụng Python requests

**Upload file:**
```python
import requests

url = "http://localhost:8000/predict/upload"
files = {"file": open("image.jpg", "rb")}

response = requests.post(url, files=files)
result = response.json()

print(f"Success: {result['success']}")
if result['success']:
    data = result['data']
    print(f"Prediction: {data['prediction']}")
    print(f"Confidence: {data['confidence']:.2%}")
    print(f"BBox: {data['bbox']}")
```

**Base64:**
```python
import requests
import base64

url = "http://localhost:8000/predict/base64"

# Encode ảnh
with open("image.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

# Gửi request
payload = {"image": img_base64}
response = requests.post(url, json=payload)
result = response.json()

print(f"Prediction: {result['data']['prediction']}")
print(f"Confidence: {result['data']['confidence']:.2%}")
```

### 3. Sử dụng JavaScript/Fetch

```javascript
// Upload file
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/predict/upload', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Success:', data.success);
    if (data.success) {
        console.log('Prediction:', data.data.prediction);
        console.log('Confidence:', data.data.confidence);
    }
});
```

### 4. Test với Swagger UI

Truy cập http://localhost:8000/docs để sử dụng interactive API documentation (Swagger UI):
1. Mở endpoint muốn test
2. Click "Try it out"
3. Upload file hoặc nhập dữ liệu
4. Click "Execute"
5. Xem kết quả

## Cấu trúc Response

### Success Response
```json
{
  "success": true,
  "message": "Face detected",
  "data": {
    "prediction": "LIVE" | "SPOOF",
    "is_live": true | false,
    "confidence": 0.0-1.0,
    "bbox": {
      "x1": int,
      "y1": int,
      "x2": int,
      "y2": int
    }
  }
}
```

### No Face Response
```json
{
  "success": false,
  "message": "No face detected",
  "data": null
}
```

### Error Response
```json
{
  "detail": "Error message here"
}
```

## Cấu hình nâng cao

### Thay đổi host và port
```python
# Trong api_server.py, dòng cuối
uvicorn.run(
    app,
    host="0.0.0.0",      # Thay đổi host (0.0.0.0 cho phép truy cập từ bên ngoài)
    port=8000,           # Thay đổi port
    log_level="info"     # Thay đổi log level (debug, info, warning, error)
)
```

### Thay đổi ngưỡng phát hiện face
```python
# Dòng 263
with mp_face.FaceDetection(
    model_selection=0, 
    min_detection_confidence=0.5  # Tăng/giảm ngưỡng (0.0-1.0)
) as face_detector:
```

### Thay đổi ngưỡng phân loại LIVE/SPOOF
```python
# Dòng 289
is_live = prob >= 0.5  # Thay đổi ngưỡng (mặc định 0.5)
```

### Thay đổi kích thước ảnh input
```python
# Dòng 208-209
cfg = SimpleNamespace()
cfg.img_size = 256  # Thay đổi thành 224, 299, v.v.
```

### CORS Configuration
Mặc định API cho phép tất cả origins. Để giới hạn, sửa dòng 312-313:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],  # Chỉ định origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Giới hạn methods
    allow_headers=["*"],
)
```

## Xử lý lỗi thường gặp

### Lỗi: "Model checkpoint not found!"
**Nguyên nhân**: Không tìm thấy file model

**Giải pháp**:
```bash
# Kiểm tra file có tồn tại
ls test_model/best_model.pth

# Nếu không có, copy từ nơi khác
cp /path/to/your/model.pth test_model/best_model.pth

# Restart server
```

### Lỗi: "Address already in use"
**Nguyên nhân**: Port 8000 đang được sử dụng

**Giải pháp 1 - Đổi port:**
```python
# Trong api_server.py
uvicorn.run(app, host="0.0.0.0", port=8001)  # Đổi sang port khác
```

**Giải pháp 2 - Kill process:**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### Lỗi: "Cannot decode image"
**Nguyên nhân**: File không phải ảnh hợp lệ hoặc bị corrupt

**Giải pháp**: Kiểm tra file ảnh, đảm bảo đúng định dạng (jpg, png, etc.)

### Lỗi: CUDA out of memory
**Nguyên nhân**: GPU không đủ bộ nhớ

**Giải pháp**: Server tự động fallback sang CPU nếu không có CUDA, hoặc giảm kích thước ảnh input

## Performance Tips

### 1. Sử dụng GPU
- Model tự động sử dụng CUDA nếu có
- Đảm bảo PyTorch có CUDA support:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. Batch Processing
Nếu cần xử lý nhiều ảnh, gọi API song song bằng async requests:
```python
import asyncio
import aiohttp

async def predict_async(session, image_path):
    async with session.post(
        'http://localhost:8000/predict/upload',
        data={'file': open(image_path, 'rb')}
    ) as response:
        return await response.json()

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [predict_async(session, f'image_{i}.jpg') for i in range(10)]
        results = await asyncio.gather(*tasks)
        return results
```

### 3. Tăng số workers
```bash
uvicorn api.api_server:app --workers 4 --host 0.0.0.0 --port 8000
```

## Production Deployment

### Sử dụng Gunicorn + Uvicorn
```bash
pip install gunicorn

gunicorn api.api_server:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120
```

### Docker
Tạo file `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

COPY . .

EXPOSE 8000

CMD ["python", "api/api_server.py"]
```

Build và run:
```bash
docker build -t face-anti-spoofing-api .
docker run -p 8000:8000 face-anti-spoofing-api
```

### Nginx Reverse Proxy
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

## Security

### 1. Rate Limiting
Cài đặt slowapi:
```bash
pip install slowapi
```

Thêm vào code:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict/upload")
@limiter.limit("10/minute")  # 10 requests per minute
async def predict_upload(request: Request, file: UploadFile = File(...)):
    # ...
```

### 2. API Key Authentication
```python
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

API_KEY = "your-secret-api-key"
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.post("/predict/upload")
async def predict_upload(
    file: UploadFile = File(...),
    api_key: str = Security(verify_api_key)
):
    # ...
```

### 3. File Size Limit
```python
from fastapi import Request

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.method == "POST":
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_FILE_SIZE:
            return JSONResponse(
                status_code=413,
                content={"detail": "File too large"}
            )
    return await call_next(request)
```

## Model Information

- **Architecture**: SSAN-R (Spatial-Style Adaptive Network)
- **Backbone**: ResNet-18
- **Input size**: 256x256
- **Output**: Binary classification (Live/Spoof)
- **Device**: Auto-detect CUDA/CPU

## License

MIT License

## Author

- **File**: `api_server.py`
- **Description**: Face Anti-Spoofing REST API
- **Version**: 1.0.0
