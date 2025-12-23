"""
Face Anti-Spoofing API Server
FastAPI-based REST API for face liveness detection
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import cv2
import base64
from types import SimpleNamespace
from typing import Optional
import os
from pydantic import BaseModel
import uvicorn

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

checkpoint_paths = [
    os.path.join(parent_dir, "test_model", "best_model.pth"),
    os.path.join(script_dir, "best_model.pth"),
    os.path.join(script_dir, "model", "best_model.pth"),
]

# ===== MODEL ARCHITECTURE =====
class GRL(nn.Module):
    def __init__(self, max_iter):
        super(GRL, self).__init__()
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter

    def forward(self, input):
        self.iter_num += 1
        return input * 1.0


class adaIN(nn.Module):
    def __init__(self, eps=1e-5):
        super(adaIN, self).__init__()
        self.eps = eps

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        out = out_in * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        return out


class ResnetAdaINBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetAdaINBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = adaIN()
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = adaIN()

    def forward(self, x, gamma, beta):
        out = self.conv1(x)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)
        return out + x


class Discriminator(nn.Module):
    def __init__(self, max_iter, num_domains):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, num_domains)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            self.fc2
        )
        self.grl_layer = GRL(max_iter)

    def forward(self, feature):
        adversarial_out = self.ad_net(self.grl_layer(feature))
        return adversarial_out


class SSAN_R(nn.Module):
    def __init__(self, ada_num=2, max_iter=4000, num_domains=1):
        super(SSAN_R, self).__init__()
        model_resnet = models.resnet18(pretrained=False)

        self.input_layer = nn.Sequential(
            model_resnet.conv1,
            model_resnet.bn1,
            model_resnet.relu,
            model_resnet.maxpool
        )
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.adaIN_layers = nn.ModuleList([ResnetAdaINBlock(256) for i in range(ada_num)])

        self.conv_final = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )
        self.cls_head = nn.Linear(512, 2, bias=True)

        self.gamma = nn.Linear(256, 256, bias=False)
        self.beta = nn.Linear(256, 256, bias=False)

        self.FC = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.ReLU(inplace=True)
        )
        self.ada_conv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.ada_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.ada_conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256)
        )
        self.dis = Discriminator(max_iter, num_domains)

    def cal_gamma_beta(self, x1):
        x1 = self.input_layer(x1)
        x1_1 = self.layer1(x1)
        x1_2 = self.layer2(x1_1)
        x1_3 = self.layer3(x1_2)
        x1_4 = self.layer4(x1_3)
        
        x1_add = x1_1
        x1_add = self.ada_conv1(x1_add) + x1_2
        x1_add = self.ada_conv2(x1_add) + x1_3
        x1_add = self.ada_conv3(x1_add)

        gmp = torch.nn.functional.adaptive_max_pool2d(x1_add, 1)
        gmp_ = self.FC(gmp.view(gmp.shape[0], -1))
        gamma, beta = self.gamma(gmp_), self.beta(gmp_)

        domain_invariant = torch.nn.functional.adaptive_avg_pool2d(x1_4, 1).reshape(x1_4.shape[0], -1)

        return x1_4, gamma, beta, domain_invariant

    def forward(self, input1, input2):
        x1, gamma1, beta1, domain_invariant = self.cal_gamma_beta(input1)
        x2, gamma2, beta2, _ = self.cal_gamma_beta(input2)

        fea_x1_x1 = x1
        for i in range(len(self.adaIN_layers)):
            fea_x1_x1 = self.adaIN_layers[i](fea_x1_x1, gamma1, beta1)
        fea_x1_x1 = self.conv_final(fea_x1_x1)
        fea_x1_x1 = torch.nn.functional.adaptive_avg_pool2d(fea_x1_x1, 1)
        fea_x1_x1 = fea_x1_x1.reshape(fea_x1_x1.shape[0], -1)
        cls_x1_x1 = self.cls_head(fea_x1_x1)

        fea_x1_x2 = x1
        for i in range(len(self.adaIN_layers)):
            fea_x1_x2 = self.adaIN_layers[i](fea_x1_x2, gamma2, beta2)
        fea_x1_x2 = self.conv_final(fea_x1_x2)
        fea_x1_x2 = torch.nn.functional.adaptive_avg_pool2d(fea_x1_x2, 1)
        fea_x1_x2 = fea_x1_x2.reshape(fea_x1_x2.shape[0], -1)

        dis_invariant = self.dis(domain_invariant)
        return cls_x1_x1, fea_x1_x1, fea_x1_x2, dis_invariant


# ===== CONFIG =====
cfg = SimpleNamespace()
cfg.img_size = 256
cfg.num_domains = 6 # Must match the trained model checkpoint


# ===== GLOBAL MODEL =====
model = None
device = None
mp_face = None


def load_model():
    """Load model khi khởi động server"""
    global model, device, mp_face
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        raise FileNotFoundError("Model checkpoint not found!")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSAN_R(max_iter=10000, num_domains=cfg.num_domains).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Load MediaPipe
    import mediapipe as mp
    mp_face = mp.solutions.face_detection
    
    print(f"✅ Model loaded! Device: {device}")


def preprocess_face(face_img):
    """Preprocess face image BGR → Tensor"""
    face_resized = cv2.resize(face_img, (cfg.img_size, cfg.img_size))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_chw = face_rgb.transpose((2, 0, 1))
    face_normalized = (face_chw - 127.5) / 128.0
    face_tensor = torch.from_numpy(face_normalized.astype(np.float32)).to(device)
    return face_tensor.unsqueeze(0)


def detect_and_predict(img_bgr):
    """Detect face and predict liveness"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detector:
        results = face_detector.process(img_rgb)
        
        if not results.detections:
            return None
        
        # Get first face
        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box
        
        h, w, _ = img_bgr.shape
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 - x1 < 40 or y2 - y1 < 40:
            return None
        
        face_img = img_bgr[y1:y2, x1:x2]
        
        if face_img.size == 0:
            return None
        
        # Inference
        inp = preprocess_face(face_img)
        with torch.no_grad():
            logits, _, _, _ = model(inp, inp)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
        
        is_live = prob >= 0.5
        
        return {
            "is_live": is_live,
            "confidence": float(prob),
            "bbox": {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)
            }
        }


# ===== FASTAPI APP =====
app = FastAPI(
    title="Face Anti-Spoofing API",
    description="REST API for face liveness detection",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== MODELS =====
class Base64ImageRequest(BaseModel):
    image: str  # Base64 encoded image


# ===== ENDPOINTS =====
@app.on_event("startup")
async def startup_event():
    """Load model khi server khởi động"""
    load_model()


@app.get("/")
async def root():
    """API info"""
    return {
        "name": "Face Anti-Spoofing API",
        "version": "1.0.0",
        "status": "running",
        "device": str(device),
        "endpoints": {
            "POST /predict/upload": "Upload image file",
            "POST /predict/base64": "Send base64 image",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }


@app.post("/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    """
    Upload image file và detect face liveness
    
    Args:
        file: Image file (jpg, png, etc.)
    
    Returns:
        JSON với kết quả detection
    """
    try:
        # Đọc file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Cannot decode image")
        
        # Detect và predict
        result = detect_and_predict(img_bgr)
        
        if result is None:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "message": "No face detected",
                    "data": None
                }
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Face detected",
                "data": {
                    "prediction": "LIVE" if result["is_live"] else "SPOOF",
                    "is_live": result["is_live"],
                    "confidence": result["confidence"],
                    "bbox": result["bbox"]
                }
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/base64")
async def predict_base64(request: Base64ImageRequest):
    """
    Nhận base64 image và detect face liveness
    
    Args:
        request: JSON với key "image" chứa base64 string
    
    Returns:
        JSON với kết quả detection
    """
    try:
        # Decode base64
        img_data = base64.b64decode(request.image)
        nparr = np.frombuffer(img_data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Cannot decode image")
        
        # Detect và predict
        result = detect_and_predict(img_bgr)
        
        if result is None:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "message": "No face detected",
                    "data": None
                }
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Face detected",
                "data": {
                    "prediction": "LIVE" if result["is_live"] else "SPOOF",
                    "is_live": result["is_live"],
                    "confidence": result["confidence"],
                    "bbox": result["bbox"]
                }
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("🚀 Starting Face Anti-Spoofing API Server...")
    print("📖 API Docs: http://localhost:8000/docs")
    print("📊 Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
