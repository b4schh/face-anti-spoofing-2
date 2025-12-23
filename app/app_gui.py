"""
Face Anti-Spoofing App
- Tab 1: Webcam real-time
- Tab 2: Upload ảnh
"""

import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import cv2
import time
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os
from queue import Queue
import traceback

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

checkpoint_paths = [
    os.path.join(parent_dir, "test_model", "best_model.pth"),
    os.path.join(script_dir, "best_model.pth"),
    os.path.join(script_dir, "model", "best_model.pth"),
]

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


cfg = SimpleNamespace()
cfg.img_size = 256
cfg.num_domains = 6  # Must match the trained model checkpoint


def preprocess_face(face_img, device):
    """Preprocess face image BGR → Tensor"""
    face_resized = cv2.resize(face_img, (cfg.img_size, cfg.img_size))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_chw = face_rgb.transpose((2, 0, 1))
    face_normalized = (face_chw - 127.5) / 128.0
    face_tensor = torch.from_numpy(face_normalized.astype(np.float32)).to(device)
    return face_tensor.unsqueeze(0)


class FaceAntiSpoofingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎭 Face Anti-Spoofing Detection")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        self.model = None
        self.device = None
        self.mp_face = None
        self.cap = None
        self.is_running = False
        self.face_scores = {}
        self.frame_queue = Queue(maxsize=2)  
        
        self.load_model()
        
        self.create_widgets()
        
    def load_model(self):
        """Load model và MediaPipe"""
        try:
            # Get the directory where this script is located

            checkpoint_path = None
            for path in checkpoint_paths:
                if os.path.exists(path):
                    checkpoint_path = path
                    break
            
            if checkpoint_path is None:
                print("❌ Không tìm thấy model checkpoint!")
                messagebox.showerror("Error", "Không tìm thấy model checkpoint!\nVui lòng đặt file 'best_model.pth' trong thư mục này")
                self.root.quit()
                return
            
            # Load model
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = SSAN_R(max_iter=10000, num_domains=cfg.num_domains).to(self.device)
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            
            # Load MediaPipe
            import mediapipe as mp
            self.mp_face = mp.solutions.face_detection
            
            print(f"✅ Model loaded! Device: {self.device}")
            
        except Exception as e:
            print(f"❌ Lỗi load model: {str(e)}")
            messagebox.showerror("Error", f"Lỗi load model:\n{str(e)}")
            self.root.quit()
    
    def create_widgets(self):
        """Tạo giao diện GUI"""
        # Title
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        title_frame.pack(fill=tk.X)
        
        title = tk.Label(
            title_frame,
            text="🎭 FACE ANTI-SPOOFING DETECTION",
            font=("Arial", 24, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title.pack(pady=20)
        
        # Device info
        device_label = tk.Label(
            title_frame,
            text=f"Device: {self.device}",
            font=("Arial", 10),
            bg="#2c3e50",
            fg="#ecf0f1"
        )
        device_label.pack()
        
        # Notebook (Tabs)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Webcam
        self.webcam_frame = tk.Frame(notebook, bg="white")
        notebook.add(self.webcam_frame, text="📷 Webcam")
        self.create_webcam_tab()
        
        # Tab 2: Upload Image
        self.upload_frame = tk.Frame(notebook, bg="white")
        notebook.add(self.upload_frame, text="📁 Upload Image")
        self.create_upload_tab()
        
    def create_webcam_tab(self):
        """Tab 1: Webcam real-time"""
        # Control frame
        control_frame = tk.Frame(self.webcam_frame, bg="white")
        control_frame.pack(pady=10)
        
        self.btn_start = tk.Button(
            control_frame,
            text="▶ Start Webcam",
            font=("Arial", 12, "bold"),
            bg="#27ae60",
            fg="white",
            padx=20,
            pady=10,
            command=self.start_webcam
        )
        self.btn_start.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = tk.Button(
            control_frame,
            text="⏹ Stop Webcam",
            font=("Arial", 12, "bold"),
            bg="#e74c3c",
            fg="white",
            padx=20,
            pady=10,
            command=self.stop_webcam,
            state=tk.DISABLED
        )
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        
        btn_screenshot = tk.Button(
            control_frame,
            text="📸 Screenshot",
            font=("Arial", 12),
            bg="#3498db",
            fg="white",
            padx=20,
            pady=10,
            command=self.save_screenshot
        )
        btn_screenshot.pack(side=tk.LEFT, padx=5)
        
        # Video canvas
        self.video_canvas = tk.Label(self.webcam_frame, bg="black")
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Status
        self.status_label = tk.Label(
            self.webcam_frame,
            text="Press 'Start Webcam' to begin",
            font=("Arial", 11),
            bg="white",
            fg="#7f8c8d"
        )
        self.status_label.pack(pady=10)
        
    def create_upload_tab(self):
        """Tab 2: Upload Image"""
        # Upload button
        btn_upload = tk.Button(
            self.upload_frame,
            text="📁 Choose Image",
            font=("Arial", 14, "bold"),
            bg="#3498db",
            fg="white",
            padx=30,
            pady=15,
            command=self.upload_image
        )
        btn_upload.pack(pady=30)
        
        # Result frame
        result_frame = tk.Frame(self.upload_frame, bg="white")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left: Original image
        left_frame = tk.Frame(result_frame, bg="white")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        tk.Label(
            left_frame,
            text="📸 Original Image",
            font=("Arial", 12, "bold"),
            bg="white"
        ).pack(pady=10)
        
        self.upload_canvas = tk.Label(left_frame, bg="#ecf0f1")
        self.upload_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Right: Result
        right_frame = tk.Frame(result_frame, bg="white")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        tk.Label(
            right_frame,
            text="🔍 Detection Result",
            font=("Arial", 12, "bold"),
            bg="white"
        ).pack(pady=10)
        
        self.result_canvas = tk.Label(right_frame, bg="#ecf0f1")
        self.result_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Result text
        self.result_text = tk.Label(
            right_frame,
            text="",
            font=("Arial", 16, "bold"),
            bg="white",
            fg="black"
        )
        self.result_text.pack(pady=20)
        
        self.confidence_text = tk.Label(
            right_frame,
            text="",
            font=("Arial", 12),
            bg="white"
        )
        self.confidence_text.pack()
        
    def start_webcam(self):
        """Bắt đầu webcam"""
        if self.is_running:
            return
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Không mở được webcam!")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.status_label.config(text="🎥 Webcam is running...", fg="#27ae60")
        
        # Start webcam thread
        self.webcam_thread = threading.Thread(target=self.webcam_loop, daemon=True)
        self.webcam_thread.start()
        
        # Start UI update loop
        self.update_video_display()
        
    def stop_webcam(self):
        """Dừng webcam"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.status_label.config(text="⏹ Webcam stopped", fg="#e74c3c")
        self.video_canvas.config(image="")
        
    def webcam_loop(self):
        """Vòng lặp webcam"""
        frame_count = 0
        fps_start = time.time()
        fps = 0
        
        with self.mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detector:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                h, w, _ = frame.shape
                frame_count += 1
                do_infer = (frame_count % 2 == 0)
                
                # FPS
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_start)
                    fps_start = time.time()
                
                # Face detection
                small_frame = cv2.resize(frame, (640, 360))
                results = face_detector.process(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
                
                h_scale = h / 360
                w_scale = w / 640
                
                current_faces = {}
                
                if results.detections:
                    for face_id, det in enumerate(results.detections):
                        bbox = det.location_data.relative_bounding_box
                        
                        x1 = int(bbox.xmin * 640 * w_scale)
                        y1 = int(bbox.ymin * 360 * h_scale)
                        x2 = int((bbox.xmin + bbox.width) * 640 * w_scale)
                        y2 = int((bbox.ymin + bbox.height) * 360 * h_scale)
                        
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        if x2 - x1 < 40 or y2 - y1 < 40:
                            continue
                        
                        face_img = frame[y1:y2, x1:x2]
                        
                        # Inference
                        if do_infer:
                            try:
                                inp = preprocess_face(face_img, self.device)
                                with torch.no_grad():
                                    logits, _, _, _ = self.model(inp, inp)
                                    prob = torch.softmax(logits, dim=1)[0, 1].item()
                                current_faces[face_id] = prob
                            except:
                                current_faces[face_id] = 0.0
                        else:
                            current_faces[face_id] = self.face_scores.get(face_id, 0.0)
                        
                        # Draw
                        prob = current_faces[face_id]
                        label = "LIVE" if prob >= 0.5 else "SPOOF"
                        color = (0, 255, 0) if prob >= 0.5 else (0, 0, 255)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        
                        text = f"{label} ({prob:.2f})"
                        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
                        cv2.putText(frame, text, (x1 + 5, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if do_infer:
                    self.face_scores = current_faces
                
                # FPS overlay
                info = f"FPS: {fps:.1f} | Faces: {len(current_faces)}"
                cv2.putText(frame, info, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Đưa frame vào queue (không block nếu full)
                self.current_frame = frame
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                
                # Thêm delay nhỏ để giảm CPU usage
                time.sleep(0.01)
                
        self.stop_webcam()
    
    def update_video_display(self):
        """Update video display từ queue (chạy trong main thread)"""
        if self.is_running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()
                    
                    # Convert và resize
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    
                    # Resize to fit canvas
                    canvas_width = self.video_canvas.winfo_width()
                    canvas_height = self.video_canvas.winfo_height()
                    
                    if canvas_width > 1 and canvas_height > 1:
                        img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
                    
                    # Update canvas
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_canvas.imgtk = imgtk  # Giữ reference
                    self.video_canvas.configure(image=imgtk)
                    
            except Exception as e:
                pass
            
            # Schedule next update (30 FPS)
            self.root.after(33, self.update_video_display)
        
    def save_screenshot(self):
        """Lưu screenshot"""
        if hasattr(self, 'current_frame'):
            filename = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, self.current_frame)
            messagebox.showinfo("Success", f"Saved: {filename}")
        
    def upload_image(self):
        """Upload và phân tích ảnh"""
        filepath = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff *.tif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("WebP files", "*.webp"),
                ("All files", "*.*")
            ]
        )
        
        if not filepath:
            print("❌ No file selected")
            return
        
        print(f"\n{'='*60}")
        print(f"📁 Processing: {filepath}")
        print(f"{'='*60}")
        
        try:
            # Load image
            print("Step 1: Loading image...")
            print(f"   File path: {filepath}")
            print(f"   File exists: {os.path.exists(filepath)}")
            
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"   File size: {file_size} bytes")
            
            # Thử đọc bằng cv2.imread trước
            img_bgr = cv2.imread(filepath)
            
            # Nếu không được, thử đọc bằng numpy để xử lý Unicode path
            if img_bgr is None:
                print("   cv2.imread() failed, trying numpy alternative for Unicode path...")
                try:
                    # Đọc file dưới dạng binary
                    with open(filepath, 'rb') as f:
                        file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                    # Decode từ buffer
                    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    print("   ✅ Successfully loaded with cv2.imdecode()")
                except Exception as decode_error:
                    print(f"   ❌ cv2.imdecode() also failed: {decode_error}")
            
            if img_bgr is None:
                error_msg = f"Cannot read image: {filepath}\n\nDetails:\n"
                error_msg += f"- File exists: {os.path.exists(filepath)}\n"
                if os.path.exists(filepath):
                    error_msg += f"- File size: {os.path.getsize(filepath)} bytes\n"
                error_msg += f"- Path contains Unicode: {'Học' in filepath or 'máy' in filepath}\n"
                error_msg += "\nPossible causes:\n"
                error_msg += "1. Image file is corrupted\n"
                error_msg += "2. Unsupported image format\n"
                error_msg += "3. Unicode path issue (try moving file to simple path like C:\\temp\\)"
                print(f"❌ {error_msg}")
                messagebox.showerror("Error", error_msg)
                return
            
            print(f"✅ Image loaded: {img_bgr.shape}")
            
            # Display original
            print("Step 2: Displaying original image...")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil.thumbnail((256, 256), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img_pil)
            self.upload_canvas.imgtk = imgtk
            self.upload_canvas.configure(image=imgtk)
            print("✅ Original image displayed")
            
            # Detect face
            print("Step 3: Detecting face with MediaPipe...")
            with self.mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detector:
                results = face_detector.process(img_rgb)
                
                if results.detections:
                    print(f"✅ Found {len(results.detections)} face(s)")
                    
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
                    
                    print(f"   Face bbox: ({x1}, {y1}) -> ({x2}, {y2})")
                    print(f"   Face size: {x2-x1}x{y2-y1}")
                    
                    face_img = img_bgr[y1:y2, x1:x2]
                    
                    if face_img.size == 0:
                        error_msg = "Face region is empty (invalid bbox)"
                        print(f"❌ {error_msg}")
                        messagebox.showerror("Error", error_msg)
                        return
                    
                    # Inference
                    print("Step 4: Running inference...")
                    try:
                        inp = preprocess_face(face_img, self.device)
                        print(f"   Input tensor shape: {inp.shape}")
                        
                        with torch.no_grad():
                            logits, _, _, _ = self.model(inp, inp)
                            prob = torch.softmax(logits, dim=1)[0, 1].item()
                        
                        print(f"✅ Inference complete: prob={prob:.4f}")
                    except Exception as e:
                        error_msg = f"Model inference failed:\n{str(e)}"
                        print(f"❌ {error_msg}")
                        print(f"Traceback:\n{traceback.format_exc()}")
                        messagebox.showerror("Error", error_msg)
                        return
                    
                    is_live = prob >= 0.5
                    label = "✅ LIVE" if is_live else "❌ SPOOF"
                    color = (0, 255, 0) if is_live else (255, 0, 0)
                    
                    print(f"Step 5: Final result: {label} (confidence: {prob:.2%})")
                    
                    # Draw result
                    result_img = img_bgr.copy()
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 3)
                    text = f"{label} ({prob:.2f})"
                    cv2.putText(result_img, text, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Display result
                    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    result_pil = Image.fromarray(result_rgb)
                    result_pil.thumbnail((256, 256), Image.Resampling.LANCZOS)
                    result_tk = ImageTk.PhotoImage(image=result_pil)
                    self.result_canvas.result_tk = result_tk
                    self.result_canvas.configure(image=result_tk)
                    
                    # Update text
                    self.result_text.config(
                        text=label,
                        fg="#27ae60" if is_live else "#e74c3c"
                    )
                    self.confidence_text.config(
                        text=f"Confidence: {prob:.2%}"
                    )
                    
                    print(f"{'='*60}")
                    print("✅ Processing complete!")
                    print(f"{'='*60}\n")
                    
                else:
                    error_msg = "No face detected in image!"
                    print(f"❌ {error_msg}")
                    messagebox.showwarning("Warning", error_msg)
                    
        except Exception as e:
            error_msg = f"Error processing image:\n{str(e)}"
            print(f"❌ {error_msg}")
            print(f"Traceback:\n{traceback.format_exc()}")
            messagebox.showerror("Error", error_msg)
            

def main():
    root = tk.Tk()
    app = FaceAntiSpoofingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
