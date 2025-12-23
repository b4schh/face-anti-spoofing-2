/**
 * Face Anti-Spoofing Web App
 * Client-side face detection + API inference
 */

class FaceAntiSpoofingApp {
    constructor() {
        // Elements
        this.video = document.getElementById('webcam');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.faceCanvas = document.getElementById('faceCanvas');
        this.faceCtx = this.faceCanvas.getContext('2d');
        
        // Buttons
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        
        // Status
        this.apiStatus = document.getElementById('apiStatus');
        this.statusDot = document.getElementById('statusDot');
        this.statusText = document.getElementById('statusText');
        
        // Stats
        this.fpsElement = document.getElementById('fps');
        this.faceCountElement = document.getElementById('faceCount');
        this.apiCallsElement = document.getElementById('apiCalls');
        
        // Results
        this.resultIcon = document.getElementById('resultIcon');
        this.resultStatus = document.getElementById('resultStatus');
        this.resultConfidence = document.getElementById('resultConfidence');
        this.resultBadge = document.getElementById('resultBadge');
        this.resultCard = document.getElementById('resultCard');
        this.previewLabel = document.getElementById('previewLabel');
        this.historyList = document.getElementById('historyList');
        
        // Settings
        this.apiUrl = document.getElementById('apiUrl');
        this.sendRate = document.getElementById('sendRate');
        this.confidenceSlider = document.getElementById('confidence');
        this.confidenceValue = document.getElementById('confidenceValue');
        
        // State
        this.isRunning = false;
        this.faceDetection = null;
        this.stream = null;
        this.lastSendTime = 0;
        this.faceCount = 0;
        this.apiCalls = 0;
        this.fps = 0;
        this.frameCount = 0;
        this.lastFpsTime = Date.now();
        this.currentFace = null;
        this.history = [];
        
        // Bind events
        this.bindEvents();
        
        // Check API on load
        this.checkAPIStatus();
    }
    
    bindEvents() {
        this.startBtn.addEventListener('click', () => this.start());
        this.stopBtn.addEventListener('click', () => this.stop());
        this.confidenceSlider.addEventListener('input', (e) => {
            this.confidenceValue.textContent = e.target.value;
        });
    }
    
    async checkAPIStatus() {
        try {
            const response = await fetch(`${this.apiUrl.value}/health`);
            const data = await response.json();
            
            if (data.status === 'healthy') {
                this.statusDot.className = 'status-dot online';
                this.statusText.textContent = `API Online (${data.device})`;
            } else {
                throw new Error('API unhealthy');
            }
        } catch (error) {
            this.statusDot.className = 'status-dot offline';
            this.statusText.textContent = 'API Offline - Check if server is running';
            console.error('API check failed:', error);
        }
    }
    
    async start() {
        if (this.isRunning) return;
        
        try {
            // Get webcam
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 1280, height: 720 }
            });
            this.video.srcObject = this.stream;
            
            // Wait for video to load
            await new Promise(resolve => {
                this.video.onloadedmetadata = resolve;
            });
            
            // Set canvas size
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            
            // Initialize MediaPipe Face Detection
            this.faceDetection = new FaceDetection({
                locateFile: (file) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`;
                }
            });
            
            this.faceDetection.setOptions({
                model: 'short',
                minDetectionConfidence: 0.5
            });
            
            this.faceDetection.onResults((results) => this.onResults(results));
            
            // Start detection loop
            this.isRunning = true;
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            
            this.detectLoop();
            
        } catch (error) {
            console.error('Error starting:', error);
            alert('Cannot access webcam: ' + error.message);
        }
    }
    
    stop() {
        this.isRunning = false;
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        if (this.faceDetection) {
            this.faceDetection.close();
            this.faceDetection = null;
        }
        
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    async detectLoop() {
        if (!this.isRunning) return;
        
        // Send frame to MediaPipe
        await this.faceDetection.send({ image: this.video });
        
        // Calculate FPS
        this.frameCount++;
        const now = Date.now();
        if (now - this.lastFpsTime >= 1000) {
            this.fps = this.frameCount;
            this.fpsElement.textContent = this.fps;
            this.frameCount = 0;
            this.lastFpsTime = now;
        }
        
        requestAnimationFrame(() => this.detectLoop());
    }
    
    onResults(results) {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        if (results.detections && results.detections.length > 0) {
            this.faceCount = results.detections.length;
            this.faceCountElement.textContent = this.faceCount;
            
            // Get first face
            const detection = results.detections[0];
            const box = detection.boundingBox;
            
            // Calculate coordinates
            const x = box.xCenter * this.canvas.width - (box.width * this.canvas.width) / 2;
            const y = box.yCenter * this.canvas.height - (box.height * this.canvas.height) / 2;
            const width = box.width * this.canvas.width;
            const height = box.height * this.canvas.height;
            
            // Draw bounding box
            this.ctx.strokeStyle = '#00ff00';
            this.ctx.lineWidth = 3;
            this.ctx.strokeRect(x, y, width, height);
            
            // Store current face
            this.currentFace = { x, y, width, height };
            
            // Send to API if enough time passed
            const sendInterval = 1000 / parseInt(this.sendRate.value);
            const now = Date.now();
            
            if (now - this.lastSendTime >= sendInterval) {
                this.sendToAPI(x, y, width, height);
                this.lastSendTime = now;
            }
            
        } else {
            this.faceCount = 0;
            this.faceCountElement.textContent = this.faceCount;
            this.currentFace = null;
        }
    }
    
    async sendToAPI(x, y, width, height) {
        try {
            // Crop face from video
            const cropCanvas = document.createElement('canvas');
            cropCanvas.width = width;
            cropCanvas.height = height;
            const cropCtx = cropCanvas.getContext('2d');
            
            cropCtx.drawImage(
                this.video,
                x, y, width, height,
                0, 0, width, height
            );
            
            // Display cropped face
            this.faceCtx.clearRect(0, 0, this.faceCanvas.width, this.faceCanvas.height);
            this.faceCtx.drawImage(cropCanvas, 0, 0, this.faceCanvas.width, this.faceCanvas.height);
            this.previewLabel.textContent = 'Analyzing...';
            
            // Convert to base64
            const base64 = cropCanvas.toDataURL('image/jpeg', 0.8).split(',')[1];
            
            // Send to API
            const response = await fetch(`${this.apiUrl.value}/predict/base64`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: base64 })
            });
            
            const result = await response.json();
            
            this.apiCalls++;
            this.apiCallsElement.textContent = this.apiCalls;
            
            if (result.success) {
                this.displayResult(result.data);
            } else {
                this.previewLabel.textContent = result.message;
            }
            
        } catch (error) {
            console.error('API call failed:', error);
            this.previewLabel.textContent = 'API Error';
        }
    }
    
    displayResult(data) {
        const isLive = data.is_live;
        const confidence = data.confidence;
        const prediction = data.prediction;
        
        // Update result card
        this.resultIcon.innerHTML = isLive ? '<span>✅</span>' : '<span>❌</span>';
        this.resultStatus.textContent = prediction;
        this.resultStatus.style.color = isLive ? '#27ae60' : '#e74c3c';
        this.resultConfidence.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
        
        // Update badge
        this.resultBadge.textContent = prediction;
        this.resultBadge.className = `result-badge ${isLive ? 'live' : 'spoof'}`;
        
        // Update card background
        this.resultCard.style.background = isLive ? '#d4edda' : '#f8d7da';
        
        // Update preview label
        this.previewLabel.textContent = `${prediction} (${(confidence * 100).toFixed(1)}%)`;
        this.previewLabel.style.color = isLive ? '#27ae60' : '#e74c3c';
        this.previewLabel.style.fontWeight = 'bold';
        
        // Draw result on canvas
        if (this.currentFace) {
            const { x, y, width, height } = this.currentFace;
            this.ctx.strokeStyle = isLive ? '#00ff00' : '#ff0000';
            this.ctx.lineWidth = 4;
            this.ctx.strokeRect(x, y, width, height);
            
            // Draw label
            this.ctx.fillStyle = isLive ? '#00ff00' : '#ff0000';
            this.ctx.fillRect(x, y - 30, width, 30);
            this.ctx.fillStyle = '#fff';
            this.ctx.font = 'bold 20px Arial';
            this.ctx.fillText(`${prediction} (${(confidence * 100).toFixed(0)}%)`, x + 5, y - 8);
        }
        
        // Add to history
        this.addToHistory(prediction, confidence, isLive);
    }
    
    addToHistory(prediction, confidence, isLive) {
        const time = new Date().toLocaleTimeString();
        
        const item = {
            prediction,
            confidence,
            isLive,
            time
        };
        
        this.history.unshift(item);
        if (this.history.length > 10) {
            this.history.pop();
        }
        
        this.renderHistory();
    }
    
    renderHistory() {
        if (this.history.length === 0) {
            this.historyList.innerHTML = '<p class="history-empty">No detections yet</p>';
            return;
        }
        
        this.historyList.innerHTML = this.history.map(item => `
            <div class="history-item">
                <span>${item.isLive ? '✅' : '❌'}</span>
                <span class="badge ${item.isLive ? 'live' : 'spoof'}">${item.prediction}</span>
                <span>${(item.confidence * 100).toFixed(1)}%</span>
                <span class="time">${item.time}</span>
            </div>
        `).join('');
    }
}

// Initialize app when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.app = new FaceAntiSpoofingApp();
});
