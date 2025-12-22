## 📌 HƯỚNG DẪN RESUME TRAINING

### **Cách sử dụng:**

1. **Training lần đầu**: Chạy bình thường, checkpoint tự động lưu vào `last_model2.pth`

2. **Muốn dừng và train tiếp**: 
   - Dừng training (Interrupt kernel)
   - Lần sau, đổi `resume = True` ở cell trên
   - Chạy lại từ cell "Khởi tạo DataLoader" trở đi

3. **Checkpoint bao gồm**:
   - Model weights
   - Optimizer state (learning rate, momentum, etc.)
   - Scheduler state
   - Epoch hiện tại
   - Best metric & early stopping counter
   - History (loss, metrics qua các epoch)

---

### **⚠️ XỬ LÝ EARLY STOPPING:**

#### **Trường hợp 1: Bị early stop ở epoch 20/50**
```python
# Training đã dừng ở epoch 20 do early stopping
# no_improve = 12 (đạt patience)

# Muốn train tiếp:
resume = True
reset_early_stop = True  # ← QUAN TRỌNG: Reset counter về 0
cfg.num_epochs = 100     # Tăng số epoch
cfg.early_stop_patience = 15  # Tăng patience (tuỳ chọn)

# → Training sẽ tiếp tục từ epoch 21, early stop counter = 0
```

#### **Trường hợp 2: Dừng thủ công (Interrupt) ở epoch 30/200**
```python
# Training chưa bị early stop, bạn tự dừng

resume = True
reset_early_stop = False  # Giữ nguyên counter
cfg.num_epochs = 200      # Giữ nguyên hoặc tăng

# → Training tiếp tục từ epoch 31 với early stop counter hiện tại
```

#### **Trường hợp 3: Load best model để fine-tune**
```python
# Load best model thay vì last model
resume_checkpoint = "results/ssan_custom/model/best_model2.pth"
resume = True
reset_early_stop = True
cfg.base_lr = 1e-5  # Giảm learning rate để fine-tune

# ⚠️ Lưu ý: best_model2.pth chỉ có weights, không có optimizer
# Cần load thủ công:
checkpoint = torch.load(resume_checkpoint)
model.load_state_dict(checkpoint)  # Chỉ load weights
```

---

### **Ví dụ cụ thể:**
```python
# === LẦN 1: Train 50 epochs ===
cfg.num_epochs = 50
resume = False
# ... Kết quả: Early stop ở epoch 18 ...

# === LẦN 2: Resume và train tiếp ===
resume = True
reset_early_stop = True  # Reset counter
cfg.num_epochs = 100
cfg.early_stop_patience = 20  # Tăng patience
# ... Training tiếp từ epoch 19 → 100 ...
```