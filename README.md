# ◈ EdgeSight – Edge-Optimized Vehicle Intelligence System

> Real-time vehicle detection · License plate recognition · OCR  
> Designed for edge deployment with minimal computational cost.

---

## 🗂️ Project Structure

```
EgdeVision_ANPR/
├── configs/
│   └── vehicle_detection.yaml     ← All model/training/preprocessing settings
├── data/
│   ├── raw/                        ← Original images & videos
│   ├── processed/                  ← Cropped plate images for OCR
│   └── vehicle_dataset/            ← YOLO-format dataset (images + labels)
│       ├── images/{train,val,test}
│       └── labels/{train,val,test}
├── models/                         ← Trained weight files (.pt, .onnx)
├── notebooks/
│   └── task1_vehicle_detection_colab.py  ← Full Colab training script
├── outputs/                        ← Detection results, metrics, videos
├── scripts/
│   └── optimize.py                 ← ONNX export + INT8 quantization
├── src/
│   ├── __init__.py
│   ├── detector.py                 ← VehicleDetector (YOLOv8n wrapper)
│   ├── preprocessor.py             ← CLAHE + denoise + resize pipeline
│   ├── metrics.py                  ← mAP, FPS, model-size evaluation
│   └── utils.py                    ← Logging, I/O, YOLO label helpers
├── app/
│   └── streamlit_app.py            ← Web UI (image + video detection)
├── main.py                         ← CLI entry point
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Deep Learning | PyTorch |
| Detection | YOLOv8n (Ultralytics) |
| OCR | EasyOCR |
| Image Processing | OpenCV |
| Web UI | Streamlit |
| Training | Google Colab |
| Optimization | ONNX + INT8 Quantization |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

### 3. CLI – detect on an image

```bash
python main.py --source data/raw/sample.jpg --save
```

### 4. CLI – detect on a video

```bash
python main.py --source data/raw/traffic.mp4 --save --json
```

### 5. CLI – with metrics benchmark

```bash
python main.py --source data/raw/sample.jpg --metrics
```

---

## 🎯 Task 1: Vehicle Detection

**Goal**: Detect vehicles in images/video frames using YOLOv8n.

**Target classes**: car · motorcycle · bus · truck

**Deliverables**:
- ✅ Bounding boxes `[x1, y1, x2, y2]` in pixel coordinates
- ✅ Confidence scores per detection
- ✅ mAP@50, per-class AP
- ✅ Inference time (ms) & FPS
- ✅ Model size (MB)

### Train on Google Colab

1. Open `notebooks/task1_vehicle_detection_colab.py`
2. Copy all cells into a new **Colab notebook** (GPU runtime)
3. Run top to bottom — training → evaluation → ONNX export
4. Download `best.pt` → place in `models/best_vehicle.pt`
5. Update `configs/vehicle_detection.yaml`:
   ```yaml
   model:
     custom_weights: models/best_vehicle.pt
   ```

---

## ⚡ Model Optimization

```bash
# Export to ONNX + INT8 quantize + benchmark
python scripts/optimize.py --weights models/best_vehicle.pt --benchmark
```

| Format | Size | Notes |
|--------|------|-------|
| PyTorch FP32 | ~6 MB | Training & full precision |
| ONNX FP32 | ~12 MB | Cross-platform inference |
| ONNX INT8 | ~3–4 MB | Edge deployment |

---

## 📊 Expected Performance (YOLOv8n, COCO vehicles)

| Metric | Expected Value |
|--------|---------------|
| mAP@50 | ~0.55 – 0.65 |
| Inference (CPU) | ~20–40 ms |
| Inference (GPU) | ~3–8 ms |
| FPS (CPU) | ~25–50 |
| Model Size | ~6 MB |

---

## 🛣️ Pipeline Roadmap

- [x] **Task 1** – Vehicle Detection (YOLOv8n)
- [ ] **Task 2** – License Plate Detection (YOLOv8n fine-tuned)
- [ ] **Task 3** – OCR & Text Extraction (EasyOCR)
- [ ] **Task 4** – Post-processing & Validation
- [ ] **Task 5** – Full Pipeline Integration
- [ ] **Task 6** – Edge Optimization & Deployment
