# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EdgeSight – Task 1: Vehicle Detection Training                         ║
# ║  Google Colab Notebook  (copy cells into Colab or run as script)        ║
# ║  Model   : YOLOv8n (Nano)                                               ║
# ║  Dataset : COCO → filtered for vehicle classes (car, moto, bus, truck)  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# %% [markdown]
# # 🚗 EdgeSight – Task 1: Vehicle Detection
# **Model**: YOLOv8n (Nano) | **Framework**: PyTorch + Ultralytics
# **Target Classes**: car · motorcycle · bus · truck
# **Deliverables**: mAP@50, Inference Time, Model Size
#
# ⚠️ **IMPORTANT**: Switch to GPU runtime for faster training:
# Runtime → Change runtime type → T4 GPU → Save

# %% ── Cell 1: Install dependencies ──────────────────────────────────────────
# Run this cell first — installs all required packages
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

print("Installing packages (this takes ~60 seconds on first run)...")
install("ultralytics>=8.2.0")
install("opencv-python-headless")
install("onnx>=1.16.0")
install("onnxruntime>=1.18.0")
print("✓ All packages installed")

import os
import yaml
import json
import shutil
import zipfile
from pathlib import Path

import torch
print(f"\nPyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.cuda.is_available()}")
print(f"Device   : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

if not torch.cuda.is_available():
    print("\n⚠️  WARNING: Running on CPU — training will be very slow!")
    print("   Go to: Runtime → Change runtime type → T4 GPU")
else:
    print("\n✅ GPU detected — good to go!")

# %% ── Cell 2: Project directories ───────────────────────────────────────────
BASE_DIR    = Path("/content/EdgeSight")
DATA_DIR    = BASE_DIR / "data"
MODELS_DIR  = BASE_DIR / "models"
RUNS_DIR    = BASE_DIR / "runs"
EXPORT_DIR  = BASE_DIR / "exports"

for d in [DATA_DIR, MODELS_DIR, RUNS_DIR, EXPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("✓ Directories created")

# %% ── Cell 3: Download Dataset ───────────────────────────────────────────────
# Option A ─ Roboflow (recommended – replace with your workspace/project)
# from roboflow import Roboflow
# rf = Roboflow(api_key="YOUR_API_KEY")
# project = rf.workspace("your-workspace").project("vehicle-detection")
# dataset = project.version(1).download("yolov8")
# DATASET_YAML = dataset.location + "/data.yaml"

# Option B ─ COCO128 demo (quick test, 128 images, vehicles included)
# !wget -q https://ultralytics.com/assets/coco128.zip -O /content/coco128.zip
# !unzip -q /content/coco128.zip -d /content/

# Option C ─ Manual upload
# from google.colab import files
# uploaded = files.upload()  # Upload your zipped dataset

# ── For this script we use COCO128 as a demo ─────────────────────────────────
import urllib.request
print("Downloading COCO128 demo dataset…")
urllib.request.urlretrieve(
    "https://ultralytics.com/assets/coco128.zip",
    "/content/coco128.zip"
)
with zipfile.ZipFile("/content/coco128.zip", "r") as z:
    z.extractall("/content/")
print("✓ Dataset extracted")

# Filter to vehicle classes (2=car, 3=moto, 5=bus, 7=truck) and build YAML
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
DATASET_YAML = str(DATA_DIR / "vehicle_dataset.yaml")

dataset_cfg = {
    "path": "/content/coco128",
    "train": "images/train2017",
    "val":   "images/train2017",   # COCO128 has no separate val → use train
    "nc": 4,
    "names": ["car", "motorcycle", "bus", "truck"],
}
with open(DATASET_YAML, "w") as f:
    yaml.dump(dataset_cfg, f)
print(f"✓ Dataset YAML → {DATASET_YAML}")

# %% ── Cell 4: Build vehicle-only label files ─────────────────────────────────
# COCO128 labels have 80 classes; remap to vehicle-only (4 classes)

COCO_TO_VEHICLE = {2: 0, 3: 1, 5: 2, 7: 3}  # COCO id → new class id
label_dir = Path("/content/coco128/labels/train2017")
remapped_dir = DATA_DIR / "labels" / "train2017"
remapped_dir.mkdir(parents=True, exist_ok=True)

kept, skipped = 0, 0
for lbl_file in label_dir.glob("*.txt"):
    out_lines = []
    with open(lbl_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id in COCO_TO_VEHICLE:
                new_cls = COCO_TO_VEHICLE[cls_id]
                out_lines.append(f"{new_cls} " + " ".join(parts[1:]))
    if out_lines:
        with open(remapped_dir / lbl_file.name, "w") as f:
            f.write("\n".join(out_lines))
        kept += 1
    else:
        skipped += 1

print(f"✓ Labels remapped: {kept} images with vehicles, {skipped} skipped")

# Update YAML to point to remapped labels
dataset_cfg["label_dir"] = str(remapped_dir)
with open(DATASET_YAML, "w") as f:
    yaml.dump(dataset_cfg, f)

# %% ── Cell 5: Training ───────────────────────────────────────────────────────
from ultralytics import YOLO

MODEL_BASE = "yolov8n.pt"   # Nano – ~6 MB, ~3.2 GFLOPs

model = YOLO(MODEL_BASE)

results = model.train(
    data      = DATASET_YAML,
    epochs    = 80,
    imgsz     = 640,
    batch     = 16,
    device    = 0 if torch.cuda.is_available() else "cpu",
    project   = str(RUNS_DIR),
    name      = "vehicle_det_v1",
    optimizer = "AdamW",
    lr0       = 0.01,
    weight_decay = 0.0005,
    warmup_epochs = 3,
    patience  = 20,          # early stopping
    # Augmentation
    hsv_h     = 0.015,
    hsv_s     = 0.7,
    hsv_v     = 0.4,
    degrees   = 5.0,
    translate = 0.1,
    scale     = 0.5,
    fliplr    = 0.5,
    mosaic    = 1.0,
    mixup     = 0.1,
    save      = True,
    plots     = True,
)

print("✓ Training complete!")
print(f"   Best weights → {RUNS_DIR}/vehicle_det_v1/weights/best.pt")

# %% ── Cell 6: Evaluate – mAP & validation metrics ───────────────────────────
best_weights = str(RUNS_DIR / "vehicle_det_v1" / "weights" / "best.pt")
eval_model = YOLO(best_weights)

metrics = eval_model.val(data=DATASET_YAML, imgsz=640, device=0 if torch.cuda.is_available() else "cpu")

print("\n─── Validation Metrics ───────────────────────────")
print(f"mAP@50          : {metrics.box.map50:.4f}")
print(f"mAP@50-95       : {metrics.box.map:.4f}")
print(f"Precision       : {metrics.box.mp:.4f}")
print(f"Recall          : {metrics.box.mr:.4f}")

# Per-class AP
class_names = ["car", "motorcycle", "bus", "truck"]
print("\nPer-class AP@50:")
for name, ap in zip(class_names, metrics.box.ap50):
    print(f"  {name:<14} {ap:.4f}")

# %% ── Cell 7: Benchmark – Inference Speed ────────────────────────────────────
import time
import cv2
import numpy as np

# Create a dummy frame for timing
dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

WARMUP = 10
ITERS  = 100

infer_model = YOLO(best_weights)
device_str  = "0" if torch.cuda.is_available() else "cpu"

# Warm up
for _ in range(WARMUP):
    infer_model.predict(dummy, imgsz=640, device=device_str, verbose=False)

times = []
for _ in range(ITERS):
    t0 = time.perf_counter()
    infer_model.predict(dummy, imgsz=640, device=device_str, verbose=False)
    times.append((time.perf_counter() - t0) * 1000)

times = np.array(times)
print("\n─── Inference Speed ──────────────────────────────")
print(f"Iterations      : {ITERS}")
print(f"Mean            : {times.mean():.2f} ms")
print(f"Std             : {times.std():.2f} ms")
print(f"Min             : {times.min():.2f} ms")
print(f"Max             : {times.max():.2f} ms")
print(f"FPS (mean)      : {1000/times.mean():.1f}")

# %% ── Cell 8: Model Size ─────────────────────────────────────────────────────
best_path = Path(best_weights)
size_mb   = best_path.stat().st_size / (1024**2)
last_path = best_path.parent / "last.pt"
last_mb   = last_path.stat().st_size / (1024**2) if last_path.exists() else 0

print("\n─── Model Size ────────────────────────────────────")
print(f"best.pt         : {size_mb:.2f} MB")
if last_mb:
    print(f"last.pt         : {last_mb:.2f} MB")

# %% ── Cell 9: Save full metrics report ───────────────────────────────────────
report = {
    "task": "Task 1 – Vehicle Detection",
    "model": "YOLOv8n",
    "dataset": DATASET_YAML,
    "training": {
        "epochs": 80,
        "batch_size": 16,
        "img_size": 640,
        "optimizer": "AdamW",
    },
    "validation": {
        "mAP@50":     round(float(metrics.box.map50), 4),
        "mAP@50-95":  round(float(metrics.box.map),   4),
        "precision":  round(float(metrics.box.mp),     4),
        "recall":     round(float(metrics.box.mr),     4),
        "per_class_AP50": {
            name: round(float(ap), 4)
            for name, ap in zip(class_names, metrics.box.ap50)
        },
    },
    "speed": {
        "mean_ms":  round(float(times.mean()), 2),
        "std_ms":   round(float(times.std()),  2),
        "fps_mean": round(1000 / float(times.mean()), 1),
    },
    "model_size": {
        "best_pt_mb": round(size_mb, 2),
    },
}

report_path = RUNS_DIR / "vehicle_det_v1" / "task1_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\n✓ Report saved → {report_path}")
print(json.dumps(report, indent=2))

# %% ── Cell 10: ONNX Export ───────────────────────────────────────────────────
onnx_path = EXPORT_DIR / "vehicle_det_v1.onnx"

export_model = YOLO(best_weights)
export_model.export(
    format   = "onnx",
    imgsz    = 640,
    opset    = 17,
    simplify = True,
    dynamic  = False,
)

# Move exported file
raw_onnx = Path(best_weights).with_suffix(".onnx")
if raw_onnx.exists():
    shutil.copy(raw_onnx, onnx_path)

onnx_mb = onnx_path.stat().st_size / (1024**2)
print(f"\n✓ ONNX model exported → {onnx_path}  ({onnx_mb:.2f} MB)")

# %% ── Cell 11: INT8 Quantization ─────────────────────────────────────────────
# Quantize ONNX model to INT8 using onnxruntime's quantization tool
# !pip install onnxruntime-tools -q   # only if not already installed

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quant_path = EXPORT_DIR / "vehicle_det_v1_int8.onnx"
    quantize_dynamic(
        model_input   = str(onnx_path),
        model_output  = str(quant_path),
        weight_type   = QuantType.QInt8,
    )
    quant_mb = quant_path.stat().st_size / (1024**2)
    print(f"✓ INT8 quantised model → {quant_path}  ({quant_mb:.2f} MB)")
    print(f"  Size reduction: {onnx_mb:.2f} MB → {quant_mb:.2f} MB  "
          f"({(1-quant_mb/onnx_mb)*100:.1f}% smaller)")
except ImportError:
    print("onnxruntime quantization not available – skipping INT8 step.")

# %% ── Cell 12: Download weights to local machine ─────────────────────────────
from google.colab import files

print("Downloading best.pt …")
files.download(best_weights)

print("Downloading task1_report.json …")
files.download(str(report_path))

print("Downloading ONNX model …")
files.download(str(onnx_path))

print("\n✅ All Task 1 deliverables downloaded!")
print("   Place best.pt → models/best_vehicle.pt in your local project.")

# %% [markdown]
# ## ✅ Task 1 Complete – Deliverables Summary
#
# | Deliverable | Value |
# |------------|-------|
# | Bounding boxes | ✓ (x1,y1,x2,y2 pixel coords) |
# | Confidence scores | ✓ (0-1 per detection) |
# | mAP@50 | See task1_report.json |
# | Inference Time | See task1_report.json |
# | Model Size | ~6 MB (FP32) → ~3 MB (INT8) |
# | ONNX Export | ✓ |
# | INT8 Quantization | ✓ |
