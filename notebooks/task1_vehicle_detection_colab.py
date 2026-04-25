# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EdgeSight – Task 1: Custom Vehicle Detection Training                  ║
# ║  Google Colab Notebook  (copy cells into Colab or run as script)        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# %% [markdown]
# # 🚗 EdgeSight – Custom Dataset Training
# **IMPORTANT**: Switch to GPU runtime for faster training!
# Go to: **Runtime → Change runtime type → T4 GPU → Save**

# %% ── Cell 1: Install dependencies ──────────────────────────────────────────
!pip install -q "ultralytics>=8.2.0"

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("⚠️ WARNING: Running on CPU! Change runtime to GPU!")
else:
    print(f"Device:  {torch.cuda.get_device_name(0)} ✅")

# %% ── Cell 2: Extract & Create Config ─────────────────────────────────────────
import yaml

# Unzip the uploaded file quietly (-q)
!unzip -q /content/vehicle_dataset.zip -d /content/

yaml_content = {
    "path": "/content/vehicle_dataset",
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": 4,
    "names": ["car", "motorcycle", "bus", "truck"]
}

with open("/content/vehicle_dataset.yaml", "w") as f:
    yaml.dump(yaml_content, f)

print("✓ Dataset extracted and YOLO config created!")

# %% ── Cell 4: Train the Model ───────────────────────────────────────────────
from ultralytics import YOLO

# Load the base nano model
model = YOLO("yolov8n.pt")

# Train! (Will take roughly 15-25 minutes on a T4 GPU for 80 epochs)
results = model.train(
    data="/content/vehicle_dataset.yaml",
    epochs=80,
    imgsz=640,
    batch=32,      # 32 is safe for Nano on T4 GPU
    device=0,
    project="/content/runs",
    name="custom_vehicle_det",
    optimizer="AdamW",
    lr0=0.01,
    patience=20,   # Early stopping if no improvement
    # Augmentations to prevent overfitting
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
)

print("✓ Training complete!")

# %% ── Cell 4: Evaluate & Save Model ───────────────────────────────
# See how well it did at the very end
metrics = model.val(data="/content/vehicle_dataset.yaml")

print("\n─── Final Accuracy ───────────────────────────")
print(f"mAP@50-95 : {metrics.box.map:.4f}")
print(f"mAP@50    : {metrics.box.map50:.4f}")

# Download the best weights to your computer
from google.colab import files
best_weights = "/content/runs/custom_vehicle_det/weights/best.pt"

print(f"\n🎉 SUCCESS! Downloading best_vehicle.pt to your computer...")
files.download(best_weights)
print("Once downloaded, place it in your local models/ folder, and update your local config!")
