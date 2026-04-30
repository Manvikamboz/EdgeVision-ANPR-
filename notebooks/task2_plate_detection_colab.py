# ═══════════════════════════════════════════════════════════════════════════════
#  EdgeSight – License Plate Detection Training (Task 2)
# ═══════════════════════════════════════════════════════════════════════════════

# --- CELL 1: Mount Google Drive ---
from google.colab import drive
drive.mount('/content/drive')

# --- CELL 2: Install Ultralytics ---
!pip install ultralytics

# --- CELL 3: Unzip Plate Dataset ---
# Assuming you uploaded plate_dataset.zip to your Drive root
!unzip /content/drive/MyDrive/plate_dataset.zip -d /content/

# --- CELL 4: Create Dataset Configuration ---
import yaml

config = {
    'path': '/content/data/plate_dataset',  # Dataset root
    'train': 'images/train',                # Train images
    'val': 'images/train',                  # Use train as val if you don't have separate val images
    'nc': 1,                                # Only 1 class: License Plate
    'names': ['license_plate']
}

with open('/content/plate_dataset.yaml', 'w') as f:
    yaml.dump(config, f)

# --- CELL 5: Start Training (YOLO11) ---
from ultralytics import YOLO

# Load the base YOLO11 nano model
model = YOLO("yolo11n.pt")

# Start training specialized for license plates
results = model.train(
    data="/content/plate_dataset.yaml",
    epochs=100,    # Plates need more epochs because they are small
    imgsz=640,     # High res is better for small objects like plates
    batch=16,      
    device=0       # Use T4 GPU
)

# --- CELL 6: Save Results to Drive ---
!cp /content/runs/detect/train/weights/best.pt /content/drive/MyDrive/best_plate_yolo11.pt
print("✅ Plate training complete! Weights saved to Drive as best_plate_yolo11.pt")
