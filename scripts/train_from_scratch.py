"""
EdgeSight – Train Custom Vehicle Detection Model from Scratch
"""
import sys
from pathlib import Path
from ultralytics import YOLO
import torch

def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.cuda.is_available()}")
    device = "cpu"
    if torch.cuda.is_available():
        print(f"Device:  {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("⚠️ WARNING: Running on CPU! Training will take a very long time.")

    print("\n[EdgeSight] Initializing YOLOv8n from scratch (NO pre-trained weights)...")
    # Initialize from the YAML architecture file, this avoids loading pre-trained weights
    model = YOLO("yolov8n.yaml")

    data_yaml = str(Path(__file__).parent.parent / "data" / "vehicle_dataset.yaml")
    print(f"[EdgeSight] Training on dataset config: {data_yaml}")

    # Set epochs to 1 for a quick test run, but for a full run it should be around 80.
    epochs = 1 
    if len(sys.argv) > 1:
        try:
            epochs = int(sys.argv[1])
        except ValueError:
            pass

    print(f"[EdgeSight] Starting training for {epochs} epoch(s)...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16, # Adjust based on RAM
        device=device,
        project="runs",
        name="custom_vehicle_det",
        optimizer="AdamW",
        lr0=0.01,
        patience=20,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
    )

    print("\n[EdgeSight] Training completed!")
    # The best weights will be saved in runs/custom_vehicle_det/weights/best.pt
    print("Best weights are located at: runs/custom_vehicle_det/weights/best.pt")
    print("To use them, copy the best.pt file to the models/ folder as best_vehicle.pt")

if __name__ == "__main__":
    main()
