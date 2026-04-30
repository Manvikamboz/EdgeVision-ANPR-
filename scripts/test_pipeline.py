import sys
import os
from pathlib import Path
import cv2

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.detector import VehicleDetector

def test_pipeline():
    config_path = "configs/vehicle_detection.yaml"
    print(f"--- Testing Two-Stage Pipeline ---")
    detector = VehicleDetector.from_config(config_path)
    
    # Use one of the test images
    test_dir = Path("data/vehicle_dataset/images/test")
    test_images = list(test_dir.glob("*.jpg"))
    
    if not test_images:
        print("No test images found in data/vehicle_dataset/images/test")
        return

    img_path = test_images[0]
    print(f"\nProcessing: {img_path.name}")
    frame = cv2.imread(str(img_path))
    
    if frame is None:
        print("Failed to read image.")
        return

    result = detector.detect_frame(frame)
    print(f"Inference Time: {result.inference_time_ms:.2f} ms")
    print(f"Found {result.count} vehicles:")
    
    for i, det in enumerate(result.detections):
        print(f"  {i+1}. Vehicle: {det.class_name} (Conf: {det.confidence:.2f})")
        if det.plate_text:
            print(f"     Plate Detected: [{det.plate_text}] (Conf: {det.plate_conf:.2f})")
        else:
            print(f"     No plate detected or OCR failed.")

if __name__ == "__main__":
    test_pipeline()
