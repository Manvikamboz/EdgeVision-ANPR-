import os
import json
import time
from pathlib import Path
import cv2
import sys

# Ensure src modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.detector import VehicleDetector

def run_evaluation():
    # Setup paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "vehicle_detection.yaml"
    test_dir = project_root / "final test images " / "images"
    
    print(f"Loading configuration from {config_path}...")
    detector = VehicleDetector.from_config(config_path)
    
    predictions = {}
    inf_times = []
    
    image_paths = list(test_dir.glob("*.jpg"))
    print(f"Found {len(image_paths)} images to process.")
    
    for idx, img_path in enumerate(image_paths):
        if idx % 10 == 0:
            print(f"Processing {idx}/{len(image_paths)}...")
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        # Detect frame handles inference time internally, but we'll measure the whole wrapper just in case
        t0 = time.perf_counter()
        result = detector.detect_frame(img)
        t_ms = (time.perf_counter() - t0) * 1000
        inf_times.append(t_ms)
        
        # Format predictions for this image
        img_preds = []
        for det in result.detections:
            pred = {
                "class_id": int(det.class_id),
                "class_name": det.class_name,
                "confidence": float(det.confidence),
                "bbox": [int(x) for x in det.bbox],
            }
            if getattr(det, 'plate_text', None):
                pred["plate_text"] = det.plate_text
            if getattr(det, 'plate_bbox', None):
                pred["plate_bbox"] = [int(x) for x in det.plate_bbox]
            img_preds.append(pred)
            
        predictions[img_path.name] = img_preds

    # Write predictions.json
    with open(project_root / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)
        
    # Calculate efficiency
    avg_ms = sum(inf_times) / max(1, len(inf_times))
    avg_fps = 1000.0 / max(1.0, avg_ms)
    
    # Get model size (bytes -> MB)
    model_size_mb = 0.0
    try:
        model_path = project_root / "models" / "best_vehicle.pt"
        if os.path.exists(model_path):
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    except Exception as e:
        print(f"Warning: Could not get model size: {e}")
        
    efficiency = {
        "total_images": len(inf_times),
        "total_time_ms": round(sum(inf_times), 2),
        "average_inference_time_ms": round(avg_ms, 2),
        "average_fps": round(avg_fps, 2),
        "model_size_mb": round(model_size_mb, 2),
        "device": "cpu"
    }
    
    with open(project_root / "efficiency.json", "w") as f:
        json.dump(efficiency, f, indent=2)
        
    print("\n✅ Evaluation Complete!")
    print(f"- predictions.json saved ({len(predictions)} images processed).")
    print(f"- efficiency.json saved (Avg FPS: {efficiency['average_fps']}).")

if __name__ == "__main__":
    run_evaluation()
