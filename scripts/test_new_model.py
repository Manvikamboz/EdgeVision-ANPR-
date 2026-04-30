import cv2
import os
from ultralytics import YOLO
from pathlib import Path

def test_on_samples():
    # Load your new model
    model_path = "models/best_vehicle_yolo11.pt"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return
        
    model = YOLO(model_path)
    
    # Samples to test
    samples = [
        "data/test/Bus/004281_17.jpg",
        "data/test/Car/004008_18.jpg",
        "data/test/Motorcycle/003464_00.jpg",
        "data/test/Truck/003490_17.jpg"
    ]
    
    output_dir = Path("outputs/test_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Testing new YOLO11 model on samples...")
    
    for img_path in samples:
        if not os.path.exists(img_path):
            print(f"  Skipping {img_path} (not found)")
            continue
            
        print(f"  Processing {img_path}...")
        results = model.predict(img_path, save=False, conf=0.45)
        
        # Load image for manual drawing (YOLO results.plot() is also an option)
        img = cv2.imread(img_path)
        
        for r in results:
            for box in r.boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])
                
                # Draw box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Save result
        out_path = output_dir / Path(img_path).name
        cv2.imwrite(str(out_path), img)
        print(f"  ✅ Saved to {out_path}")

if __name__ == "__main__":
    test_on_samples()
