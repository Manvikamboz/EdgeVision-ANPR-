import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import shutil

def auto_label(source_dir, output_base):
    # Load YOLO11 Nano model (Optimized for Edge)
    print("Loading YOLO11n model...")
    model = YOLO("models/yolo11n.pt")
    
    # Mapping COCO IDs to our project IDs
    # COCO: car(2), motorcycle(3), bus(5), truck(7)
    # Project: car(0), motorcycle(1), bus(2), truck(3)
    class_map = {2: 0, 3: 1, 5: 2, 7: 3}
    
    img_exts = [".jpg", ".jpeg", ".png", ".webp"]
    images = [f for f in Path(source_dir).iterdir() if f.suffix.lower() in img_exts]
    
    print(f"Found {len(images)} images in {source_dir}")
    
    # Ensure output directories exist
    img_out = Path(output_base) / "images/train"
    lbl_out = Path(output_base) / "labels/train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in images:
        results = model.predict(img_path, conf=0.25, verbose=False)
        
        found_vehicle = False
        label_lines = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in class_map:
                    found_vehicle = True
                    new_id = class_map[cls_id]
                    # Get normalized coordinates [x_center, y_center, width, height]
                    xywhn = box.xywhn[0].tolist()
                    label_lines.append(f"{new_id} {' '.join(map(str, xywhn))}")
        
        if found_vehicle:
            # Copy image to dataset if not already there
            target_img = img_out / img_path.name
            if not target_img.exists():
                shutil.copy(img_path, target_img)
            
            # Save label file
            target_lbl = lbl_out / f"{img_path.stem}.txt"
            with open(target_lbl, "w") as f:
                f.write("\n".join(label_lines))
            
            count += 1
            if count % 10 == 0:
                print(f"Processed {count} images...")

    print(f"\n✅ Done! Generated labels for {count} images using YOLO11n.")
    print(f"Images are in: {img_out}")
    print(f"Labels are in: {lbl_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Folder containing new images")
    parser.add_argument("--output", default="data/vehicle_dataset", help="Dataset root folder")
    args = parser.parse_args()
    
    auto_label(args.source, args.output)
