import json
from pathlib import Path
from PIL import Image

data_dir = Path("data/vehicle_dataset")
images_dir = data_dir / "images"
labels_dir = data_dir / "labels"

splits = ["train", "val", "test"]

count_success = 0
count_missing = 0

print("Starting JSON to YOLO format conversion...")

for split in splits:
    split_images_dir = images_dir / split
    split_labels_dir = labels_dir / split
    
    # Ensure label subdirectories exist
    split_labels_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in split_images_dir.glob("*.jpg"):
        json_path = labels_dir / (img_path.stem + ".json")
        txt_path = split_labels_dir / (img_path.stem + ".txt")
        
        if not json_path.exists():
            count_missing += 1
            continue
            
        with open(json_path, "r") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Error reading {json_path.name}: {e}")
                continue
                
        # Get image dimensions to normalize coordinates
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error opening image {img_path.name}: {e}")
            continue
            
        yolo_lines = []
        for box in data:
            cls_id = box.get("class_id", 1)
            
            # YOLO requires 0-indexed class IDs. If your JSON starts at 1, map it to 0.
            # Assuming 1 -> 0, 2 -> 1, etc.
            yolo_cls_id = max(0, cls_id - 1)
            
            x = box.get("x", 0)
            y = box.get("y", 0)
            w = box.get("width", 0)
            h = box.get("height", 0)
            
            # YOLO format requires normalized center_x, center_y, width, height
            cx = (x + w / 2) / img_width
            cy = (y + h / 2) / img_height
            nw = w / img_width
            nh = h / img_height
            
            # Clamp values between 0.0 and 1.0 just in case
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))
            
            yolo_lines.append(f"{yolo_cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            
        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))
            
        count_success += 1

print(f"✅ Successfully converted and sorted {count_success} labels into train/val/test folders.")
if count_missing > 0:
    print(f"⚠️ Missing JSON labels for {count_missing} images (this is normal if some images had no vehicles).")
