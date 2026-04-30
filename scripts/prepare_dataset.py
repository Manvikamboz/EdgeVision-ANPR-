import os
import shutil
from pathlib import Path

def prepare_yolo_dataset(source_base, target_base):
    source_base = Path(source_base)
    target_base = Path(target_base)
    
    # Class mapping based on your folders
    class_map = {
        "Car": 0,
        "Motorcycle": 1,
        "Bus": 2,
        "Truck": 3
    }
    
    splits = ["train", "test"] # Your current folders
    
    for split in splits:
        source_split = source_base / split
        if not source_split.exists():
            continue
            
        # Target folders (YOLO format uses 'val' instead of 'test' usually, but we'll map test to val for training)
        target_split_name = "train" if split == "train" else "val"
        img_dir = target_base / "images" / target_split_name
        lbl_dir = target_base / "labels" / target_split_name
        
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {split} split...")
        
        for class_folder, class_id in class_map.items():
            folder_path = source_split / class_folder
            if not folder_path.exists():
                print(f"  Warning: Folder {class_folder} not found in {split}")
                continue
                
            images = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.jpeg")) + list(folder_path.glob("*.png"))
            print(f"  Found {len(images)} images in {class_folder}")
            
            for img_path in images:
                # 1. Move/Copy image
                target_img_path = img_dir / img_path.name
                shutil.copy(img_path, target_img_path)
                
                # 2. Create placeholder label (Full frame: class_id 0.5 0.5 1.0 1.0)
                # Note: It's better to run auto_label.py later for precise boxes.
                target_lbl_path = lbl_dir / f"{img_path.stem}.txt"
                with open(target_lbl_path, "w") as f:
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
                    
    print(f"\n✅ Dataset organized in {target_base}")
    print("Next Step: Run 'scripts/auto_label.py' to refine these boxes!")

if __name__ == "__main__":
    prepare_yolo_dataset("data", "data/vehicle_dataset")
