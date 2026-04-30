import os
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download

def auto_label_plates(image_dir, label_dir):
    # Load a specialized pre-trained Plate Detector from HuggingFace
    print("🚀 Loading specialized License Plate Detector from HuggingFace Hub...")
    try:
        # This downloads the best.pt from the keremberke repo
        repo_id = "keremberke/yolov8n-license-plate-detector"
        weights_path = hf_hub_download(repo_id=repo_id, filename="best.pt")
        model = YOLO(weights_path)
    except Exception as e:
        print(f"❌ Could not load from Hub: {e}")
        print("Falling back to standard YOLO11n (might be less accurate for plates)")
        model = YOLO("yolo11n.pt")
    
    img_path = Path(image_dir)
    lbl_path = Path(label_dir)
    lbl_path.mkdir(parents=True, exist_ok=True)
    
    images = list(img_path.glob("*.jpg")) + list(img_path.glob("*.jpeg")) + list(img_path.glob("*.png"))
    print(f"🔍 Found {len(images)} images. Starting auto-labeling on CPU...")
    
    count = 0
    for img_p in tqdm(images):
        results = model.predict(img_p, conf=0.4, verbose=False)
        
        label_file = lbl_path / f"{img_p.stem}.txt"
        
        with open(label_file, "w") as f:
            for r in results:
                for box in r.boxes:
                    # Class 0 = license_plate
                    xywhn = box.xywhn[0].tolist()
                    f.write(f"0 {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n")
        count += 1
        
    print(f"\n✅ Finished! Generated {count} label files in {label_dir}")
    print("Next step: Zip these and move to Google Colab for training!")

if __name__ == "__main__":
    auto_label_plates("data/plate_dataset/images/train", "data/plate_dataset/labels/train")
