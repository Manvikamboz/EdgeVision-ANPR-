import os
from pathlib import Path

def analyze_dataset(dataset_path):
    labels_path = Path(dataset_path) / "labels/train"
    if not labels_path.exists():
        print(f"Error: {labels_path} not found.")
        return

    class_counts = {}
    class_names = ["car", "motorcycle", "bus", "truck"]
    
    label_files = list(labels_path.glob("*.txt"))
    print(f"Found {len(label_files)} label files in {labels_path}")

    for label_file in label_files:
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                class_id = int(parts[0])
                class_counts[class_id] = class_counts.get(class_id, 0) + 1

    print("\n--- Class Distribution (Training Set) ---")
    for i in range(4):
        count = class_counts.get(i, 0)
        status = "✅ OK" if count > 0 else "❌ MISSING"
        name = class_names[i] if i < len(class_names) else f"Unknown({i})"
        print(f"ID {i} ({name:10}): {count:5} instances {status}")

    if len(class_counts) < 4:
        print("\n⚠️ WARNING: Your dataset is missing one or more classes!")
        print("To fix this, add more images and labels for the missing classes.")

if __name__ == "__main__":
    analyze_dataset("data/vehicle_dataset")
