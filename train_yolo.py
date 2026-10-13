"""
PocketScope AI — Training Script for Optional Blood Cell Model
Team BIONEX · BIOBYTE 2K26 College Expo
B.S. Abdur Rahman Crescent Institute of Science and Technology

This script is OPTIONAL. The PocketScope AI web application runs fully
without any trained model using classical OpenCV algorithms.

Instructions:
1. Download a public blood cell dataset in YOLO format (such as BCCD).
   Ensure you verify the dataset license for educational/research use.
2. Update the DATA_YAML path below to point to your dataset's data.yaml.
3. Recommended: Run this script in Google Colab with a free GPU runtime.
4. After training finishes, copy 'models/bloodcell.pt' and 'models/metrics.json'
   into the 'models/' folder of PocketScope AI.
"""

import os
import json
import shutil

# Try importing ultralytics (installed via: pip install -r requirements-yolo.txt)
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: 'ultralytics' is not installed.")
    print("Please install it by running: pip install -r requirements-yolo.txt")
    exit(1)

# Path to your dataset configuration file
# Replace this with the actual path to your downloaded dataset's data.yaml
DATA_YAML = "bccd/data.yaml"

# Training configuration
EPOCHS = 30
IMAGE_SIZE = 640
BASE_MODEL = "yolov8n.pt"  # Nano model: lightweight and fast
OUTPUT_DIR = "models"


def main():
    print("=" * 60)
    print("PocketScope AI — Optional Blood Cell Model Training")
    print("=" * 60)

    # Check if dataset yaml file exists
    if not os.path.exists(DATA_YAML):
        print(f"\n[Notice] Dataset file '{DATA_YAML}' not found.")
        print("Steps to prepare the dataset:")
        print("  1. Download a YOLO-format blood cell dataset (e.g. BCCD from Roboflow or Kaggle).")
        print("  2. Extract it into a folder, e.g. 'bccd/'.")
        print("  3. Ensure 'bccd/data.yaml' exists and points to train/val image folders.")
        print(f"  4. Set DATA_YAML = '{DATA_YAML}' at the top of this script.\n")
        return

    # Create destination models directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n1. Loading base architecture: {BASE_MODEL}...")
    model = YOLO(BASE_MODEL)

    print(f"\n2. Starting training for {EPOCHS} epochs on {DATA_YAML}...")
    train_results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        project="runs/detect",
        name="bloodcell_train",
        exist_ok=True,
        verbose=True,
    )

    print("\n3. Validating the trained model...")
    val_results = model.val()

    # Extract actual computed metrics
    metrics = {
        "mAP50": round(float(val_results.box.map50), 4),
        "mAP50_95": round(float(val_results.box.map), 4),
        "precision": round(float(val_results.box.mp), 4),
        "recall": round(float(val_results.box.mr), 4),
        "classes": list(val_results.names.values()) if hasattr(val_results, "names") else [],
        "epochs": EPOCHS,
        "base_model": BASE_MODEL,
    }

    # Save metrics.json
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved validation metrics to: {metrics_path}")
    print(json.dumps(metrics, indent=2))

    # Copy best weights to models/bloodcell.pt
    best_pt = os.path.join(train_results.save_dir, "weights", "best.pt")
    target_pt = os.path.join(OUTPUT_DIR, "bloodcell.pt")

    if os.path.exists(best_pt):
        shutil.copyfile(best_pt, target_pt)
        print(f"\nSuccessfully saved trained weights to: {target_pt}")
    else:
        print(f"\n[Warning] Could not locate {best_pt}. Please check training output.")

    print("\nTraining complete! When you start PocketScope AI, 'Blood smear (trained model)'")
    print("will now detect cells using these weights.")


if __name__ == "__main__":
    main()
