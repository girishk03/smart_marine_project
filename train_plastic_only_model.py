#!/usr/bin/env python3
"""
Train Plastic-Only Model
========================

Train a YOLOv5 model on the plastic-only dataset.
This will create a smaller, faster model that only detects plastic.
"""

import os
import sys
from pathlib import Path
import torch

print("=" * 70)
print("🚀 TRAIN PLASTIC-ONLY MODEL")
print("=" * 70)
print()

# Check if YOLOv5 is available
yolov5_path = Path(__file__).parent / "yolov5"
if not yolov5_path.exists():
    print("❌ YOLOv5 not found!")
    print(f"   Expected: {yolov5_path}")
    print()
    print("📥 Cloning YOLOv5...")
    os.system("git clone https://github.com/ultralytics/yolov5.git")
    print()

# Add YOLOv5 to path
sys.path.insert(0, str(yolov5_path))

# Training configuration
PROJECT_ROOT = Path(__file__).parent
DATA_YAML = PROJECT_ROOT / "data_plastic_only.yaml"
WEIGHTS = "yolov5m.pt"  # Start from pretrained YOLOv5m
IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 100
DEVICE = "0" if torch.cuda.is_available() else "cpu"

print("📋 Training Configuration:")
print(f"   Dataset: {DATA_YAML}")
print(f"   Base weights: {WEIGHTS}")
print(f"   Image size: {IMG_SIZE}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Epochs: {EPOCHS}")
print(f"   Device: {DEVICE}")
print()

# Check if dataset exists
if not DATA_YAML.exists():
    print("❌ Error: data_plastic_only.yaml not found!")
    print("   Run: python3 create_plastic_only_dataset.py first")
    sys.exit(1)

# Check if plastic-only dataset exists
train_dir = PROJECT_ROOT / "train_plastic_only" / "images"
if not train_dir.exists() or len(list(train_dir.glob("*"))) == 0:
    print("❌ Error: Plastic-only dataset not found!")
    print("   Run: python3 create_plastic_only_dataset.py first")
    sys.exit(1)

print("✅ Dataset ready")
print(f"   Training images: {len(list(train_dir.glob('*')))} images")
print()

# Training command
train_cmd = f"""
python3 yolov5/train.py \
    --img {IMG_SIZE} \
    --batch {BATCH_SIZE} \
    --epochs {EPOCHS} \
    --data {DATA_YAML} \
    --weights {WEIGHTS} \
    --device {DEVICE} \
    --project runs/train_plastic_only \
    --name plastic_model \
    --cache \
    --patience 20 \
    --save-period 10
"""

print("🎯 Starting training...")
print("-" * 70)
print(train_cmd.strip())
print("-" * 70)
print()

# Run training
result = os.system(train_cmd.strip().replace('\n', ' ').replace('\\', ''))

if result == 0:
    print()
    print("=" * 70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print()
    print("📂 Model saved to:")
    print("   runs/train_plastic_only/plastic_model/weights/best.pt")
    print()
    print("📝 Next steps:")
    print("   1. Backup old model:")
    print("      mv smart_marine_project/models/ocean_waste_model_m2/weights/best.pt \\")
    print("         smart_marine_project/models/ocean_waste_model_m2/weights/best_4class_backup.pt")
    print()
    print("   2. Copy new plastic-only model:")
    print("      cp runs/train_plastic_only/plastic_model/weights/best.pt \\")
    print("         smart_marine_project/models/ocean_waste_model_m2/weights/best.pt")
    print()
    print("   3. Update data.yaml:")
    print("      cp data_plastic_only.yaml data.yaml")
    print()
    print("   4. Test the new model:")
    print("      streamlit run reliable_web_app.py")
    print()
else:
    print()
    print("❌ Training failed!")
    print("   Check the error messages above")
    print()
