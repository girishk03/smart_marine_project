#!/usr/bin/env python3
"""
Create Plastic-Only Dataset
============================

This script filters the dataset to keep ONLY plastic images.
Removes metal (class 1), wood (class 3), and concrete (class 0) data.

Based on data.yaml:
- Class 0: concrete (REMOVE)
- Class 1: metal (REMOVE)
- Class 2: plastic (KEEP) ✅
- Class 3: wood (REMOVE)
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Configuration
PROJECT_ROOT = Path(__file__).parent
TRAIN_IMAGES = PROJECT_ROOT / "train" / "images"
TRAIN_LABELS = PROJECT_ROOT / "train" / "labels"
VALID_IMAGES = PROJECT_ROOT / "valid" / "images"
VALID_LABELS = PROJECT_ROOT / "valid" / "labels"

# New plastic-only directories
PLASTIC_TRAIN_IMAGES = PROJECT_ROOT / "train_plastic_only" / "images"
PLASTIC_TRAIN_LABELS = PROJECT_ROOT / "train_plastic_only" / "labels"
PLASTIC_VALID_IMAGES = PROJECT_ROOT / "valid_plastic_only" / "images"
PLASTIC_VALID_LABELS = PROJECT_ROOT / "valid_plastic_only" / "labels"

# Class to keep (plastic = class 2)
PLASTIC_CLASS = 2

def parse_label_file(label_path):
    """
    Parse YOLO label file and return classes found.
    Format: class_id x_center y_center width height
    """
    classes = set()
    lines = []
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        classes.add(class_id)
                        lines.append(line)
    except Exception as e:
        print(f"Error reading {label_path}: {e}")
        return set(), []
    
    return classes, lines

def filter_plastic_only(lines):
    """
    Filter label lines to keep only plastic (class 2).
    Also remap class 2 to class 0 (since it will be the only class).
    """
    plastic_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            if class_id == PLASTIC_CLASS:
                # Remap class 2 -> class 0 (plastic will be class 0 in new dataset)
                parts[0] = '0'
                plastic_lines.append(' '.join(parts))
    return plastic_lines

def process_dataset(images_dir, labels_dir, output_images_dir, output_labels_dir):
    """
    Process a dataset split (train or valid) and create plastic-only version.
    """
    # Create output directories
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all label files
    label_files = list(labels_dir.glob("*.txt"))
    
    stats = {
        'total': 0,
        'has_plastic': 0,
        'plastic_only': 0,
        'no_plastic': 0,
        'mixed': 0,
        'copied': 0
    }
    
    print(f"\n📂 Processing {images_dir.name}...")
    print(f"   Found {len(label_files)} label files")
    
    for label_path in tqdm(label_files, desc=f"Filtering {images_dir.name}"):
        stats['total'] += 1
        
        # Parse label file
        classes, lines = parse_label_file(label_path)
        
        if not classes:
            continue
        
        # Check if image has plastic
        has_plastic = PLASTIC_CLASS in classes
        has_other = any(c != PLASTIC_CLASS for c in classes)
        
        if has_plastic:
            stats['has_plastic'] += 1
            
            if not has_other:
                stats['plastic_only'] += 1
            else:
                stats['mixed'] += 1
            
            # Filter to keep only plastic annotations
            plastic_lines = filter_plastic_only(lines)
            
            if plastic_lines:
                # Find corresponding image
                image_name = label_path.stem
                image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
                image_path = None
                
                for ext in image_extensions:
                    potential_path = images_dir / f"{image_name}{ext}"
                    if potential_path.exists():
                        image_path = potential_path
                        break
                
                if image_path and image_path.exists():
                    # Copy image
                    shutil.copy2(image_path, output_images_dir / image_path.name)
                    
                    # Write filtered label file
                    output_label_path = output_labels_dir / label_path.name
                    with open(output_label_path, 'w') as f:
                        f.write('\n'.join(plastic_lines) + '\n')
                    
                    stats['copied'] += 1
        else:
            stats['no_plastic'] += 1
    
    return stats

def main():
    print("=" * 70)
    print("🔍 CREATE PLASTIC-ONLY DATASET")
    print("=" * 70)
    print()
    print("This will create a new dataset with ONLY plastic images.")
    print("Metal, wood, and concrete will be removed.")
    print()
    print("📋 Configuration:")
    print(f"   Keep: Class {PLASTIC_CLASS} (plastic) ✅")
    print(f"   Remove: Class 0 (concrete), Class 1 (metal), Class 3 (wood) 🚫")
    print()
    
    # Check if directories exist
    if not TRAIN_IMAGES.exists() or not TRAIN_LABELS.exists():
        print("❌ Error: Training dataset not found!")
        print(f"   Expected: {TRAIN_IMAGES}")
        return
    
    if not VALID_IMAGES.exists() or not VALID_LABELS.exists():
        print("❌ Error: Validation dataset not found!")
        print(f"   Expected: {VALID_IMAGES}")
        return
    
    print("✅ Source datasets found")
    print()
    
    # Process training set
    train_stats = process_dataset(
        TRAIN_IMAGES, TRAIN_LABELS,
        PLASTIC_TRAIN_IMAGES, PLASTIC_TRAIN_LABELS
    )
    
    # Process validation set
    valid_stats = process_dataset(
        VALID_IMAGES, VALID_LABELS,
        PLASTIC_VALID_IMAGES, PLASTIC_VALID_LABELS
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 FILTERING SUMMARY")
    print("=" * 70)
    
    print("\n🎯 TRAINING SET:")
    print(f"   Total images: {train_stats['total']}")
    print(f"   Has plastic: {train_stats['has_plastic']} ({train_stats['has_plastic']/train_stats['total']*100:.1f}%)")
    print(f"   Plastic only: {train_stats['plastic_only']}")
    print(f"   Mixed (plastic + other): {train_stats['mixed']}")
    print(f"   No plastic (removed): {train_stats['no_plastic']}")
    print(f"   ✅ Copied: {train_stats['copied']} images")
    
    print("\n🎯 VALIDATION SET:")
    print(f"   Total images: {valid_stats['total']}")
    print(f"   Has plastic: {valid_stats['has_plastic']} ({valid_stats['has_plastic']/valid_stats['total']*100:.1f}%)")
    print(f"   Plastic only: {valid_stats['plastic_only']}")
    print(f"   Mixed (plastic + other): {valid_stats['mixed']}")
    print(f"   No plastic (removed): {valid_stats['no_plastic']}")
    print(f"   ✅ Copied: {valid_stats['copied']} images")
    
    total_original = train_stats['total'] + valid_stats['total']
    total_plastic = train_stats['copied'] + valid_stats['copied']
    reduction = (1 - total_plastic/total_original) * 100
    
    print("\n📈 OVERALL:")
    print(f"   Original dataset: {total_original} images")
    print(f"   Plastic-only dataset: {total_plastic} images")
    print(f"   Reduction: {reduction:.1f}%")
    
    print("\n" + "=" * 70)
    print("✅ PLASTIC-ONLY DATASET CREATED")
    print("=" * 70)
    print()
    print("📂 New dataset location:")
    print(f"   Train: {PLASTIC_TRAIN_IMAGES.parent}")
    print(f"   Valid: {PLASTIC_VALID_IMAGES.parent}")
    print()
    print("📝 Next steps:")
    print("   1. Create data_plastic_only.yaml")
    print("   2. Train new model with plastic-only data")
    print("   3. Replace old model with new one")
    print()
    print("Run: python3 train_plastic_only_model.py")
    print()

if __name__ == "__main__":
    main()
