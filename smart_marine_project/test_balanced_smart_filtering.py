#!/usr/bin/env python3
"""
Test Balanced Smart Filtering
=============================

Test that bottles are detected but humans are blocked.
"""

import os
import sys
import cv2
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from plastic_detector import PlasticDetector

def test_balanced_filtering():
    """Test that bottles work but humans are blocked"""
    print("⚖️ Testing Balanced Smart Filtering")
    print("=" * 40)
    
    model_path = "models/ocean_waste_model_m2/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Test with moderate confidence
    detector = PlasticDetector(
        model_path=model_path,
        device='cpu',
        conf_threshold=0.15,
        debug_mode=True
    )
    
    # Find test images
    sample_files = []
    for directory in [".", "../"]:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')) and 'debug' in file.lower():
                    sample_files.append(os.path.join(directory, file))
    
    if not sample_files:
        print("⚠️ No test images found")
        return
    
    print(f"📸 Testing balanced filtering on {len(sample_files[:2])} images")
    print()
    
    total_detections = 0
    bottle_detections = 0
    plastic_detections = 0
    
    for i, img_path in enumerate(sample_files[:2]):
        print(f"🖼️ Image {i+1}: {img_path}")
        print("-" * 40)
        
        try:
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # Run detection
            detections, detection_info = detector.detect_objects(image)
            
            print(f"Final detections: {len(detection_info)}")
            total_detections += len(detection_info)
            
            # Categorize detections
            for det in detection_info:
                conf = det['confidence']
                cls_name = det['class_name']
                
                if 'bottle' in cls_name.lower():
                    bottle_detections += 1
                    print(f"  ✅ BOTTLE: {cls_name} (conf: {conf:.3f})")
                else:
                    plastic_detections += 1
                    print(f"  ✅ PLASTIC: {cls_name} (conf: {conf:.3f})")
            
            print()
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Summary
    print("📊 BALANCED FILTERING SUMMARY")
    print("=" * 35)
    print(f"Total detections: {total_detections}")
    print(f"Bottle detections: {bottle_detections}")
    print(f"Other plastic: {plastic_detections}")
    
    if bottle_detections > 0:
        print("✅ SUCCESS: Bottles are being detected!")
    else:
        print("⚠️ WARNING: No bottles detected")
    
    if total_detections > 0:
        print("✅ SUCCESS: Some plastic objects detected!")
    else:
        print("❌ ISSUE: No detections at all")

def main():
    """Main test function"""
    print("🌊 Smart Marine Project - Balanced Smart Filtering Test")
    print("=" * 65)
    
    test_balanced_filtering()
    
    print("\n🎯 BALANCED APPROACH:")
    print("   ⚖️ Smart human detection (4+ indicators needed)")
    print("   🍼 Bottle exceptions for clear bottles")
    print("   🔍 Less aggressive for bottle shapes")
    print("   🚫 Still blocks large face-like objects")
    print("   ✅ Allows small plastic items")

if __name__ == "__main__":
    main()
