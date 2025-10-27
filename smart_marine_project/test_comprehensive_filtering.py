#!/usr/bin/env python3
"""
Test Comprehensive Face/Skin Filtering and Plastic Detection
===========================================================

Test script to verify:
1. Zero tolerance face/skin filtering
2. Improved plastic object detection (not just bottles)
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from plastic_detector import PlasticDetector

def test_zero_tolerance_face_filtering():
    """Test that faces are NEVER detected as plastic"""
    print("🚫 Testing Zero Tolerance Face/Skin Filtering")
    print("=" * 55)
    
    model_path = "models/ocean_waste_model_m2/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Test with very low confidence to catch any face detections
    detector = PlasticDetector(
        model_path=model_path,
        device='cpu',
        conf_threshold=0.05,  # Very low to catch everything
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
    
    print(f"📸 Testing face filtering on {len(sample_files[:2])} images")
    print()
    
    total_face_detections = 0
    total_valid_plastic = 0
    
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
            
            # Analyze each detection for potential face/skin issues
            for j, det in enumerate(detection_info):
                conf = det['confidence']
                cls_name = det['class_name']
                bbox = det['bbox']
                
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                aspect_ratio = width / max(height, 1)
                
                # Calculate relative position and size
                img_h, img_w = image.shape[:2]
                rel_x = (bbox[0] + bbox[2]) / 2 / img_w
                rel_y = (bbox[1] + bbox[3]) / 2 / img_h
                rel_w = width / img_w
                rel_h = height / img_h
                
                print(f"  {j+1}. {cls_name}: {conf:.3f}")
                print(f"     Position: ({rel_x:.2f}, {rel_y:.2f}), Size: {rel_w:.2f}x{rel_h:.2f}")
                print(f"     Aspect: {aspect_ratio:.2f}")
                
                # Flag suspicious detections that might be faces
                suspicious_flags = []
                
                if 0.5 <= aspect_ratio <= 2.0:
                    suspicious_flags.append("face-like aspect")
                if rel_y < 0.7 and 0.2 <= rel_x <= 0.8:
                    suspicious_flags.append("face region")
                if rel_w > 0.15 and rel_h > 0.15:
                    suspicious_flags.append("large object")
                if conf > 0.2:
                    suspicious_flags.append("medium+ confidence")
                
                if suspicious_flags:
                    print(f"     🚨 SUSPICIOUS: {', '.join(suspicious_flags)}")
                    print(f"     ⚠️ This might be a face detection!")
                    total_face_detections += 1
                else:
                    print(f"     ✅ Likely valid plastic")
                    total_valid_plastic += 1
            
            print()
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Summary
    print("📊 FACE FILTERING SUMMARY")
    print("=" * 30)
    print(f"Suspicious face detections: {total_face_detections}")
    print(f"Valid plastic detections: {total_valid_plastic}")
    
    if total_face_detections == 0:
        print("✅ SUCCESS: Zero face detections!")
    else:
        print("❌ FAILURE: Still detecting faces as plastic!")

def test_plastic_object_variety():
    """Test detection of various plastic objects"""
    print("\n🔍 Testing Plastic Object Variety")
    print("=" * 40)
    
    model_path = "models/ocean_waste_model_m2/weights/best.pt"
    
    # Test different confidence levels for plastic detection
    confidence_levels = [0.1, 0.15, 0.2, 0.25]
    
    for conf in confidence_levels:
        print(f"\n🎯 Testing plastic detection at confidence {conf}")
        print("-" * 35)
        
        detector = PlasticDetector(
            model_path=model_path,
            device='cpu',
            conf_threshold=conf,
            debug_mode=False
        )
        
        # Find test image
        test_image = None
        for directory in [".", "../"]:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if file.lower().endswith('.jpg') and 'debug' in file.lower():
                        test_image = os.path.join(directory, file)
                        break
            if test_image:
                break
        
        if not test_image:
            continue
        
        try:
            image = cv2.imread(test_image)
            if image is None:
                continue
            
            detections, detection_info = detector.detect_objects(image)
            
            # Categorize detections
            bottles = [d for d in detection_info if 'bottle' in d['class_name'].lower()]
            general_plastic = [d for d in detection_info if 'bottle' not in d['class_name'].lower()]
            
            print(f"  Total detections: {len(detection_info)}")
            print(f"  Plastic bottles: {len(bottles)}")
            print(f"  General plastic: {len(general_plastic)}")
            
            # Show variety
            if detection_info:
                class_counts = {}
                for det in detection_info:
                    cls = det['class_name']
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                
                print(f"  Class variety: {list(class_counts.keys())}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")

def main():
    """Main test function"""
    print("🌊 Smart Marine Project - Comprehensive Filtering Test")
    print("=" * 65)
    
    # Test 1: Zero tolerance face filtering
    test_zero_tolerance_face_filtering()
    
    # Test 2: Plastic object variety
    test_plastic_object_variety()
    
    print("\n🏁 Comprehensive Test Complete!")
    print("\n🎯 IMPLEMENTED IMPROVEMENTS:")
    print("   🚫 ZERO TOLERANCE face/skin filtering")
    print("   🔍 Aggressive face detection (2+ indicators = reject)")
    print("   🎯 Better plastic object detection (all types)")
    print("   ⚡ Lower confidence thresholds for plastic classes")
    print("   📏 Improved shape recognition for plastic items")

if __name__ == "__main__":
    main()
