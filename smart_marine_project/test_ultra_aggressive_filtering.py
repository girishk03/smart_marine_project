#!/usr/bin/env python3
"""
Test Ultra-Aggressive Human Filtering
====================================

Test to verify absolutely NO human detection as plastic.
"""

import os
import sys
import cv2
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from plastic_detector import PlasticDetector

def test_ultra_aggressive_filtering():
    """Test that absolutely no humans are detected as plastic"""
    print("🚫 Testing Ultra-Aggressive Human Filtering")
    print("=" * 50)
    
    model_path = "models/ocean_waste_model_m2/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Test with very low confidence to catch everything
    detector = PlasticDetector(
        model_path=model_path,
        device='cpu',
        conf_threshold=0.05,  # Very low
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
    
    print(f"📸 Testing ultra-aggressive filtering on {len(sample_files[:2])} images")
    print()
    
    total_detections = 0
    human_detections = 0
    
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
            
            # Check each detection
            for det in detection_info:
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
                
                print(f"  Detection: {cls_name} (conf: {conf:.3f})")
                print(f"    Position: ({rel_x:.2f}, {rel_y:.2f})")
                print(f"    Size: {rel_w:.2f}x{rel_h:.2f}")
                print(f"    Aspect: {aspect_ratio:.2f}")
                
                # Check if this could be human
                human_flags = []
                if 0.3 <= aspect_ratio <= 3.0:
                    human_flags.append("human aspect")
                if rel_w > 0.15 or rel_h > 0.15:
                    human_flags.append("large size")
                if rel_y < 0.8 and 0.3 <= rel_x <= 0.7:
                    human_flags.append("center region")
                if conf > 0.2:
                    human_flags.append("medium+ confidence")
                
                if human_flags:
                    print(f"    🚨 POTENTIAL HUMAN: {', '.join(human_flags)}")
                    human_detections += 1
                else:
                    print(f"    ✅ Likely genuine plastic")
            
            print()
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Summary
    print("📊 ULTRA-AGGRESSIVE FILTERING SUMMARY")
    print("=" * 40)
    print(f"Total detections: {total_detections}")
    print(f"Potential human detections: {human_detections}")
    
    if human_detections == 0:
        print("✅ PERFECT: Zero human detections!")
    else:
        print(f"❌ FAILURE: {human_detections} potential human detections found!")
        print("   Need even more aggressive filtering!")

def main():
    """Main test function"""
    print("🌊 Smart Marine Project - Ultra-Aggressive Filtering Test")
    print("=" * 65)
    
    test_ultra_aggressive_filtering()
    
    print("\n🎯 ULTRA-AGGRESSIVE IMPROVEMENTS:")
    print("   🚫 1+ human indicator = automatic rejection")
    print("   🎯 Center region detections = 3 indicators")
    print("   ⚡ Any confidence >0.2 = 2 indicators")
    print("   📏 Large objects = 3 indicators")
    print("   🚨 Human proportions = 2 indicators")
    print("   ❌ NO EXCEPTIONS for any human detection")

if __name__ == "__main__":
    main()
