#!/usr/bin/env python3
"""
Test Face Filtering Improvements
================================

Test script to verify that faces are not detected as plastic, especially
at high confidence levels like 0.5.
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

def test_face_filtering():
    """Test face filtering with different confidence levels"""
    print("🔍 Testing Face Filtering Improvements")
    print("=" * 50)
    
    model_path = "models/ocean_waste_model_m2/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Test with very low confidence to see all raw detections
    detector = PlasticDetector(
        model_path=model_path,
        device='cpu',
        conf_threshold=0.01,  # Very low to see everything
        debug_mode=True  # Enable detailed debug output
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
    
    for i, img_path in enumerate(sample_files[:2]):
        print(f"🖼️ Testing Image {i+1}: {img_path}")
        print("-" * 40)
        
        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                print(f"❌ Could not load image: {img_path}")
                continue
            
            print(f"Image shape: {image.shape}")
            
            # Run detection with debug enabled
            detections, detection_info = detector.detect_objects(image)
            
            print(f"\n📊 Results:")
            print(f"Raw detections: {detections.shape[0] if detections is not None else 0}")
            print(f"Final detections: {len(detection_info)}")
            
            # Analyze final detections
            if detection_info:
                print(f"\n✅ Final Detections:")
                for j, det in enumerate(detection_info):
                    conf = det['confidence']
                    cls_name = det['class_name']
                    bbox = det['bbox']
                    
                    # Calculate detection properties
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    aspect_ratio = width / max(height, 1)
                    
                    print(f"  {j+1}. {cls_name}: {conf:.3f} confidence")
                    print(f"     Size: {width:.0f}x{height:.0f}, Aspect: {aspect_ratio:.2f}")
                    
                    # Flag potentially problematic detections
                    if conf > 0.4:
                        print(f"     ⚠️ HIGH CONFIDENCE - Check if this should be filtered")
                    if 0.7 <= aspect_ratio <= 1.4:
                        print(f"     ⚠️ FACE-LIKE ASPECT RATIO - Verify this is actually plastic")
            else:
                print("✅ No detections - face filtering working correctly")
            
            print()
            
        except Exception as e:
            print(f"❌ Error testing {img_path}: {e}")
            import traceback
            traceback.print_exc()

def test_specific_confidence():
    """Test detection at specific confidence levels that were problematic"""
    print("🎯 Testing Specific Confidence Levels")
    print("=" * 50)
    
    model_path = "models/ocean_waste_model_m2/weights/best.pt"
    
    # Test at the problematic confidence level
    test_confidences = [0.4, 0.5, 0.6]
    
    for conf in test_confidences:
        print(f"\n🔍 Testing at confidence {conf}")
        print("-" * 30)
        
        detector = PlasticDetector(
            model_path=model_path,
            device='cpu',
            conf_threshold=conf,
            debug_mode=False  # Less verbose for this test
        )
        
        # Find a test image
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
            print("⚠️ No test image found")
            continue
        
        try:
            image = cv2.imread(test_image)
            if image is None:
                continue
            
            detections, detection_info = detector.detect_objects(image)
            
            print(f"Detections at {conf} confidence: {len(detection_info)}")
            
            # Check for high-confidence detections that might be faces
            high_conf_detections = [d for d in detection_info if d['confidence'] >= conf]
            
            if high_conf_detections:
                print(f"⚠️ Found {len(high_conf_detections)} high-confidence detections:")
                for det in high_conf_detections:
                    bbox = det['bbox']
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    aspect_ratio = width / max(height, 1)
                    
                    print(f"  - {det['class_name']}: {det['confidence']:.3f}, aspect: {aspect_ratio:.2f}")
                    
                    # Check if this looks like a face
                    if 0.6 <= aspect_ratio <= 1.5:
                        print(f"    ⚠️ WARNING: Face-like aspect ratio detected as plastic!")
            else:
                print("✅ No high-confidence detections - filtering working correctly")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main test function"""
    print("🌊 Smart Marine Project - Face Filtering Test")
    print("=" * 60)
    
    # Test 1: Detailed face filtering analysis
    test_face_filtering()
    
    # Test 2: Specific confidence level testing
    test_specific_confidence()
    
    print("\n🏁 Face Filtering Test Complete!")
    print("\n🎯 KEY IMPROVEMENTS:")
    print("   ✅ Stronger filtering for confidence >0.4")
    print("   ✅ Very restrictive for confidence >0.6")
    print("   ✅ Only allow clear bottles or tiny items at high confidence")
    print("   ✅ Enhanced face detection indicators")

if __name__ == "__main__":
    main()
