#!/usr/bin/env python3
"""
Debug Plastic Detection System
==============================

Diagnostic script to test and debug plastic detection issues.
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

def test_model_loading():
    """Test if the model loads correctly"""
    print("🔍 Testing Model Loading...")
    print("=" * 50)
    
    model_path = "models/ocean_waste_model_m2/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    try:
        # Test with very low confidence to see all detections
        detector = PlasticDetector(
            model_path=model_path,
            device='cpu',  # Force CPU usage
            conf_threshold=0.01,  # Very low threshold
            debug_mode=True
        )
        print("✅ Model loaded successfully!")
        
        # Check model details
        print(f"📊 Model device: {detector.device}")
        print(f"📊 Model path: {detector.model_path}")
        print(f"📊 Class names: {detector.class_names}")
        print(f"📊 Confidence threshold: {detector.conf_threshold}")
        
        return detector
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sample_images(detector):
    """Test detection on sample images"""
    print("\n🖼️ Testing Sample Images...")
    print("=" * 50)
    
    # Look for sample images
    sample_dirs = [
        ".",
        "test",
        "data",
        "../"
    ]
    
    sample_files = []
    for directory in sample_dirs:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    sample_files.append(os.path.join(directory, file))
    
    if not sample_files:
        print("⚠️ No sample images found. Creating a test image...")
        # Create a simple test image
        test_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.rectangle(test_img, (100, 100), (200, 300), (0, 255, 0), -1)  # Green rectangle
        cv2.putText(test_img, "TEST PLASTIC BOTTLE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite("test_image.jpg", test_img)
        sample_files = ["test_image.jpg"]
    
    # Test first few images
    for i, img_path in enumerate(sample_files[:3]):
        print(f"\n📸 Testing image {i+1}: {img_path}")
        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                print(f"❌ Could not load image: {img_path}")
                continue
            
            print(f"   Image shape: {image.shape}")
            
            # Run detection with different confidence levels
            for conf in [0.01, 0.1, 0.3, 0.5]:
                detector.conf_threshold = conf
                print(f"\n   🎯 Testing with confidence {conf}")
                
                detections, detection_info = detector.detect_objects(image)
                print(f"   📊 Raw detections shape: {detections.shape if detections is not None else 'None'}")
                print(f"   📊 Processed detections: {len(detection_info)}")
                
                if detection_info:
                    for j, det in enumerate(detection_info):
                        print(f"      Detection {j+1}: {det['class_name']} (conf: {det['confidence']:.3f})")
                
        except Exception as e:
            print(f"❌ Error testing {img_path}: {e}")
            import traceback
            traceback.print_exc()

def test_model_classes(detector):
    """Test what classes the model can detect"""
    print("\n🏷️ Testing Model Classes...")
    print("=" * 50)
    
    try:
        # Try to inspect model structure
        model = detector.model
        print(f"📊 Model type: {type(model)}")
        
        # Try to get class names from model
        if hasattr(model, 'names'):
            print(f"📊 Model class names: {model.names}")
        elif hasattr(model, 'module') and hasattr(model.module, 'names'):
            print(f"📊 Model class names: {model.module.names}")
        else:
            print("⚠️ Could not extract class names from model")
        
        # Check number of classes
        if hasattr(model, 'nc'):
            print(f"📊 Number of classes: {model.nc}")
        elif hasattr(model, 'module') and hasattr(model.module, 'nc'):
            print(f"📊 Number of classes: {model.module.nc}")
        
    except Exception as e:
        print(f"❌ Error inspecting model: {e}")

def main():
    """Main diagnostic function"""
    print("🌊 Smart Marine Project - Detection Diagnostics")
    print("=" * 60)
    
    # Test 1: Model Loading
    detector = test_model_loading()
    if not detector:
        return
    
    # Test 2: Model Classes
    test_model_classes(detector)
    
    # Test 3: Sample Images
    test_sample_images(detector)
    
    print("\n" + "=" * 60)
    print("🏁 Diagnostic Complete!")
    print("\n💡 Recommendations:")
    print("   1. If no detections found, try lowering confidence threshold")
    print("   2. Check if model was trained on the right classes")
    print("   3. Ensure input images contain visible plastic objects")
    print("   4. Try different lighting conditions and angles")

if __name__ == "__main__":
    main()
