#!/usr/bin/env python3
"""
Custom detection script for simplified plastic detection.
Only detects 2 classes: 'plastic' and 'plastic bottles'
"""

import os
import sys
import argparse
from pathlib import Path

# Add yolov5 to path
sys.path.append('yolov5')

import torch
import cv2
import numpy as np
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import check_img_size, non_max_suppression
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator, colors

def load_model(weights, device):
    """Load YOLOv5 model"""
    model = attempt_load(weights)
    model.to(device)
    return model

def detect_objects(model, img, device, conf_thres=0.3, iou_thres=0.45, max_det=1000):
    """Run object detection on image"""
    # Prepare image
    img_size = 640
    img = cv2.resize(img, (img_size, img_size))
    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        pred = model(img_tensor)[0]
    
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)
    
    return pred[0] if len(pred) > 0 else None

def simplify_labels(detections, img_shape):
    """Simplify detections to only 2 classes: plastic and plastic bottles"""
    if detections is None:
        return []
    
    simplified_detections = []
    
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        
        # Map all plastic-related classes to simplified labels
        if cls in [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18]:  # All plastic types
            # Check if it's specifically a bottle
            if cls in [8, 11, 14, 17]:  # Plastic bottle classes
                simplified_cls = 1  # "plastic bottle"
            else:
                simplified_cls = 0  # "plastic"
            
            simplified_detections.append([x1, y1, x2, y2, conf, simplified_cls])
    
    return simplified_detections

def draw_detections(img, detections, class_names, line_thickness=1):
    """Draw detections with custom labels and thinner lines"""
    annotator = Annotator(img, line_width=line_thickness, font_size=12)
    
    if detections is not None and len(detections) > 0:
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            label = f'{class_names[int(cls)]} {conf:.2f}'
            annotator.box_label([x1, y1, x2, y2], label, color=colors(int(cls), True))
    
    return annotator.result()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train/ocean_waste_model_m2/weights/best.pt', help='model weights')
    parser.add_argument('--source', type=str, default='test/images', help='source images')
    parser.add_argument('--conf', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--line-thickness', type=int, default=1, help='line thickness')
    parser.add_argument('--name', type=str, default='custom_plastic_detection', help='save results to project/name')
    args = parser.parse_args()
    
    # Setup
    device = select_device('')
    model = load_model(args.weights, device)
    
    # Simplified class names - only 2 classes
    class_names = ['plastic', 'plastic bottle']
    
    # Create output directory
    save_dir = Path(f'runs/detect/{args.name}')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images
    source_path = Path(args.source)
    if source_path.is_file():
        image_paths = [source_path]
    else:
        image_paths = list(source_path.glob('*.jpg')) + list(source_path.glob('*.jpeg')) + list(source_path.glob('*.png'))
    
    print(f'Processing {len(image_paths)} images...')
    
    for i, img_path in enumerate(image_paths):
        print(f'Processing {i+1}/{len(image_paths)}: {img_path.name}')
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Detect objects
        detections = detect_objects(model, img, device, args.conf)
        
        # Simplify labels
        simplified_detections = simplify_labels(detections, img.shape)
        
        # Draw detections
        result_img = draw_detections(img, simplified_detections, class_names, args.line_thickness)
        
        # Save result
        save_path = save_dir / img_path.name
        cv2.imwrite(str(save_path), result_img)
    
    print(f'Results saved to {save_dir}')

if __name__ == '__main__':
    main()
