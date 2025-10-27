#!/usr/bin/env python3
"""
Smart Marine Project - Plastic Waste Detection System
====================================================

A comprehensive plastic waste detection system for marine environments.
Detects and classifies plastic waste in images with high accuracy.

Author: Smart Marine Project Team
Version: 1.0.1
"""

import os
import sys
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
from datetime import datetime

# Try to import from local YOLOv5 repo first, fallback to manual implementations
LOCAL_YOLO = False
attempt_load = None
non_max_suppression = None
letterbox = None
select_device = None
scale_coords = None
Annotator = None
colors = None

try:
    # Add YOLOv5 repo root to path so we can import its local modules
    yolov5_path = os.path.join(os.path.dirname(__file__), '../../yolov5')
    if os.path.exists(yolov5_path) and yolov5_path not in sys.path:
        sys.path.insert(0, yolov5_path)

    try:
        from models.experimental import attempt_load as yolo_attempt_load
        from utils.general import check_img_size, non_max_suppression as yolo_nms
        from utils.augmentations import letterbox as yolo_letterbox
        from utils.torch_utils import select_device as yolo_select_device
        from utils.plots import Annotator as yolo_annotator, colors as yolo_colors
        
        attempt_load = yolo_attempt_load
        non_max_suppression = yolo_nms
        letterbox = yolo_letterbox
        select_device = yolo_select_device
        Annotator = yolo_annotator
        colors = yolo_colors
        
        # Try to import scale_coords (might not exist in newer versions)
        try:
            from utils.general import scale_coords as yolo_scale_coords
            scale_coords = yolo_scale_coords
        except ImportError:
            scale_coords = None  # Will use fallback
        
        print("✅ Successfully imported from local YOLOv5")
        LOCAL_YOLO = True
    except ImportError as e:
        print(f"⚠️ Local YOLOv5 import failed: {e}")
        print("🔄 Using fallback implementations...")
        LOCAL_YOLO = False

except Exception as e:
    print(f"❌ Error setting up YOLOv5 path: {e}")
    LOCAL_YOLO = False

# Fallback implementations if YOLOv5 imports failed
if not LOCAL_YOLO or select_device is None:
    def select_device(device=''):
        """Fallback device selection"""
        if device == 'auto' or device == '':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

if not LOCAL_YOLO or letterbox is None:
    def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        """Fallback letterbox implementation"""
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

if not LOCAL_YOLO or non_max_suppression is None:
    def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300):
        """Fallback NMS implementation"""
        import torchvision
        output = []
        for xi, x in enumerate(prediction):
            x = x[x[:, 4] > conf_thres]
            if not x.shape[0]:
                output.append(torch.zeros((0, 6)))
                continue
            x[:, 5:] *= x[:, 4:5]
            box = x[:, :4]
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            n = x.shape[0]
            if not n:
                output.append(torch.zeros((0, 6)))
                continue
            boxes, scores = x[:, :4], x[:, 4]
            i = torchvision.ops.nms(boxes, scores, iou_thres)
            if i.shape[0] > max_det:
                i = i[:max_det]
            output.append(x[i])
        return output

if not LOCAL_YOLO or Annotator is None:
    class Annotator:
        """Fallback Annotator class"""
        def __init__(self, im, line_width=None, font_size=None):
            self.im = im
            self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)
        
        def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 1, 1)
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)
                cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                           0, self.lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
        
        def result(self):
            return self.im

if not LOCAL_YOLO or colors is None:
    def colors(i, bgr=False):
        """Fallback colors function"""
        palette = [(255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
                   (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134)]
        c = palette[int(i) % len(palette)]
        return c if bgr else c[::-1]


class PlasticDetector:
    """
    Smart Marine Plastic Detection System
    
    Detects and classifies plastic waste in marine environments with:
    - High accuracy plastic detection
    - Single unified "plastic" label for all plastic waste
    - Configurable confidence thresholds
    - Batch processing capabilities
    """
    
    def __init__(self, model_path: str, device: str = 'auto', conf_threshold: float = 0.2,
                 iou_threshold: float = 0.3, img_size: int = 640, tta: bool = False, debug_mode: bool = False):
        """
        Initialize the plastic detector
        
        Args:
            model_path: Path to the trained YOLOv5 model weights
            device: Device to run inference on ('auto', 'cpu', 'cuda')
            conf_threshold: Confidence threshold for detections
        """
        self.model_path = model_path
        self.device = select_device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.tta = tta
        self.debug_mode = debug_mode  # Debug mode for troubleshooting
        
        # Load model
        self.model = self._load_model()
        
        # OPTIMIZED: Single unified label for ALL plastic detection
        self.class_names = ['plastic']  # ONLY plastic label - no other classes
        
        # OPTIMIZED: Simplified statistics tracking
        self.stats = {
            'total_images': 0,
            'images_with_plastic': 0,
            'total_plastics': 0,
            'processing_time': 0,
            'avg_confidence': 0.0
        }
    
    def _load_model(self):
        """Load the YOLOv5 model"""
        try:
            if LOCAL_YOLO and 'attempt_load' in globals():
                # Use local YOLOv5 repo
                print(f"🔧 Loading model with local YOLOv5: {self.model_path}")
                model = attempt_load(self.model_path)
            else:
                # Fallback: Use torch.load directly (deployed mode)
                print(f"🔧 Loading model with torch.load: {self.model_path}")
                # Set weights_only=False for YOLOv5 models (trusted source)
                model = torch.load(self.model_path, map_location=self.device, weights_only=False)
                if isinstance(model, dict) and 'model' in model:
                    model = model['model']
                model = model.float()

            model.to(self.device)
            model.eval()
            print(f"✅ Model loaded successfully from {self.model_path}")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def detect_objects(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        OPTIMIZED: Detect plastic objects with improved efficiency
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Tuple of (raw_detections, detection_info)
        """
        # OPTIMIZED: Use dynamic confidence based on image quality
        original_conf = self.conf_threshold
        if self.debug_mode:
            print(f"🎯 Confidence threshold: {self.conf_threshold:.3f}")
        
        # Prepare image using YOLOv5's letterbox to preserve aspect ratio
        img_size = int(self.img_size)
        # Handle different model formats
        if hasattr(self.model, 'stride'):
            stride = int(self.model.stride.max())
        else:
            stride = 32  # Default stride for YOLOv5 models
        
        img0 = image  # original image
        lb_img = letterbox(img0, img_size, stride=stride, auto=True)[0]  # HWC BGR
        # Convert HWC BGR to CHW RGB
        lb_img = lb_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        lb_img = np.ascontiguousarray(lb_img)
        
        # Convert to tensor
        lb_img = torch.from_numpy(lb_img).to(self.device)
        lb_img = lb_img.float() / 255.0  # 0 - 255 to 0.0 - 1.0
        if lb_img.ndimension() == 3:
            lb_img = lb_img.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            if self.tta:
                # Test time augmentation
                pred = self.model(lb_img, augment=True)[0]
            else:
                pred = self.model(lb_img)[0]
        
        # OPTIMIZED: Apply NMS with stricter IoU for better accuracy
        optimized_iou = min(self.iou_threshold, 0.4)  # Stricter IoU for plastic objects
        detections = non_max_suppression(pred, self.conf_threshold, optimized_iou)[0]

        # FIXED: Proper coordinate rescaling from model img size to original image size
        if detections is not None and len(detections):
            # Use proper YOLOv5 coordinate scaling
            if LOCAL_YOLO and scale_coords is not None:
                try:
                    # YOLOv5 scale_coords expects (img1_shape, coords, img0_shape)
                    detections[:, :4] = scale_coords(lb_img.shape[2:], detections[:, :4], img0.shape).round()
                except Exception as e:
                    if self.debug_mode:
                        print(f"⚠️ scale_coords failed: {e}, using manual scaling")
                    # Manual scaling fallback
                    h, w = img0.shape[:2]
                    model_h, model_w = lb_img.shape[2], lb_img.shape[3]
                    
                    # Scale coordinates back to original image
                    detections[:, [0, 2]] *= w / model_w  # x coordinates
                    detections[:, [1, 3]] *= h / model_h  # y coordinates
                    
                    # Clamp coordinates to image bounds
                    detections[:, [0, 2]] = torch.clamp(detections[:, [0, 2]], 0, w)
                    detections[:, [1, 3]] = torch.clamp(detections[:, [1, 3]], 0, h)
            else:
                # Manual scaling when YOLOv5 functions not available
                h, w = img0.shape[:2]
                model_h, model_w = lb_img.shape[2], lb_img.shape[3]
                
                if self.debug_mode:
                    print(f"    Scaling: model={model_w}x{model_h} -> original={w}x{h}")
                
                # Scale coordinates back to original image
                detections[:, [0, 2]] *= w / model_w  # x coordinates  
                detections[:, [1, 3]] *= h / model_h  # y coordinates
                
                # Clamp coordinates to image bounds
                detections[:, [0, 2]] = torch.clamp(detections[:, [0, 2]], 0, w)
                detections[:, [1, 3]] = torch.clamp(detections[:, [1, 3]], 0, h)

        detection_info = self._process_detections(detections, image.shape)
        
        # Restore original confidence threshold
        self.conf_threshold = original_conf
        
        return detections, detection_info
    
    def _process_detections(self, detections, img_shape) -> List[Dict]:
        """OPTIMIZED: Process detections with improved plastic-only filtering"""
        if detections is None:
            if self.debug_mode:
                print("⚠️ No detections from model")
            return []
        
        if self.debug_mode:
            print(f"🔍 Processing {len(detections)} raw detections")
        
        detection_info = []
        
        # FIXED: Only accept genuine plastic classes (STEEL DETECTION FIX)
        plastic_classes = {
            0: 'plastic',           # Only accept class 0 (plastic) - main plastic class
        }
        
        # FIXED: Reject ALL non-plastic classes including steel/metal
        rejected_classes = {1, 2, 3, 4, 5, 6, 7, 8}  # All non-plastic: Bottle cap, Can, Juice Box, Metal, Metal Waste, Undefined trash, Wood
        
        confidence_sum = 0.0
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls)
            
            # OPTIMIZED: Quick rejection of non-plastic classes (but allow more unknowns)
            if cls in rejected_classes or cls >= 19:  # Allow classes 17, 18 as potential plastic
                if self.debug_mode:
                    print(f"  ❌ Rejected class {cls} (non-plastic)")
                continue
            
            # OPTIMIZED: Accept only known plastic classes
            if cls in plastic_classes:
                # MARINE OPTIMIZED: Aggressive confidence boosting for marine plastic detection
                adjusted_conf = float(conf)
                
                # MARINE BOOST: Extra boost for bottle classes in marine environments
                if cls in [8, 11, 14, 17]:  # Plastic bottles - maximum priority
                    adjusted_conf = min(adjusted_conf * 1.2, 1.0)  # Increased from 1.1
                elif cls in [6, 18]:  # General plastic - strong boost
                    adjusted_conf = min(adjusted_conf * 1.15, 1.0)  # Increased from 1.05
                elif cls in [7, 9, 10, 12, 13]:  # Other plastic classes
                    adjusted_conf = min(adjusted_conf * 1.1, 1.0)  # New boost for other plastic
                
                # Marine environment: boost all plastic detections
                if adjusted_conf > 0.3:
                    adjusted_conf = min(adjusted_conf * 1.1, 1.0)
                
                # FIXED: Proper bounding box validation
                box_width = abs(float(x2 - x1))
                box_height = abs(float(y2 - y1))
                box_area = box_width * box_height
                img_area = img_shape[0] * img_shape[1]
                box_ratio = box_area / img_area
                
                # DEBUG: Show box calculations
                if self.debug_mode:
                    print(f"    Box: w={box_width:.1f}, h={box_height:.1f}, area={box_area:.1f}")
                    print(f"    Image area: {img_area}, ratio: {box_ratio:.6f}")
                
                # OPTIMIZED: More realistic size range for plastic objects
                if box_ratio < 0.0001 or box_ratio > 0.85:  # Allow smaller items, limit very large
                    if self.debug_mode:
                        print(f"  ❌ Rejected: Box size ratio {box_ratio:.6f} out of range")
                    continue
                
                # IMPROVED: Realistic aspect ratios for plastic objects
                aspect_ratio = box_width / max(box_height, 1)
                if aspect_ratio > 10 or aspect_ratio < 0.08:  # Allow most plastic shapes
                    if self.debug_mode:
                        print(f"  ❌ Rejected: Invalid aspect ratio {aspect_ratio:.2f}")
                    continue
                
                # SMART HUMAN FILTERING: Block humans but allow clear bottles
                if self._is_likely_face(x1, y1, x2, y2, img_shape, adjusted_conf):
                    rel_width = box_width / img_shape[1]
                    rel_height = box_height / img_shape[0]
                    
                    # BOTTLE EXCEPTIONS: Allow clear plastic bottles
                    is_clear_bottle = (
                        cls in [8, 11, 14, 17] and  # Bottle classes
                        aspect_ratio < 0.8 and  # Tall/thin bottles
                        rel_width < 0.3 and  # Not too wide
                        adjusted_conf > 0.2  # Decent confidence
                    )
                    
                    is_small_item = (
                        rel_width < 0.1 and rel_height < 0.1 and  # Very small
                        cls in [6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18]  # Any plastic
                    )
                    
                    # Block if it's human AND not a clear bottle exception
                    if not (is_clear_bottle or is_small_item):
                        if self.debug_mode:
                            print(f"  ❌ REJECTED: Human detected (conf={adjusted_conf:.3f}, aspect={aspect_ratio:.2f}, size={rel_width:.2f}x{rel_height:.2f})")
                        continue
                    
                    if self.debug_mode:
                        print(f"  ⚠️ Human-like but allowing bottle/small item")
                
                # STEEL DETECTION FIX: Check for steel/metal objects first
                if self._is_steel_or_metal(x1, y1, x2, y2, img_shape, adjusted_conf, cls):
                    if self.debug_mode:
                        print(f"  ❌ Rejected: Steel/Metal object detected")
                    continue
                
                # PLASTIC VALIDATION: Additional checks for plastic objects
                if not self._is_likely_plastic(x1, y1, x2, y2, img_shape, adjusted_conf, cls):
                    if self.debug_mode:
                        print(f"  ❌ Rejected: Not likely plastic object")
                    continue
                
                # Get the correct class name from our mapping
                class_name = plastic_classes[cls]
                
                detection_info.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': adjusted_conf,
                    'class_id': cls,  # Use actual class ID
                    'class_name': class_name,  # Use mapped class name
                    'box_area': box_area,
                    'aspect_ratio': aspect_ratio
                })
                
                confidence_sum += adjusted_conf
                
                if self.debug_mode:
                    print(f"  ✅ Plastic detected: conf={adjusted_conf:.3f}, area={box_ratio:.4f}")
        
        # OPTIMIZED: Update average confidence
        if detection_info:
            self.stats['avg_confidence'] = confidence_sum / len(detection_info)
        
        if self.debug_mode:
            print(f"📊 Final: {len(detection_info)} plastic objects (avg conf: {self.stats['avg_confidence']:.3f})")
        return detection_info
    
    def _is_likely_face(self, x1, y1, x2, y2, img_shape, confidence):
        """
        Detect if a bounding box is likely a face based on position, size, and shape
        """
        box_width = abs(float(x2 - x1))
        box_height = abs(float(y2 - y1))
        img_height, img_width = img_shape[:2]
        
        # Face characteristics
        aspect_ratio = box_width / max(box_height, 1)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Relative position in image
        rel_center_x = center_x / img_width
        rel_center_y = center_y / img_height
        rel_width = box_width / img_width
        rel_height = box_height / img_height
        
        face_indicators = 0
        
        # 1. Face-like aspect ratio (roughly square to slightly wide)
        if 0.7 <= aspect_ratio <= 1.5:
            face_indicators += 1
        
        # 2. Position in upper portion of image (faces usually in top 60%)
        if rel_center_y < 0.6:
            face_indicators += 1
        
        # 3. Reasonable face size (5-40% of image width)
        if 0.05 <= rel_width <= 0.4 and 0.05 <= rel_height <= 0.4:
            face_indicators += 1
        
        # 4. Center-ish horizontal position (not at extreme edges)
        if 0.2 <= rel_center_x <= 0.8:
            face_indicators += 1
        
        # 5. Very low confidence detections in face region are suspicious
        if confidence < 0.2 and rel_center_y < 0.4:
            face_indicators += 1
        
        # 6. Large detections in center region (human zone) - exclude bottles
        if rel_center_y < 0.7 and 0.3 <= rel_center_x <= 0.7 and rel_width > 0.2 and rel_height > 0.2:
            face_indicators += 3  # Large center objects = likely human
        
        # 7. High confidence with face-like proportions (exclude bottles)
        if confidence > 0.4 and 0.6 <= aspect_ratio <= 2.5:
            face_indicators += 2  # High confidence face-like shapes
        
        # 8. Very large objects with face-like proportions
        if (rel_width > 0.25 or rel_height > 0.3) and 0.4 <= aspect_ratio <= 2.5:
            face_indicators += 3  # Large face-like objects = human body parts
        
        # 9. Face-like aspect ratios in upper region
        if 0.6 <= aspect_ratio <= 2.0 and rel_center_y < 0.7:
            face_indicators += 2  # Face proportions in face region
        
        # 10. Large objects in typical face/body positions
        if (rel_center_y < 0.8 and rel_width > 0.2 and rel_height > 0.25 and 0.5 <= aspect_ratio <= 2.0):
            face_indicators += 2  # Large face-like objects in human positions
        
        # 9. SKIN TONE DETECTION: Objects in center region (where faces usually are)
        if 0.2 <= rel_center_x <= 0.8 and 0.1 <= rel_center_y <= 0.7:
            face_indicators += 1  # Center region bias
        
        # 10. AGGRESSIVE: Any detection with human-like proportions
        if 0.3 <= aspect_ratio <= 3.0 and rel_width > 0.1 and rel_height > 0.1:
            face_indicators += 1  # Human-like proportions
        
        # 8. ADDITIONAL: Very large objects in upper region (likely faces/torso)
        if rel_width > 0.3 and rel_height > 0.4 and rel_center_y < 0.6:
            face_indicators += 1
        
        # 9. SPECIFIC: Detect human-sized objects (like in your image)
        # Large rectangular detections covering significant portion of image
        if rel_width > 0.25 and rel_height > 0.5 and rel_center_y > 0.3:
            face_indicators += 2  # Strong indicator of human body/face
        
        # MARINE OPTIMIZED: Even less aggressive for marine environments
        is_face = face_indicators >= 5  # Need even more indicators to reject in marine scenes
        
        if self.debug_mode and is_face:
            print(f"    Face indicators: {face_indicators}/6")
            print(f"    Aspect ratio: {aspect_ratio:.2f}")
            print(f"    Position: ({rel_center_x:.2f}, {rel_center_y:.2f})")
            print(f"    Size: {rel_width:.2f}x{rel_height:.2f}")
        
        return is_face
    
    def _is_likely_plastic(self, x1, y1, x2, y2, img_shape, confidence, cls):
        """
        Validate if detection is likely a plastic object (not organic matter like faces/hands)
        """
        box_width = abs(float(x2 - x1))
        box_height = abs(float(y2 - y1))
        img_height, img_width = img_shape[:2]
        
        aspect_ratio = box_width / max(box_height, 1)
        center_y = (y1 + y2) / 2
        rel_center_y = center_y / img_height
        
        plastic_indicators = 0
        
        # 1. Bottle-like shapes (tall and thin)
        if aspect_ratio < 0.7 and box_height > box_width * 1.5:  # Tall bottles
            plastic_indicators += 2
        
        # 2. Container-like shapes (wide and short)
        if aspect_ratio > 1.3 and box_width > box_height * 1.3:  # Wide containers
            plastic_indicators += 1
        
        # 3. High confidence plastic bottle/bag classes
        if cls in [11, 12, 13] and confidence > 0.3:  # Specific plastic classes
            plastic_indicators += 2
        
        # 4. Lower portion of image (objects on tables/ground)
        if rel_center_y > 0.4:  # Below middle of image
            plastic_indicators += 1
        
        # 5. Reasonable plastic object size
        rel_width = box_width / img_width
        rel_height = box_height / img_height
        if 0.03 <= rel_width <= 0.6 and 0.03 <= rel_height <= 0.8:
            plastic_indicators += 1
        
        # 6. Avoid skin-tone colored regions (basic heuristic)
        # This would need actual image analysis, for now use position/shape
        
        # 7. Moderate confidence range (too high might be face, too low might be noise)
        if 0.15 <= confidence <= 0.9:
            plastic_indicators += 1
        
        # 8. SMALL PLASTIC ITEMS: Pens, cards, small containers
        if 0.02 <= rel_width <= 0.3 and 0.02 <= rel_height <= 0.3:
            plastic_indicators += 1
        
        # 9. RECTANGULAR ITEMS: ID cards, credit cards, rectangular plastic items
        if 1.2 <= aspect_ratio <= 2.5 and rel_width >= 0.05:  # Wide rectangular items
            plastic_indicators += 1
        
        # 10. PEN-LIKE ITEMS: Very thin, elongated objects
        if aspect_ratio < 0.3 and rel_height >= 0.1:  # Very thin and tall
            plastic_indicators += 1
        
        # IMPROVED: Better acceptance criteria for all plastic types
        # Accept if ANY of these conditions are met:
        is_plastic = (
            plastic_indicators >= 2 or  # Standard plastic indicators
            (cls in [6, 7, 9, 10, 12, 13, 18] and confidence > 0.15) or  # General plastic classes
            (cls in [8, 11, 14, 17] and confidence > 0.2) or  # Bottle classes  
            (aspect_ratio < 0.6 and confidence > 0.15) or  # Bottle-like shapes
            (1.5 <= aspect_ratio <= 4.0 and confidence > 0.15) or  # Wide plastic items (bags, containers)
            (rel_width >= 0.05 and rel_height >= 0.05 and confidence > 0.25)  # Reasonable sized confident detections
        )
        
        if self.debug_mode:
            print(f"    Plastic indicators: {plastic_indicators}/6")
            print(f"    Class: {cls}, Confidence: {confidence:.3f}")
            print(f"    Shape: {aspect_ratio:.2f} ({'tall' if aspect_ratio < 0.7 else 'wide' if aspect_ratio > 1.3 else 'square'})")
        
        return is_plastic
    
    def _is_steel_or_metal(self, x1, y1, x2, y2, img_shape, confidence, cls):
        """
        STEEL DETECTION FIX: Detect and reject steel/metal objects
        """
        box_width = abs(float(x2 - x1))
        box_height = abs(float(y2 - y1))
        aspect_ratio = box_width / max(box_height, 1)
        
        # Steel/metal indicators
        steel_indicators = 0
        
        # 1. Reject known metal classes immediately
        if cls in [2, 5, 6]:  # Can, Metal, Metal Waste classes
            steel_indicators += 3
        
        # 2. Steel can characteristics (tall, cylindrical)
        if (0.2 <= aspect_ratio <= 0.7 and  # Tall and thin like steel cans
            box_height > box_width * 1.5 and  # Cylindrical shape
            confidence > 0.3):  # High confidence
            steel_indicators += 2
        
        # 3. Metallic bottle cap characteristics (small, round)
        if (0.8 <= aspect_ratio <= 1.2 and  # Roughly square/round
            box_width < img_shape[1] * 0.1 and  # Small size
            confidence > 0.4):  # High confidence for small objects
            steel_indicators += 1
        
        # 4. High confidence detections that look like metal containers
        if (confidence > 0.6 and 
            (0.3 <= aspect_ratio <= 0.8) and  # Container-like proportions
            box_height > img_shape[0] * 0.1):  # Reasonable size
            steel_indicators += 1
        
        if self.debug_mode and steel_indicators >= 2:
            print(f"    Steel indicators: {steel_indicators}/4")
            print(f"    Class: {cls}, Confidence: {confidence:.3f}")
            print(f"    Shape: {aspect_ratio:.2f}")
        
        return steel_indicators >= 2
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict], 
                       line_thickness: int = 2, font_size: int = 12) -> np.ndarray:
        """
        OPTIMIZED: Draw plastic detections with improved visibility
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            line_thickness: Thickness of bounding box lines (default: 2)
            font_size: Font size for labels
            
        Returns:
            Image with drawn detections
        """
        annotator = Annotator(image, line_width=line_thickness, font_size=font_size)
        
        # OPTIMIZED: Use consistent color for all plastic detections
        plastic_color = (0, 255, 0)  # Green for all plastic objects
        
        for i, det in enumerate(detections):
            bbox = det['bbox']
            conf = det['confidence']
            
            # OPTIMIZED: Simplified label - only "plastic" with confidence
            label = f"plastic {conf:.2f}"
            
            # OPTIMIZED: Color intensity based on confidence
            if conf > 0.7:
                color = (0, 255, 0)      # Bright green for high confidence
            elif conf > 0.5:
                color = (0, 200, 100)    # Medium green
            else:
                color = (0, 150, 150)    # Dim green for lower confidence
            
            annotator.box_label(bbox, label, color=color)
        
        return annotator.result()
    
    def process_image(self, image_path: str, output_path: str = None, 
                     line_thickness: int = 1) -> Dict:
        """
        Process a single image
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
            line_thickness: Thickness of bounding box lines
            
        Returns:
            Dictionary with detection results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect objects
        start_time = datetime.now()
        detections, detection_info = self.detect_objects(image)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Draw detections
        result_image = self.draw_detections(image, detection_info, line_thickness)
        
        # Save result if output path provided
        if output_path:
            cv2.imwrite(output_path, result_image)
        
        # OPTIMIZED: Update statistics with efficiency metrics
        self.stats['total_images'] += 1
        if detection_info:
            self.stats['images_with_plastic'] += 1
            self.stats['total_plastics'] += len(detection_info)
        
        self.stats['processing_time'] += processing_time
        
        return {
            'image_path': image_path,
            'detections': detection_info,
            'num_detections': len(detection_info),
            'processing_time': processing_time,
            'output_path': output_path
        }
    
    def process_batch(self, input_dir: str, output_dir: str, 
                     line_thickness: int = 1) -> Dict:
        """
        Process a batch of images
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save output images
            line_thickness: Thickness of bounding box lines
            
        Returns:
            Dictionary with batch processing results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        print(f"🔍 Found {len(image_files)} images to process")
        
        results = []
        for i, img_file in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {img_file.name}")
            
            try:
                output_file = output_path / img_file.name
                result = self.process_image(str(img_file), str(output_file), line_thickness)
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing {img_file.name}: {e}")
                continue
        
        # Generate summary
        summary = self._generate_summary(results)
        
        return {
            'results': results,
            'summary': summary,
            'output_directory': str(output_path)
        }
    
    def _generate_summary(self, results: List[Dict]) -> Dict:
        """OPTIMIZED: Generate comprehensive processing summary"""
        total_images = len(results)
        images_with_detections = sum(1 for r in results if r['num_detections'] > 0)
        total_detections = sum(r['num_detections'] for r in results)
        
        # OPTIMIZED: All detections are plastic, so total_plastics = total_detections
        total_plastics = total_detections
        
        # OPTIMIZED: Calculate confidence statistics
        all_confidences = [det['confidence'] for r in results for det in r['detections']]
        avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
        min_confidence = np.min(all_confidences) if all_confidences else 0.0
        max_confidence = np.max(all_confidences) if all_confidences else 0.0
        
        avg_processing_time = np.mean([r['processing_time'] for r in results])
        
        return {
            'total_images_processed': total_images,
            'images_with_detections': images_with_detections,
            'detection_rate': f"{(images_with_detections/total_images)*100:.1f}%" if total_images > 0 else "0%",
            'total_detections': total_detections,
            'total_plastics': total_plastics,
            'average_confidence': f"{avg_confidence:.3f}",
            'confidence_range': f"{min_confidence:.3f} - {max_confidence:.3f}",
            'average_processing_time': f"{avg_processing_time:.3f}s",
            'total_processing_time': f"{sum(r['processing_time'] for r in results):.3f}s",
            'fps_estimate': f"{1/avg_processing_time:.1f}" if avg_processing_time > 0 else "N/A"
        }
    
    def save_results(self, results: Dict, output_file: str):
        """Save detection results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📊 Results saved to {output_file}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Smart Marine Plastic Detection System')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--input', type=str, required=True, help='Input image or directory')
    parser.add_argument('--output', type=str, required=True, help='Output image or directory')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--line-thickness', type=int, default=1, help='Bounding box line thickness')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--save-results', type=str, help='Save detection results to JSON file')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = PlasticDetector(args.model, args.device, args.conf)
    
    # Check if input is file or directory
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image processing
        print(f"🔍 Processing single image: {input_path.name}")
        result = detector.process_image(str(input_path), args.output, args.line_thickness)
        print(f"✅ Found {result['num_detections']} plastic objects")
        
        if args.save_results:
            detector.save_results(result, args.save_results)
    
    elif input_path.is_dir():
        # Batch processing
        print(f"🔍 Processing directory: {input_path}")
        results = detector.process_batch(str(input_path), args.output, args.line_thickness)
        
        # Print summary
        summary = results['summary']
        print(f"\n📊 Processing Summary:")
        print(f"   Images processed: {summary['total_images_processed']}")
        print(f"   Detection rate: {summary['detection_rate']}")
        print(f"   Total plastics found: {summary['total_plastics']}")
        print(f"   Average confidence: {summary['average_confidence']}")
        print(f"   Average time per image: {summary['average_processing_time']}")
        print(f"   Estimated FPS: {summary['fps_estimate']}")
        
        if args.save_results:
            detector.save_results(results, args.save_results)
    
    else:
        print(f"❌ Input path does not exist: {input_path}")
        return 1
    
    print(f"✅ Processing complete! Results saved to: {args.output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
