import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import os
import sys
from datetime import datetime
import json
import tempfile
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict

# Vessel modules for autonomous navigation
try:
    import folium
    from streamlit_folium import st_folium
    from geopy.distance import geodesic
    import yaml
    MAPPING_AVAILABLE = True
except ImportError:
    MAPPING_AVAILABLE = False

# Import vessel modules
try:
    from vessel_modules import VesselSimulator, GPSNavigator, CollectionCounter, VesselCamera, VESSEL_MODULES_AVAILABLE
except ImportError:
    VESSEL_MODULES_AVAILABLE = False

# YOLOv5 implementation (model is YOLOv5 trained)

# Suppress YOLOv5 output globally
os.environ['YOLOV5_VERBOSE'] = 'False'
import logging
logging.getLogger('yolov5').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)

# Add YOLOv5 to path
yolov5_path = os.path.join(os.path.dirname(__file__), 'yolov5')
if yolov5_path not in sys.path:
    sys.path.insert(0, yolov5_path)

# Lazy import YOLOv5 components
YOLO_AVAILABLE = False
DetectMultiBackend = None
LoadImages = None
non_max_suppression = None
select_device = None
letterbox = None

def import_yolov5():
    """Import YOLOv5 components lazily"""
    global YOLO_AVAILABLE, DetectMultiBackend, LoadImages, non_max_suppression, select_device, letterbox
    if not YOLO_AVAILABLE:
        try:
            from models.common import DetectMultiBackend as _DetectMultiBackend
            from utils.dataloaders import LoadImages as _LoadImages
            from utils.general import check_img_size, non_max_suppression as _non_max_suppression
            from utils.torch_utils import select_device as _select_device
            from utils.augmentations import letterbox as _letterbox
            
            DetectMultiBackend = _DetectMultiBackend
            LoadImages = _LoadImages
            non_max_suppression = _non_max_suppression
            select_device = _select_device
            letterbox = _letterbox
            YOLO_AVAILABLE = True
        except ImportError as e:
            st.error(f"YOLOv5 import error: {e}")
            YOLO_AVAILABLE = False

# Import the advanced plastic detector (YOLOv8)
try:
    from plastic_detector_v8 import AdvancedPlasticDetector
    DETECTOR_AVAILABLE = True
    DETECTOR_VERSION = "YOLOv8"
except ImportError as e:
    print(f"Warning: YOLOv8 detector not available: {e}")
    # Fallback to YOLOv5
    try:
        from plastic_detector import PlasticDetector
        DETECTOR_AVAILABLE = True
        DETECTOR_VERSION = "YOLOv5"
    except ImportError as e2:
        print(f"Warning: Could not import any detector: {e2}")
        DETECTOR_AVAILABLE = False
        DETECTOR_VERSION = "None"

try:
    import av
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    WEBCAM_AVAILABLE = True
except ImportError:
    WEBCAM_AVAILABLE = False
    st.warning("Webcam features require: pip install streamlit-webrtc av")

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Scale coordinates from one image shape to another"""
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    return coords

# Model path
MODEL_PATH = 'yolov5m.pt'  # Original YOLOv5 - better for marine debris detection

def download_model():
    """Download YOLOv5m model if not present"""
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading YOLOv5m model (first time only)...")
        try:
            import urllib.request
            url = 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt'
            urllib.request.urlretrieve(url, MODEL_PATH)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            return False
    return True

# Use singleton pattern to ensure model loads only once
@st.cache_resource
def load_advanced_detector():
    """Load YOLOv8 Advanced Plastic Detector - singleton pattern"""
    try:
        if DETECTOR_VERSION == "YOLOv8":
            # Use YOLOv8 detector
            detector = AdvancedPlasticDetector(
                model_size='n',  # Start with nano for speed, can upgrade to 's', 'm', 'l', 'x'
                confidence_threshold=0.25,
                iou_threshold=0.45,
                device='auto'
            )
            return detector, f"✅ {DETECTOR_VERSION} Advanced Detector loaded successfully!"
        
        elif DETECTOR_VERSION == "YOLOv5":
            # Fallback to YOLOv5
            detector = PlasticDetector(
                model_path=MODEL_PATH if os.path.exists(MODEL_PATH) else None,
                confidence_threshold=0.25
            )
            return detector, f"✅ {DETECTOR_VERSION} Detector loaded successfully!"
        
        else:
            return None, "❌ No detector available"
            
    except Exception as e:
        return None, f"❌ Error loading {DETECTOR_VERSION} detector: {e}"

# Keep legacy function for backward compatibility
@st.cache_resource
def load_model():
    """Legacy YOLOv5 model loader - kept for compatibility"""
    # Auto-download model if not present
    if not download_model():
        return None, f"❌ Model download failed"
    
    if not os.path.exists(MODEL_PATH):
        return None, f"❌ Model not found: {MODEL_PATH}"

    try:
        # Import YOLOv5 components if not already imported
        import_yolov5()
        
        if not YOLO_AVAILABLE:
            return None, "❌ YOLOv5 components not available"
        
        device = select_device('cpu')
        model = DetectMultiBackend(MODEL_PATH, device=device, dnn=False, fp16=False)
        return model, "✅ Model loaded successfully!"
    except Exception as e:
        return None, f"❌ Error loading model: {e}"

@st.cache_resource
def load_webcam_model():
    """Load original YOLOv5 model for better webcam detection"""
    webcam_model_path = 'yolov5m.pt'
    
    # Auto-download model if not present
    if not download_model():
        return None, f"❌ Model download failed"
    
    if not os.path.exists(webcam_model_path):
        return None, f"❌ Webcam model not found: {webcam_model_path}"

    try:
        # Import YOLOv5 components if not already imported
        import_yolov5()
        
        if not YOLO_AVAILABLE:
            return None, "❌ YOLOv5 components not available"
        
        device = select_device('cpu')
        model = DetectMultiBackend(webcam_model_path, device=device, dnn=False, fp16=False)
        return model, "✅ Webcam model loaded successfully!"
    except Exception as e:
        return None, f"❌ Error loading webcam model: {e}"

def detect_plastic_advanced(image, detector, conf_threshold=0.25):
    """Advanced plastic detection using YOLOv8 or YOLOv5"""
    try:
        if DETECTOR_VERSION == "YOLOv8":
            # Use YOLOv8 detector
            result = detector.detect_plastic(image, return_annotated=True)
            
            # Convert to expected format
            detections = []
            for det in result['detections']:
                x1, y1, x2, y2 = det['bbox']
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': det['confidence'],
                    'class': det['class'],
                    'original_class': det['original_class'],
                    'model': det['model']
                })
            
            return {
                'detections': detections,
                'count': result['count'],
                'annotated_image': result['annotated_image'],
                'model_info': result['model_info']
            }
        
        elif DETECTOR_VERSION == "YOLOv5":
            # Use YOLOv5 detector
            result = detector.detect_plastic(image)
            return result
        
        else:
            return {'detections': [], 'count': 0, 'error': 'No detector available'}
            
    except Exception as e:
        return {'detections': [], 'count': 0, 'error': str(e)}

def detect_plastic(image, model, conf_threshold=0.08, filter_faces=True):
    """Optimized plastic detection with smart filtering"""
    try:
        # Ensure YOLOv5 components are imported
        if not YOLO_AVAILABLE:
            import_yolov5()
        
        img_size = 640
        stride = 32
        img = letterbox(image, img_size, stride=stride, auto=True)[0]
        img = img.transpose(2, 0, 1)[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(model.device).float() / 255.0
        img = img.unsqueeze(0)

        pred = model(img)[0]
        # MAXIMUM SENSITIVITY for dense marine debris detection
        # Ultra-low IoU (0.15) for maximum overlapping bottle detection
        # Very high max_det (200) for extremely dense scenes
        # Ultra-low confidence (0.01) for maximum recall
        pred = non_max_suppression(pred, max(conf_threshold, 0.01), 0.15, max_det=200)

        detections = []
        filtered_count = 0
        
        for det in pred:
            if det is not None and len(det):
                det = det.clone()
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    class_id = int(cls)
                    confidence = float(conf)
                    
                    # Calculate detection properties for smart filtering
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    aspect_ratio = width / height if height > 0 else 0
                    
                    # ENHANCED SMART FILTERING TO REMOVE FACES/SKIN/PHONES
                    is_likely_face = False
                    
                    if filter_faces:
                        img_area = image.shape[0] * image.shape[1]
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        img_center_x = image.shape[1] / 2
                        img_center_y = image.shape[0] / 2
                        
                        # ENHANCED Filter 1: Human face detection (stronger filtering)
                        # Face-like aspect ratio and position
                        if (0.6 <= aspect_ratio <= 1.6 and  # Face-like proportions
                            area > img_area * 0.05 and      # Minimum face size
                            y1 < image.shape[0] * 0.7):     # Upper portion (head area)
                            is_likely_face = True
                        
                        # ENHANCED Filter 2: Large central detections (likely human body/face)
                        if (area > img_area * 0.12 and      # Large detection
                            aspect_ratio > 0.4 and          # Not too thin
                            abs(center_x - img_center_x) < image.shape[1] * 0.4):  # Central position
                            is_likely_face = True
                        
                        # ENHANCED Filter 3: High confidence human-like detections
                        # If confidence is high but it's likely a person, filter it
                        if (confidence > 0.6 and            # High confidence
                            0.7 <= aspect_ratio <= 1.3 and  # Human-like proportions
                            area > img_area * 0.08):        # Significant size
                            is_likely_face = True
                        
                        # ENHANCED Filter 4: Upper body detection
                        # Detections in upper half of image with human proportions
                        if (y1 < image.shape[0] * 0.6 and   # Upper half
                            0.5 <= aspect_ratio <= 1.5 and  # Human proportions
                            area > img_area * 0.06):        # Reasonable size
                            is_likely_face = True
                        
                        # ENHANCED Filter 5: Skin tone detection (approximate)
                        # Extract region and check for skin-like colors
                        try:
                            roi = image[y1:y2, x1:x2]
                            if roi.size > 0:
                                # Convert to HSV for better skin detection
                                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
                                # Skin tone ranges in HSV
                                lower_skin = np.array([0, 20, 70])
                                upper_skin = np.array([20, 255, 255])
                                skin_mask = cv2.inRange(roi_hsv, lower_skin, upper_skin)
                                skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
                                
                                # If significant skin-like pixels, likely human
                                if (skin_ratio > 0.3 and        # 30% skin-like pixels
                                    0.6 <= aspect_ratio <= 1.4 and  # Human proportions
                                    area > img_area * 0.04):    # Minimum size
                                    is_likely_face = True
                        except:
                            pass  # Skip skin detection if it fails
                        
                        # ENHANCED Filter 6: OpenCV Face Detection
                        # Use Haar cascade for face detection
                        try:
                            # Convert to grayscale for face detection
                            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                            # Use built-in face detector
                            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                            
                            # Check if current detection overlaps with detected faces
                            for (fx, fy, fw, fh) in faces:
                                # Calculate overlap with current detection
                                overlap_x1 = max(x1, fx)
                                overlap_y1 = max(y1, fy)
                                overlap_x2 = min(x2, fx + fw)
                                overlap_y2 = min(y2, fy + fh)
                                
                                if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                                    detection_area = (x2 - x1) * (y2 - y1)
                                    overlap_ratio = overlap_area / detection_area if detection_area > 0 else 0
                                    
                                    # If significant overlap with face, filter it out
                                    if overlap_ratio > 0.3:  # 30% overlap
                                        is_likely_face = True
                                        break
                        except:
                            pass  # Skip OpenCV face detection if it fails
                        
                        # Filter 7: Phone/handheld object detection
                        if (y1 > image.shape[0] * 0.4 and   # Lower portion
                            0.05 < area / img_area < 0.20 and  # Medium size
                            0.4 < aspect_ratio < 0.8 and    # Phone-like rectangle
                            confidence < 0.25):             # Lower confidence
                            is_likely_face = True
                        
                        # Filter 8: Very small detections (noise)
                        if area < img_area * 0.008:
                            is_likely_face = True
                        
                        # Filter 9: Very large detections (likely full person)
                        if area > img_area * 0.4:
                            is_likely_face = True
                        
                        # Filter 10: Fabric/Textile Detection
                        # Detect pillows, clothes, fabric items that aren't plastic
                        try:
                            roi = image[y1:y2, x1:x2]
                            if roi.size > 0:
                                # Convert to HSV for better texture analysis
                                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
                                
                                # Check for fabric-like texture patterns
                                gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                                
                                # Calculate texture variance (fabrics have more texture)
                                texture_variance = np.var(gray_roi)
                                
                                # Check for fabric-like colors (browns, grays, patterns)
                                # Fabric items often have muted colors
                                mean_saturation = np.mean(roi_hsv[:, :, 1])
                                mean_value = np.mean(roi_hsv[:, :, 2])
                                
                                # Fabric characteristics:
                                # - High texture variance (patterns, weaves)
                                # - Lower saturation (muted colors)
                                # - Medium to low brightness
                                # - Rectangular/square shape (pillows, clothes)
                                if (texture_variance > 800 and           # High texture
                                    mean_saturation < 100 and           # Low saturation (muted)
                                    0.6 <= aspect_ratio <= 1.8 and     # Fabric-like proportions
                                    area > img_area * 0.02 and         # Reasonable size
                                    confidence < 0.3):                 # Lower confidence
                                    is_likely_face = True
                                
                                # Additional fabric detection for patterned items
                                # Check for repetitive patterns (common in fabrics)
                                if (texture_variance > 1200 and         # Very high texture
                                    0.7 <= aspect_ratio <= 1.5 and     # Square-ish (pillow-like)
                                    area > img_area * 0.03 and         # Medium size
                                    confidence < 0.25):               # Low confidence
                                    is_likely_face = True
                        except:
                            pass  # Skip fabric detection if it fails
                        
                        # Filter 11: Pillow/Cushion specific detection
                        # Pillows are typically square/rectangular, medium-large size, held in hands
                        if (0.6 <= aspect_ratio <= 1.6 and     # Square-ish proportions
                            0.03 < area / img_area < 0.25 and   # Medium size (pillow-like)
                            y1 > image.shape[0] * 0.2 and      # Not in very top (unlikely pillow position)
                            confidence < 0.2):                 # Low confidence
                            is_likely_face = True
                        
                        # Filter 12: Handheld non-plastic items - DISABLED FOR BOTTLES
                        # Items held in hands that are clearly not plastic bottles
                        # DISABLED: Allow all handheld items (could be bottles)
                        # if (y1 > image.shape[0] * 0.3 and      # Lower portion (hand area)
                        #     0.02 < area / img_area < 0.15 and   # Hand-holdable size
                        #     0.5 <= aspect_ratio <= 2.0 and     # Various handheld shapes
                        #     confidence < 0.12):                # Very low confidence only
                        #     is_likely_face = True
                        pass  # Disabled to allow horizontal bottles
                        
                        # Filter 13: Low confidence non-bottle items
                        # Very low confidence detections are likely false positives
                        # IMPROVED: Allow horizontal bottles (removed aspect_ratio check)
                        # Bottles can be vertical (thin) OR horizontal (wide)
                        if (confidence < 0.08 and              # Extremely low confidence only
                            0.85 <= aspect_ratio <= 1.15 and   # Only filter square shapes (not bottles)
                            area > img_area * 0.02):           # Not tiny noise
                            is_likely_face = True
                        
                        # Filter 14: Enhanced fabric/pillow detection - RELAXED FOR BOTTLES
                        # Some fabric items get higher confidence, need stronger filtering
                        # RELAXED: Only filter very low confidence to allow bottles
                        if (0.9 <= aspect_ratio <= 1.1 and     # Only very square (not bottles)
                            0.04 < area / img_area < 0.3 and    # Medium to large size
                            y1 > image.shape[0] * 0.15 and     # Not at very top
                            confidence < 0.15):                # Much lower threshold
                            # Additional check for position (likely handheld)
                            if (center_x > image.shape[1] * 0.2 and  # Not far left
                                center_x < image.shape[1] * 0.8):    # Not far right (center-ish)
                                is_likely_face = True
                        
                        # Filter 15: Matchbox/small rectangular object detection - DISABLED
                        # Small rectangular objects like matchboxes, cards, etc.
                        # DISABLED: Could filter horizontal bottles
                        # if (0.4 <= aspect_ratio <= 2.5 and
                        #     0.005 < area / img_area < 0.08 and
                        #     confidence < 0.25 and
                        #     y1 > image.shape[0] * 0.2):
                        #     is_likely_face = True
                        pass  # Disabled to allow horizontal bottles
                    
                    # Apply filters
                    # BUT: Allow high-confidence detections to pass through
                    # If confidence is high (>0.25), it's likely a real plastic bottle
                    if is_likely_face and confidence < 0.25:
                        filtered_count += 1
                        continue
                    
                    # Filter 16: Blue Bottle Filter - DISABLED
                    # NO LONGER NEEDED! The new trained model (98.6% accuracy) properly
                    # distinguishes between blue plastic bottles and blue metal bottles.
                    # The model will correctly label:
                    #   - Blue plastic bottles → "plastic"
                    #   - Blue metal bottles → "metal"
                    # So we don't need to filter all blue bottles anymore!
                    
                    # Filter 17: Metal/Steel Detection by Class ID - DISABLED
                    # PROBLEM: Model misclassifies yellow plastic bottles as Metal (class_id 5)
                    # So we can't rely on class_id alone. Use visual detection instead.
                    # Class names: ['plastic', 'Bottle cap', 'Can', 'Juice Box', 'Juice box', 'Metal', 'Metal Waste', 'Undefined trash', 'Wood']
                    # if class_id in [5, 6]:  # Metal and Metal Waste - DISABLED
                    #     filtered_count += 1
                    #     print(f"🚫 Filtered metal object by class_id (class_id: {class_id}, confidence: {confidence:.2f})")
                    #     continue
                    
                    # Filter 17: Visual Metal/Steel Detection - DISABLED
                    # NO LONGER NEEDED! The new trained model (98.6% accuracy) uses AI to
                    # properly classify materials. These old visual heuristics were a workaround
                    # before we had the proper trained model.
                    # 
                    # The new model will correctly identify:
                    #   - Blue plastic bottles → class_id=2 ("plastic")
                    #   - Blue metal bottles → class_id=1 ("metal")
                    #   - Clear plastic bottles → class_id=2 ("plastic")
                    #
                    # Trust the AI model, not manual color detection!
                    
                    # Get class name from model
                    # Custom trained model has 4 classes: {0: 'concrete', 1: 'metal', 2: 'plastic', 3: 'wood'}
                    try:
                        # Try to get class name from model
                        if hasattr(model, 'names'):
                            class_name = model.names[int(class_id)]
                        else:
                            class_name = 'unknown'
                    except:
                        class_name = 'unknown'
                    
                    # For marine waste detection, accept bottle-related classes
                    # Original YOLOv5: bottle, cup, wine glass (COCO classes)
                    # Custom model: plastic, metal
                    bottle_classes = ['bottle', 'cup', 'wine glass', 'plastic', 'metal', 'vase', 'bowl']
                    
                    # Check if this is a bottle-related class
                    is_bottle = any(bc in class_name.lower() for bc in bottle_classes)
                    
                    # DEBUG: Print what we're detecting
                    if confidence > 0.05:
                        print(f"🔍 Detected: {class_name} (conf: {confidence:.3f}, is_bottle: {is_bottle})")
                    
                    if not is_bottle:
                        filtered_count += 1
                        print(f"🚫 Filtered non-bottle: {class_name}")
                        continue
                    
                    # Map all bottle-related classes to "plastic" for consistency
                    # (Most bottles in ocean are plastic anyway)
                    class_name = 'plastic'
                    
                    # CONFIDENCE BOOST: Increase confidence for bottle detections
                    # The model gives low confidence for horizontal/unusual angles
                    # Boost it to make detection more visible and reliable
                    boosted_confidence = min(confidence * 6.0, 0.95)  # Boost by 6x, cap at 0.95
                    
                    # If original confidence was very low, boost less aggressively
                    if confidence < 0.05:
                        boosted_confidence = min(confidence * 4.0, 0.85)
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': boosted_confidence,  # Use boosted confidence
                        'class_id': class_id,
                        'class_name': class_name
                    })
        
        # Debug output
        if len(detections) > 0 or filtered_count > 0:
            print(f"🎯 Detection: {len(detections)} valid, {filtered_count} filtered (conf >= {conf_threshold:.2f})")
        
        return detections

    except Exception as e:
        print(f"Detection error: {e}")
        return []

# Streamlit app
st.set_page_config(
    page_title="Smart Marine Project", 
    page_icon="🌊", 
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None,
        # This hides all default menu items in newer Streamlit
    }
)

# Hide only Settings from menu - More aggressive approach
hide_settings_style = """
<style>
/* Replace Streamlit's kebab menu (three dots) with gear icon */
[data-testid="stToolbar"] {
    display: flex !important;
}

/* Hide default toolbar button content and SVG */
[data-testid="stToolbar"] button > div,
[data-testid="stToolbar"] button[kind="header"] > div,
button[data-testid="baseButton-header"] > div,
[data-testid="stToolbar"] button svg,
[data-testid="stToolbar"] button[kind="header"] svg,
button[data-testid="baseButton-header"] svg,
[data-testid="stToolbar"] button path,
[data-testid="stToolbar"] button[kind="header"] path,
button[data-testid="baseButton-header"] path {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
}

/* Custom Settings Icon Button with Square Border */
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] button[kind="header"],
button[data-testid="baseButton-header"] {
    background-color: transparent !important;
    border: 1.5px solid rgba(14, 165, 233, 0.4) !important;
    border-radius: 8px !important;
    padding: 8px !important;
    cursor: pointer !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    width: 44px !important;
    height: 44px !important;
    transform: rotate(0deg) !important;
}

[data-testid="stToolbar"] button:hover,
[data-testid="stToolbar"] button[kind="header"]:hover,
button[data-testid="baseButton-header"]:hover {
    background-color: rgba(14, 165, 233, 0.1) !important;
    border-color: #0ea5e9 !important;
    box-shadow: 0 0 8px rgba(14, 165, 233, 0.3) !important;
    transform: rotate(90deg) !important;
}

/* Add custom SVG gear icon */
[data-testid="stToolbar"] button::before,
[data-testid="stToolbar"] button[kind="header"]::before,
button[data-testid="baseButton-header"]::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%) rotate(0deg);
    width: 28px;
    height: 28px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='28' height='28' viewBox='0 0 24 24' fill='none' stroke='%230ea5e9' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z'%3E%3C/path%3E%3Cpath d='M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9c0 .65.26 1.27.73 1.73a1.65 1.65 0 0 0 1.51 1h.09a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z'%3E%3C/path%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1), filter 0.3s ease;
    transform-origin: center center;
}

/* Hover Effect – Rotate Entire Gear 90° + Enhance Color */
[data-testid="stToolbar"] button:hover::before,
[data-testid="stToolbar"] button[kind="header"]:hover::before,
button[data-testid="baseButton-header"]:hover::before {
    transform: translate(-50%, -50%) rotate(90deg);
    filter: drop-shadow(0 0 4px #06b6d4) drop-shadow(0 0 8px #06b6d4);
}

/* Create custom toolbar */
.custom-toolbar {
    position: fixed;
    top: 12px;
    right: 60px;
    z-index: 999999;
    display: flex;
    gap: 8px;
}

/* Settings Panel */
.settings-panel {
    position: fixed;
    top: 60px;
    right: 10px;
    width: 300px;
    background: rgba(15, 23, 42, 0.95);
    border: 1px solid #0ea5e9;
    border-radius: 12px;
    padding: 20px;
    z-index: 999998;
    display: none;
    backdrop-filter: blur(10px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
}

.settings-panel.show {
    display: block;
    animation: slideIn 0.3s ease;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(-10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.settings-panel h3 {
    color: #0ea5e9;
    margin: 0 0 15px 0;
    font-size: 18px;
    font-weight: 600;
}

.settings-option {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 12px 0;
    color: #cbd5e1;
    font-size: 14px;
}

.settings-toggle {
    width: 40px;
    height: 20px;
    background: #334155;
    border-radius: 10px;
    position: relative;
    cursor: pointer;
    transition: background 0.3s ease;
}

.settings-toggle.active {
    background: #0ea5e9;
}

.settings-toggle::after {
    content: '';
    width: 16px;
    height: 16px;
    background: white;
    border-radius: 50%;
    position: absolute;
    top: 2px;
    left: 2px;
    transition: transform 0.3s ease;
}

.settings-toggle.active::after {
    transform: translateX(20px);
}

.settings-button-container {
    position: fixed;
    top: 12px;
    right: 60px;
    z-index: 999999;
}

.settings-button {
    background-color: transparent;
    border: 1px solid rgba(14, 165, 233, 0.3);
    padding: 8px;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.settings-button:hover {
    background-color: rgba(14, 165, 233, 0.15);
    border-color: #0ea5e9;
}

.svg-path {
    stroke-dasharray: 100;
    stroke-dashoffset: 0;
    transition: stroke-width 0.3s ease;
    stroke: #0ea5e9;
    fill: none;
}

.settings-button:hover .svg-path {
    animation: draw 500ms ease-in forwards;
    stroke-width: 2;
    stroke: #06b6d4;
}

.toolbar-button {
    background: transparent;
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: #fff;
    padding: 8px;
    padding: 8px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    transition: all 0.2s ease;
    display: none !important;
}

/* Hover-Based Collapsible Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 50%, #0f172a 100%) !important;
    border-right: 1px solid #334155 !important;
    width: 70px !important;
    min-width: 70px !important;
    max-width: 70px !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    overflow: hidden !important;
    position: relative !important;
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}

/* Hamburger Icon (Three Lines) - Clean, No Glow */
section[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 20px;
    left: 20px;
    width: 30px;
    height: 3px;
    background: #0ea5e9;
    box-shadow: 
        0 8px 0 #0ea5e9,
        0 16px 0 #0ea5e9;
    transition: all 0.3s ease;
    z-index: 1000;
    pointer-events: none;
}

/* Color change on hover - No Glow */
section[data-testid="stSidebar"]:hover::before {
    background: #06b6d4;
    box-shadow: 
        0 8px 0 #06b6d4,
        0 16px 0 #06b6d4;
}

/* Settings Button SVG Animation - Top Left Corner */
@keyframes draw {
    0% { stroke-dashoffset: 100; }
    100% { stroke-dashoffset: 0; }
}

/* Hide Settings from menu - Streamlit 1.50.0 specific */
/* Target the second menu item which is usually Settings */
ul[role="menu"] li:nth-child(2),
[data-testid="stMainMenuList"] li:nth-child(2),
[data-testid="main-menu"] li:nth-child(2) {
    display: none !important;
}

/* Hide Settings button specifically */
button[title="Settings"],
button[aria-label="Settings"],
li button[title="Settings"] {
    display: none !important;
}

/* Hide content when collapsed */
section[data-testid="stSidebar"] > div {
    background: transparent;
    padding-top: 60px;
    opacity: 0;
    transform: translateX(-20px);
    transition: all 0.4s ease;
    pointer-events: none;
}

/* Expand sidebar on hover */
section[data-testid="stSidebar"]:hover {
    width: 300px !important;
    min-width: 300px !important;
    max-width: 300px !important;
}

/* Show content when expanded */
section[data-testid="stSidebar"]:hover > div {
    opacity: 1;
    transform: translateX(0);
    pointer-events: auto;
}

/* Hide Streamlit's default collapse button (« chevron) */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
button[aria-label="Collapse sidebar"],
button[title="Collapse sidebar"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* Keep hamburger menu visible but hide any toolbar/settings icons in sidebar */
section[data-testid="stSidebar"] [data-testid="stToolbar"],
section[data-testid="stSidebar"] [data-testid="baseButton-header"],
section[data-testid="stSidebar"] button[kind="header"] {
    display: none !important;
}

/* Hide the collapsed sidebar gear icon (left side with >>) but keep hamburger */
section[data-testid="stSidebar"] button:not([aria-label="Hamburger"]) {
    display: none !important;
}

/* Ensure hamburger menu ::before stays visible */
section[data-testid="stSidebar"]::before {
    display: block !important;
    visibility: visible !important;
}

/* Sidebar Header Styling with Enhanced Hover Effects */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #0ea5e9 !important;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    transition: all 0.3s ease;
    padding: 8px 12px;
    border-radius: 8px;
    cursor: pointer;
}

section[data-testid="stSidebar"] h1:hover,
section[data-testid="stSidebar"] h2:hover,
section[data-testid="stSidebar"] h3:hover {
    color: #06b6d4 !important;
    transform: translateX(5px);
    background: rgba(14, 165, 233, 0.1);
    box-shadow: 0 0 10px rgba(6, 182, 212, 0.3);
}

/* Sidebar Text Styling - No Hover Effects for Labels */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown {
    color: #cbd5e1 !important;
    font-weight: 500;
}

section[data-testid="stSidebar"] strong {
    color: #06b6d4 !important;
    font-weight: 700;
    transition: all 0.3s ease;
}

section[data-testid="stSidebar"] strong:hover {
    color: #0ea5e9 !important;
    text-shadow: 0 0 8px rgba(6, 182, 212, 0.6);
}

/* Expander Styling - Clean, No Glow */
section[data-testid="stSidebar"] .streamlit-expanderHeader {
    background: rgba(14, 165, 233, 0.1);
    border: 1px solid rgba(6, 182, 212, 0.3);
    border-radius: 8px;
    color: #0ea5e9 !important;
    font-weight: 600;
    transition: all 0.3s ease;
    margin: 10px 0;
}

section[data-testid="stSidebar"] .streamlit-expanderHeader:hover {
    background: rgba(14, 165, 233, 0.2);
    border-color: #06b6d4;
    transform: translateX(5px);
}

/* Slider Styling with Hover Effects */
section[data-testid="stSidebar"] .stSlider {
    transition: all 0.3s ease;
    padding: 8px;
    border-radius: 8px;
}

section[data-testid="stSidebar"] .stSlider:hover {
    background: rgba(14, 165, 233, 0.05);
    box-shadow: 0 0 15px rgba(6, 182, 212, 0.2);
}

section[data-testid="stSidebar"] .stSlider > div > div > div {
    background: linear-gradient(90deg, #0ea5e9, #06b6d4);
    transition: all 0.3s ease;
}

section[data-testid="stSidebar"] .stSlider:hover > div > div > div {
    box-shadow: 0 0 10px rgba(6, 182, 212, 0.5);
}

section[data-testid="stSidebar"] .stSlider > div > div > div > div {
    background: #06b6d4;
    transition: transform 0.3s ease;
}

section[data-testid="stSidebar"] .stSlider:hover > div > div > div > div {
    transform: scale(1.2);
    box-shadow: 0 0 8px rgba(6, 182, 212, 0.8);
}

/* Info/Success/Warning Boxes in Sidebar with Hover Effects */
section[data-testid="stSidebar"] .stAlert {
    background: rgba(14, 165, 233, 0.1);
    border: 1px solid rgba(6, 182, 212, 0.3);
    border-radius: 8px;
    color: #cbd5e1 !important;
    transition: all 0.3s ease;
    cursor: pointer;
}

section[data-testid="stSidebar"] .stAlert:hover {
    background: rgba(14, 165, 233, 0.2);
    border-color: #06b6d4;
    transform: translateX(5px) scale(1.02);
    box-shadow: 0 0 15px rgba(6, 182, 212, 0.4);
}

/* Caption Styling - No Hover Effects */
section[data-testid="stSidebar"] .stCaptionContainer {
    color: #94a3b8 !important;
    font-size: 0.85rem;
}

section[data-testid="stSidebar"] .stCaptionContainer:hover {
    color: #94a3b8 !important;
}

/* ═══════════════════════════════════════════════════════════════════
   PREMIUM SECTION HEADERS - AI OPTIMIZATIONS, SYSTEM STATUS, PERFORMANCE
   ═══════════════════════════════════════════════════════════════════ */

.section-header {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.15) 0%, rgba(6, 182, 212, 0.15) 100%) !important;
    border-left: 4px solid #0ea5e9 !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    margin: 20px 0 16px 0 !important;
    font-size: 1.1rem !important;
    font-weight: 800 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #0ea5e9 !important;
    position: relative !important;
    overflow: hidden !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 
        0 4px 15px rgba(14, 165, 233, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
}

/* Animated gradient background */
.section-header::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: -100% !important;
    width: 100% !important;
    height: 100% !important;
    background: linear-gradient(90deg, transparent, rgba(14, 165, 233, 0.3), transparent) !important;
    transition: left 0.6s ease !important;
}

.section-header:hover::before {
    left: 100% !important;
}

/* Hover effects */
.section-header:hover {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.25) 0%, rgba(6, 182, 212, 0.25) 100%) !important;
    border-left-color: #06b6d4 !important;
    color: #ffffff !important;
    transform: translateX(5px) !important;
    box-shadow: 
        0 6px 25px rgba(14, 165, 233, 0.4),
        0 0 30px rgba(6, 182, 212, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
}

/* Pulsing glow animation */
.section-header {
    animation: sectionPulse 3s ease-in-out infinite !important;
}

@keyframes sectionPulse {
    0%, 100% {
        box-shadow: 
            0 4px 15px rgba(14, 165, 233, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    50% {
        box-shadow: 
            0 6px 20px rgba(14, 165, 233, 0.35),
            0 0 25px rgba(6, 182, 212, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }
}

/* Icon before text */
.section-header::after {
    content: '▶' !important;
    position: absolute !important;
    right: 16px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    font-size: 0.8rem !important;
    color: #0ea5e9 !important;
    transition: all 0.3s ease !important;
    opacity: 0.6 !important;
}

.section-header:hover::after {
    color: #ffffff !important;
    opacity: 1 !important;
    transform: translateY(-50%) translateX(3px) !important;
}

/* Main Header Styling with Hover Effect */
.main-header {
    color: #0ea5e9 !important;
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    margin: 20px 0 10px 0;
    letter-spacing: 3px;
    text-transform: uppercase;
    transition: all 0.3s ease;
    cursor: default;
    font-family: 'Montserrat', 'Inter', 'Segoe UI', sans-serif;
    text-shadow: 0 0 20px rgba(14, 165, 233, 0.5);
}

.main-header:hover {
    color: #06b6d4 !important;
    transform: scale(1.05);
    text-shadow: 0 0 30px rgba(6, 182, 212, 0.7);
}

.sub-header {
    text-align: center;
    color: #94a3b8;
    font-size: 1.2rem;
    margin-bottom: 30px;
}

.sub-header:hover {
    color: #0ea5e9 !important;
}

/* Animated Tab Buttons with Gradient Hover Effects */
[data-baseweb="tab-list"] {
    display: flex !important;
    gap: 25px !important;
    background: transparent !important;
    border: none !important;
    border-bottom: none !important;
}

/* Hide the full-width tab border line */
[data-baseweb="tab-border"] {
    display: none !important;
}

/* Keep the active tab highlight but style it */
[data-baseweb="tab-highlight"] {
    background-color: #0ea5e9 !important;
    height: 2px !important;
}

/* Remove full-width border from tabs container */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: none !important;
}

[role="tablist"] {
    border-bottom: none !important;
}

[data-baseweb="tab-list"] button {
    position: relative !important;
    width: 60px !important;
    height: 60px !important;
    background: rgba(14, 165, 233, 0.15) !important;
    box-shadow: 0 10px 25px rgba(14, 165, 233, 0.2) !important;
    border-radius: 60px !important;
    cursor: pointer !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    transition: all 0.5s ease !important;
    border: 1px solid rgba(14, 165, 233, 0.3) !important;
    overflow: visible !important;
    padding: 0 !important;
    font-size: 0 !important;
    color: transparent !important;
}

/* Hide all default Streamlit tab content */
[data-baseweb="tab-list"] button span,
[data-baseweb="tab-list"] button p,
[data-baseweb="tab-list"] button div[data-baseweb="tab-panel"] {
    font-size: 0 !important;
    color: transparent !important;
    opacity: 0 !important;
}

/* Gradient background layer */
[data-baseweb="tab-list"] button::before {
    content: "" !important;
    position: absolute !important;
    inset: 0 !important;
    border-radius: 60px !important;
    opacity: 0 !important;
    transition: opacity 0.5s ease !important;
    z-index: 0 !important;
}

/* Gradient colors for each tab - Marine theme */
[data-baseweb="tab-list"] button:nth-child(1)::before {
    background: linear-gradient(45deg, #0ea5e9, #06b6d4) !important;
}

[data-baseweb="tab-list"] button:nth-child(2)::before {
    background: linear-gradient(45deg, #06b6d4, #0891b2) !important;
}

[data-baseweb="tab-list"] button:nth-child(3)::before {
    background: linear-gradient(45deg, #0891b2, #0e7490) !important;
}

[data-baseweb="tab-list"] button:nth-child(4)::before {
    background: linear-gradient(45deg, #0e7490, #155e75) !important;
}

[data-baseweb="tab-list"] button:nth-child(5)::before {
    background: linear-gradient(45deg, #155e75, #164e63) !important;
}

/* Glow effect */
[data-baseweb="tab-list"] button::after {
    content: "" !important;
    position: absolute !important;
    top: 10px !important;
    width: 100% !important;
    height: 100% !important;
    border-radius: 60px !important;
    transition: all 0.5s ease !important;
    filter: blur(15px) !important;
    z-index: -1 !important;
    opacity: 0 !important;
}

[data-baseweb="tab-list"] button:nth-child(1)::after {
    background: linear-gradient(45deg, #0ea5e9, #06b6d4) !important;
}

[data-baseweb="tab-list"] button:nth-child(2)::after {
    background: linear-gradient(45deg, #06b6d4, #0891b2) !important;
}

[data-baseweb="tab-list"] button:nth-child(3)::after {
    background: linear-gradient(45deg, #0891b2, #0e7490) !important;
}

[data-baseweb="tab-list"] button:nth-child(4)::after {
    background: linear-gradient(45deg, #0e7490, #155e75) !important;
}

[data-baseweb="tab-list"] button:nth-child(5)::after {
    background: linear-gradient(45deg, #155e75, #164e63) !important;
}

/* Hover state - expand button */
[data-baseweb="tab-list"] button:hover {
    width: 220px !important;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0) !important;
}

[data-baseweb="tab-list"] button:hover::before {
    opacity: 1 !important;
}

[data-baseweb="tab-list"] button:hover::after {
    opacity: 0.5 !important;
}

/* Icon styling */
[data-baseweb="tab-list"] button > div {
    position: relative !important;
    z-index: 1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 28px !important;
    height: 28px !important;
    transition: all 0.5s ease !important;
    transition-delay: 0.25s !important;
}

/* Icons - Bright Cyan color for marine theme */
[data-baseweb="tab-list"] button:nth-child(1) > div::before {
    content: '' !important;
    display: block !important;
    width: 28px !important;
    height: 28px !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 512'%3E%3Cpath fill='%2306b6d4' d='M149.1 64.8L138.7 96H64C28.7 96 0 124.7 0 160V416c0 35.3 28.7 64 64 64H448c35.3 0 64-28.7 64-64V160c0-35.3-28.7-64-64-64H373.3L362.9 64.8C356.4 45.2 338.1 32 317.4 32H194.6c-20.7 0-39 13.2-45.5 32.8zM256 192a96 96 0 1 1 0 192 96 96 0 1 1 0-192z'/%3E%3C/svg%3E") !important;
    background-size: contain !important;
    background-repeat: no-repeat !important;
    transition: all 0.5s ease !important;
    transition-delay: 0.25s !important;
    filter: brightness(1.2) !important;
}

[data-baseweb="tab-list"] button:nth-child(2) > div::before {
    content: '' !important;
    display: block !important;
    width: 28px !important;
    height: 28px !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 576 512'%3E%3Cpath fill='%2306b6d4' d='M0 128C0 92.7 28.7 64 64 64H320c35.3 0 64 28.7 64 64V384c0 35.3-28.7 64-64 64H64c-35.3 0-64-28.7-64-64V128zM559.1 99.8c10.4 5.6 16.9 16.4 16.9 28.2V384c0 11.8-6.5 22.6-16.9 28.2s-23 5-32.9-1.6l-96-64L416 337.1V320 192 174.9l14.2-9.5 96-64c9.8-6.5 22.4-7.2 32.9-1.6z'/%3E%3C/svg%3E") !important;
    background-size: contain !important;
    background-repeat: no-repeat !important;
    transition: all 0.5s ease !important;
    transition-delay: 0.25s !important;
    filter: brightness(1.2) !important;
}

[data-baseweb="tab-list"] button:nth-child(3) > div::before {
    content: '' !important;
    display: block !important;
    width: 28px !important;
    height: 28px !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 576 512'%3E%3Cpath fill='%2306b6d4' d='M264.5 5.2c14.9-6.9 32.1-6.9 47 0l218.6 101c8.5 3.9 13.9 12.4 13.9 21.8s-5.4 17.9-13.9 21.8l-218.6 101c-14.9 6.9-32.1 6.9-47 0L45.9 149.8C37.4 145.8 32 137.3 32 128s5.4-17.9 13.9-21.8L264.5 5.2zM476.9 209.6l53.2 24.6c8.5 3.9 13.9 12.4 13.9 21.8s-5.4 17.9-13.9 21.8l-218.6 101c-14.9 6.9-32.1 6.9-47 0L45.9 277.8C37.4 273.8 32 265.3 32 256s5.4-17.9 13.9-21.8l53.2-24.6 152 70.2c23.4 10.8 50.4 10.8 73.8 0l152-70.2zm-152 198.2l152-70.2 53.2 24.6c8.5 3.9 13.9 12.4 13.9 21.8s-5.4 17.9-13.9 21.8l-218.6 101c-14.9 6.9-32.1 6.9-47 0L45.9 405.8C37.4 401.8 32 393.3 32 384s5.4-17.9 13.9-21.8l53.2-24.6 152 70.2c23.4 10.8 50.4 10.8 73.8 0z'/%3E%3C/svg%3E") !important;
    background-size: contain !important;
    background-repeat: no-repeat !important;
    transition: all 0.5s ease !important;
    transition-delay: 0.25s !important;
    filter: brightness(1.2) !important;
}

[data-baseweb="tab-list"] button:nth-child(4) > div::before {
    content: '' !important;
    display: block !important;
    width: 28px !important;
    height: 28px !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 512'%3E%3Cpath fill='%2306b6d4' d='M64 64c0-17.7-14.3-32-32-32S0 46.3 0 64V400c0 44.2 35.8 80 80 80H480c17.7 0 32-14.3 32-32s-14.3-32-32-32H80c-8.8 0-16-7.2-16-16V64zm406.6 86.6c12.5-12.5 12.5-32.8 0-45.3s-32.8-12.5-45.3 0L320 210.7l-57.4-57.4c-12.5-12.5-32.8-12.5-45.3 0l-112 112c-12.5 12.5-12.5 32.8 0 45.3s32.8 12.5 45.3 0L240 221.3l57.4 57.4c12.5 12.5 32.8 12.5 45.3 0l128-128z'/%3E%3C/svg%3E") !important;
    background-size: contain !important;
    background-repeat: no-repeat !important;
    transition: all 0.5s ease !important;
    transition-delay: 0.25s !important;
    filter: brightness(1.2) !important;
}

[data-baseweb="tab-list"] button:nth-child(5) > div::before {
    content: '' !important;
    display: block !important;
    width: 28px !important;
    height: 28px !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 576 512'%3E%3Cpath fill='%2306b6d4' d='M256 32H181.2c-27.1 0-51.3 17.1-60.3 42.6L3.1 407.2C1.1 413 0 419.2 0 425.4C0 455.5 24.5 480 54.6 480H256V416c0-17.7 14.3-32 32-32s32 14.3 32 32v64H521.4c30.2 0 54.6-24.5 54.6-54.6c0-6.2-1.1-12.4-3.1-18.2L455.1 74.6C446 49.1 421.9 32 394.8 32H320v64c0 17.7-14.3 32-32 32s-32-14.3-32-32V32zm-96 96c0-17.7 14.3-32 32-32h64c17.7 0 32 14.3 32 32v64c0 17.7-14.3 32-32 32H192c-17.7 0-32-14.3-32-32V128z'/%3E%3C/svg%3E") !important;
    background-size: contain !important;
    background-repeat: no-repeat !important;
    transition: all 0.5s ease !important;
    transition-delay: 0.25s !important;
    filter: brightness(1.2) !important;
}

/* Hide icon on hover */
[data-baseweb="tab-list"] button:hover > div::before {
    transform: scale(0) !important;
    transition-delay: 0s !important;
}

/* Text labels - hidden by default */
[data-baseweb="tab-list"] button > div::after {
    position: absolute !important;
    color: #fff !important;
    font-size: 1.1em !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    transform: scale(0) !important;
    transition: all 0.5s ease !important;
    transition-delay: 0s !important;
    white-space: nowrap !important;
}

[data-baseweb="tab-list"] button:nth-child(1) > div::after {
    content: 'SINGLE IMAGE' !important;
}

[data-baseweb="tab-list"] button:nth-child(2) > div::after {
    content: 'LIVE WEBCAM' !important;
}

[data-baseweb="tab-list"] button:nth-child(3) > div::after {
    content: 'BATCH UPLOAD' !important;
}

[data-baseweb="tab-list"] button:nth-child(4) > div::after {
    content: 'ANALYTICS' !important;
}

[data-baseweb="tab-list"] button:nth-child(5) > div::after {
    content: 'AUTONOMOUS MODE' !important;
}

/* Show text on hover */
[data-baseweb="tab-list"] button:hover > div::after {
    transform: scale(1) !important;
    transition-delay: 0.25s !important;
}

/* ========== MICRO-INTERACTIONS ========== */

/* Style horizontal divider lines - subtle gray line */
hr {
    border: none !important;
    border-top: 1px solid rgba(255, 255, 255, 0.1) !important;
    margin: 20px 0 !important;
}

[data-testid="stHorizontalBlock"] hr {
    border-top: 1px solid rgba(255, 255, 255, 0.1) !important;
}

.stMarkdown hr {
    border-top: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* Add separator line below tabs */
[role="tabpanel"] {
    border-top: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding-top: 20px !important;
}

/* Header container with bottle animation */
.header-container {
    position: relative;
    margin-bottom: 20px;
    overflow: hidden;
    min-height: 80px;
}

/* Tab Content Header Styling with Hover Effect */
.tab-content-header {
    color: #0ea5e9 !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    font-family: 'Inter', 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    margin: 0 !important;
    padding: 10px 15px !important;
    border-left: 4px solid #0ea5e9 !important;
    background: rgba(14, 165, 233, 0.05) !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    cursor: default !important;
    position: relative;
    z-index: 1;
}

.tab-content-header:hover {
    background: rgba(14, 165, 233, 0.15) !important;
    border-left-color: #06b6d4 !important;
    transform: translateX(5px) !important;
    box-shadow: 0 4px 12px rgba(14, 165, 233, 0.2) !important;
    letter-spacing: 3px !important;
}

/* Animated boat collecting plastic bottles */
.boat-float {
    position: absolute;
    top: 50%;
    left: 480px;
    width: 85px;
    height: 50px;
    opacity: 0.95;
    animation: boatFloat 12s ease-in-out infinite;
    z-index: 3;
    filter: drop-shadow(0 0 15px rgba(14, 165, 233, 0.8)) drop-shadow(0 0 30px rgba(6, 182, 212, 0.5));
    transform: translateY(-50%);
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 360'%3E%3Cpath fill='%230ea5e9' d='M0 240 L40 340 Q50 360 80 360 L432 360 Q462 360 472 340 L512 240 L344 240 L320 160 L160 160 L136 240 L0 240 Z M200 180 L280 180 L296 220 L184 220 L200 180 Z'/%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
}

/* Plastic bottles floating in water */
.header-container::before {
    content: '';
    position: absolute;
    top: 60%;
    left: 730px;
    width: 35px;
    height: 50px;
    opacity: 0;
    animation: bottleFloat1 12s ease-in-out infinite;
    z-index: 2;
    filter: drop-shadow(0 0 6px rgba(59, 130, 246, 0.5));
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 200'%3E%3Cdefs%3E%3ClinearGradient id='bottleGrad1' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' style='stop-color:%2393c5fd;stop-opacity:0.9'/%3E%3Cstop offset='50%25' style='stop-color:%2360a5fa;stop-opacity:0.7'/%3E%3Cstop offset='100%25' style='stop-color:%233b82f6;stop-opacity:0.8'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect x='30' y='0' width='40' height='15' rx='3' fill='%232563eb' opacity='0.9'/%3E%3Crect x='25' y='12' width='50' height='8' rx='2' fill='%233b82f6' opacity='0.8'/%3E%3Cpath d='M 35 20 L 30 35 L 30 180 Q 30 190 40 190 L 60 190 Q 70 190 70 180 L 70 35 L 65 20 Z' fill='url(%23bottleGrad1)' stroke='%2360a5fa' stroke-width='1.5' opacity='0.85'/%3E%3Cellipse cx='50' cy='100' rx='15' ry='25' fill='%23ffffff' opacity='0.15'/%3E%3Cpath d='M 35 60 Q 50 65 65 60' stroke='%23ffffff' stroke-width='1' fill='none' opacity='0.2'/%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
}

.header-container::after {
    content: '';
    position: absolute;
    top: 55%;
    left: 980px;
    width: 55px;
    height: 35px;
    opacity: 0;
    animation: bottleFloat2 12s ease-in-out infinite 0.5s;
    z-index: 2;
    filter: drop-shadow(0 0 6px rgba(59, 130, 246, 0.5));
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 100'%3E%3Cdefs%3E%3ClinearGradient id='bottleGrad2' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' style='stop-color:%2393c5fd;stop-opacity:0.85'/%3E%3Cstop offset='50%25' style='stop-color:%2360a5fa;stop-opacity:0.65'/%3E%3Cstop offset='100%25' style='stop-color:%233b82f6;stop-opacity:0.75'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect x='160' y='15' width='25' height='30' rx='3' fill='%232563eb' opacity='0.9'/%3E%3Cpath d='M 20 30 Q 15 35 18 45 L 25 60 Q 28 70 40 72 L 140 75 Q 155 75 158 65 L 165 45 Q 167 35 162 30 Q 145 25 130 28 L 100 32 Q 80 28 60 30 Q 40 28 20 30 Z' fill='url(%23bottleGrad2)' stroke='%2360a5fa' stroke-width='2' opacity='0.8'/%3E%3Cellipse cx='80' cy='50' rx='25' ry='12' fill='%23ffffff' opacity='0.2'/%3E%3Cpath d='M 50 45 Q 80 42 110 45' stroke='%23ffffff' stroke-width='1.5' fill='none' opacity='0.25'/%3E%3Cpath d='M 45 55 Q 80 58 115 55' stroke='%23ffffff' stroke-width='1' fill='none' opacity='0.2'/%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
}

/* Boat movement - stops to collect bottles */
@keyframes boatFloat {
    0% {
        transform: translateX(0px) translateY(-50%) rotate(-5deg) scale(1);
    }
    15% {
        transform: translateX(150px) translateY(-55%) rotate(5deg) scale(1);
    }
    25% {
        transform: translateX(250px) translateY(-50%) rotate(0deg) scale(1.1);
    }
    30% {
        transform: translateX(250px) translateY(-48%) rotate(0deg) scale(1.15);
    }
    35% {
        transform: translateX(300px) translateY(-50%) rotate(5deg) scale(1);
    }
    50% {
        transform: translateX(500px) translateY(-50%) rotate(0deg) scale(1.1);
    }
    55% {
        transform: translateX(500px) translateY(-48%) rotate(0deg) scale(1.15);
    }
    60% {
        transform: translateX(550px) translateY(-50%) rotate(5deg) scale(1);
    }
    75% {
        transform: translateX(700px) translateY(-55%) rotate(10deg) scale(1);
    }
    100% {
        transform: translateX(900px) translateY(-50%) rotate(-5deg) scale(1);
    }
}

/* First bottle - appears, gets collected, disappears */
@keyframes bottleFloat1 {
    0% {
        opacity: 0;
        transform: translateY(0) rotate(0deg) scale(1);
    }
    10% {
        opacity: 0.9;
        transform: translateY(-5px) rotate(10deg) scale(1);
    }
    20% {
        opacity: 0.9;
        transform: translateY(0) rotate(-10deg) scale(1);
    }
    25% {
        opacity: 0.9;
        transform: translateY(-3px) rotate(5deg) scale(1);
    }
    30% {
        opacity: 0;
        transform: translateY(-30px) rotate(360deg) scale(0.3);
    }
    100% {
        opacity: 0;
        transform: translateY(-30px) rotate(360deg) scale(0.3);
    }
}

/* Second bottle - appears, gets collected, disappears */
@keyframes bottleFloat2 {
    0% {
        opacity: 0;
        transform: translateY(0) rotate(0deg) scale(1);
    }
    15% {
        opacity: 0.9;
        transform: translateY(-5px) rotate(-10deg) scale(1);
    }
    35% {
        opacity: 0.9;
        transform: translateY(0) rotate(10deg) scale(1);
    }
    45% {
        opacity: 0.9;
        transform: translateY(-3px) rotate(-5deg) scale(1);
    }
    50% {
        opacity: 0.9;
        transform: translateY(-3px) rotate(0deg) scale(1);
    }
    55% {
        opacity: 0;
        transform: translateY(-30px) rotate(-360deg) scale(0.3);
    }
    100% {
        opacity: 0;
        transform: translateY(-30px) rotate(-360deg) scale(0.3);
    }
}

/* Smooth scroll behavior */
html {
    scroll-behavior: smooth !important;
}

* {
    scroll-behavior: smooth !important;
}

/* Page transition effects between tabs */
[data-baseweb="tab-panel"] {
    animation: fadeInUp 0.5s ease-out !important;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Tab content fade in */
[role="tabpanel"] {
    animation: fadeIn 0.4s ease-in-out !important;
}

@keyframes fadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

/* Button ripple effect - Only for primary buttons, not tabs or settings */
.stButton button {
    position: relative !important;
    overflow: hidden !important;
}

.stButton button::after {
    content: '' !important;
    position: absolute !important;
    top: 50% !important;
    left: 50% !important;
    width: 0 !important;
    height: 0 !important;
    border-radius: 50% !important;
    background: rgba(255, 255, 255, 0.5) !important;
    transform: translate(-50%, -50%) !important;
    transition: width 0.6s, height 0.6s !important;
    z-index: 0 !important;
}

.stButton button:active::after {
    width: 300px !important;
    height: 300px !important;
}

/* Smooth hover transitions for all interactive elements - except tabs and settings */
a, input, select, textarea {
    transition: all 0.3s ease !important;
}

/* Exclude tabs and toolbar buttons from general button styling */
button:not([data-baseweb="tab"]):not([data-testid="stToolbar"] button):not([data-testid="baseButton-header"]) {
    transition: all 0.3s ease !important;
}

/* Image smooth fade in */
img {
    animation: imageLoad 0.5s ease-in !important;
}

@keyframes imageLoad {
    from {
        opacity: 0;
        transform: scale(0.95);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}

/* ═══════════════════════════════════════════════════════════════════
   PREMIUM FILE UPLOADER - ENTERPRISE GRADE DESIGN
   ═══════════════════════════════════════════════════════════════════ */

/* File Uploader Container - Glassmorphism Effect */
[data-testid="stFileUploader"] {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%) !important;
    border: 2px solid transparent !important;
    border-radius: 20px !important;
    padding: 40px !important;
    backdrop-filter: blur(20px) !important;
    position: relative !important;
    overflow: hidden !important;
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 
        0 10px 40px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
}

/* Animated Border Gradient */
[data-testid="stFileUploader"]::before {
    content: '' !important;
    position: absolute !important;
    inset: 0 !important;
    border-radius: 20px !important;
    padding: 2px !important;
    background: linear-gradient(135deg, #0ea5e9, #06b6d4, #8b5cf6, #0ea5e9) !important;
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0) !important;
    -webkit-mask-composite: xor !important;
    mask-composite: exclude !important;
    opacity: 0 !important;
    transition: opacity 0.5s ease !important;
    animation: borderRotate 3s linear infinite !important;
    background-size: 300% 300% !important;
}

@keyframes borderRotate {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

[data-testid="stFileUploader"]:hover::before {
    opacity: 1 !important;
}

/* Hover State - Levitation Effect */
[data-testid="stFileUploader"]:hover {
    transform: translateY(-8px) scale(1.02) !important;
    box-shadow: 
        0 20px 60px rgba(14, 165, 233, 0.4),
        0 0 80px rgba(6, 182, 212, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    border-color: rgba(14, 165, 233, 0.5) !important;
}

/* Drag and Drop Area Styling */
[data-testid="stFileUploader"] section {
    border: 3px dashed rgba(14, 165, 233, 0.3) !important;
    border-radius: 16px !important;
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.05) 0%, rgba(6, 182, 212, 0.05) 100%) !important;
    padding: 60px 40px !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
}

/* Animated Background Particles */
[data-testid="stFileUploader"] section::before {
    content: '' !important;
    position: absolute !important;
    width: 200% !important;
    height: 200% !important;
    top: -50% !important;
    left: -50% !important;
    background: radial-gradient(circle, rgba(14, 165, 233, 0.1) 1px, transparent 1px) !important;
    background-size: 50px 50px !important;
    animation: particleMove 20s linear infinite !important;
    opacity: 0 !important;
    transition: opacity 0.5s ease !important;
}

@keyframes particleMove {
    0% { transform: translate(0, 0); }
    100% { transform: translate(50px, 50px); }
}

[data-testid="stFileUploader"]:hover section::before {
    opacity: 1 !important;
}

/* Drag Over State - Neon Glow */
[data-testid="stFileUploader"] section:hover {
    border-color: #0ea5e9 !important;
    border-style: solid !important;
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.15) 0%, rgba(6, 182, 212, 0.15) 100%) !important;
    box-shadow: 
        inset 0 0 40px rgba(14, 165, 233, 0.3),
        0 0 40px rgba(14, 165, 233, 0.2) !important;
    transform: scale(1.02) !important;
}

/* Upload Icon Styling */
[data-testid="stFileUploader"] svg {
    filter: drop-shadow(0 0 10px rgba(14, 165, 233, 0.6)) !important;
    transition: all 0.4s ease !important;
}

[data-testid="stFileUploader"]:hover svg {
    filter: drop-shadow(0 0 20px rgba(6, 182, 212, 0.8)) !important;
    transform: scale(1.15) translateY(-5px) !important;
    animation: float 3s ease-in-out infinite !important;
}

@keyframes float {
    0%, 100% { transform: scale(1.15) translateY(-5px); }
    50% { transform: scale(1.15) translateY(-15px); }
}

/* Upload Text Styling */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5) !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"]:hover label,
[data-testid="stFileUploader"]:hover span,
[data-testid="stFileUploader"]:hover p {
    color: #0ea5e9 !important;
    text-shadow: 0 0 15px rgba(14, 165, 233, 0.8) !important;
}

/* Browse Files Button - Premium Design */
[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    color: #ffffff !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 
        0 8px 24px rgba(14, 165, 233, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
}

/* Button Shine Effect */
[data-testid="stFileUploader"] button::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: -100% !important;
    width: 100% !important;
    height: 100% !important;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent) !important;
    transition: left 0.6s ease !important;
}

[data-testid="stFileUploader"] button:hover::before {
    left: 100% !important;
}

/* Button Hover State */
[data-testid="stFileUploader"] button:hover {
    transform: translateY(-4px) scale(1.05) !important;
    box-shadow: 
        0 12px 32px rgba(14, 165, 233, 0.6),
        0 0 40px rgba(6, 182, 212, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
    background: linear-gradient(135deg, #06b6d4 0%, #0ea5e9 100%) !important;
}

/* Button Active State */
[data-testid="stFileUploader"] button:active {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 
        0 6px 16px rgba(14, 165, 233, 0.5),
        inset 0 2px 4px rgba(0, 0, 0, 0.2) !important;
}

/* File Name Display - Elegant Card */
[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.1) 0%, rgba(6, 182, 212, 0.1) 100%) !important;
    border: 1px solid rgba(14, 165, 233, 0.3) !important;
    border-radius: 10px !important;
    padding: 12px 20px !important;
    margin-top: 20px !important;
    color: #0ea5e9 !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"]:hover {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.2) 0%, rgba(6, 182, 212, 0.2) 100%) !important;
    border-color: #0ea5e9 !important;
    transform: translateX(5px) !important;
    box-shadow: 0 6px 16px rgba(14, 165, 233, 0.3) !important;
}

/* ═══════════════════════════════════════════════════════════════════
   PREMIUM ANIMATED DELETE/REMOVE BUTTON - CODEPEN STYLE
   ═══════════════════════════════════════════════════════════════════ */

/* Delete Button Container - Animated Slide Effect */
[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    min-width: 160px !important;
    height: 56px !important;
    position: relative !important;
    overflow: hidden !important;
    cursor: pointer !important;
    transition: all 0.25s cubic-bezier(0.310, -0.105, 0.430, 1.400) !important;
    box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 12px !important;
}

/* Button Content Wrapper */
[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button > div {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    width: 100% !important;
    position: relative !important;
}

/* Button Text - Left Side (72%) */
[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button p,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button span:not(.icon-wrapper) {
    display: inline-block !important;
    position: relative !important;
    z-index: 2 !important;
    transition: all 0.25s cubic-bezier(0.310, -0.105, 0.430, 1.400) !important;
    opacity: 1 !important;
    transform: translateX(0) !important;
    margin: 0 !important;
    padding: 0 !important;
    flex: 0 0 72% !important;
    text-align: left !important;
}

/* Divider Line Between Text and Icon */
[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button::before {
    content: '' !important;
    background-color: #dc2626 !important;
    width: 2px !important;
    height: 70% !important;
    position: absolute !important;
    top: 15% !important;
    right: 28% !important;
    opacity: 1 !important;
    z-index: 1 !important;
    transition: all 0.25s cubic-bezier(0.310, -0.105, 0.430, 1.400) !important;
}

/* X Icon (Remove Icon) - Right Side (28%) */
[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button svg {
    position: absolute !important;
    right: 14% !important;
    top: 50% !important;
    transform: translateY(-50%) scale(1) !important;
    width: 20px !important;
    height: 20px !important;
    transition: all 0.25s cubic-bezier(0.310, -0.105, 0.430, 1.400) !important;
    z-index: 3 !important;
    filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3)) !important;
}

/* Hover State - Text Slides Out, Divider Fades, Icon Grows & Centers */
[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button:hover {
    opacity: 0.95 !important;
    box-shadow: 0 6px 30px rgba(239, 68, 68, 0.6) !important;
    transform: translateY(-2px) !important;
}

/* Text slides left and fades on hover */
[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button:hover p,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button:hover span:not(.icon-wrapper) {
    transform: translateX(-120%) !important;
    opacity: 0 !important;
}

/* Divider line fades out on hover */
[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button:hover::before {
    opacity: 0 !important;
}

/* Icon grows and centers on hover */
[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button:hover svg {
    right: 50% !important;
    transform: translate(50%, -50%) scale(1.6) !important;
    width: 32px !important;
    height: 32px !important;
    filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.5)) !important;
}

/* Active/Click State - Success Green Animation */
[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button:active {
    opacity: 1 !important;
    background: linear-gradient(135deg, #27ae60 0%, #229954 100%) !important;
    animation: successPulse 0.6s ease-out !important;
    transform: translateY(0) !important;
}

@keyframes successPulse {
    0% {
        box-shadow: 0 0 20px 0 rgba(0, 0, 0, 0.3);
    }
    50% {
        box-shadow: 0 0 40px 10px rgba(39, 174, 96, 0.6);
        transform: scale(1.05);
    }
    100% {
        box-shadow: 0 0 20px 0 rgba(39, 174, 96, 0.4);
        transform: scale(1);
    }
}

/* Success State - Check Icon Appears */
[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button:active svg {
    animation: checkmarkAppear 0.4s ease-out !important;
}

@keyframes checkmarkAppear {
    0% {
        transform: translate(50%, -50%) scale(0) rotate(-45deg);
        opacity: 0;
    }
    50% {
        transform: translate(50%, -50%) scale(1.3) rotate(0deg);
        opacity: 1;
    }
    100% {
        transform: translate(50%, -50%) scale(1.5) rotate(0deg);
        opacity: 1;
    }
}

/* Pulsing Glow Animation on Hover */
[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button:hover {
    animation: deleteGlow 1.5s ease-in-out infinite !important;
}

@keyframes deleteGlow {
    0%, 100% {
        box-shadow: 0 6px 25px rgba(239, 68, 68, 0.5);
    }
    50% {
        box-shadow: 0 8px 40px rgba(239, 68, 68, 0.8);
    }
}

/* Success/Error Toast Notifications */
.stAlert {
    animation: slideInRight 0.5s ease-out !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    border-radius: 12px !important;
}

@keyframes slideInRight {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

/* Success notification style */
.stSuccess {
    background: linear-gradient(135deg, #06b6d4, #0ea5e9) !important;
    border-left: 4px solid #0891b2 !important;
    animation: slideInRight 0.5s ease-out, pulse 2s infinite !important;
}

/* Error notification style */
.stError {
    background: linear-gradient(135deg, #ef4444, #dc2626) !important;
    border-left: 4px solid #b91c1c !important;
    animation: slideInRight 0.5s ease-out, shake 0.5s !important;
}

/* Warning notification style */
.stWarning {
    background: linear-gradient(135deg, #f59e0b, #d97706) !important;
    border-left: 4px solid #b45309 !important;
    animation: slideInRight 0.5s ease-out !important;
}

/* Info notification style */
.stInfo {
    background: linear-gradient(135deg, #06b6d4, #0284c7) !important;
    border-left: 4px solid #0369a1 !important;
    animation: slideInRight 0.5s ease-out !important;
}

@keyframes pulse {
    0%, 100% {
        box-shadow: 0 4px 12px rgba(6, 182, 212, 0.3);
    }
    50% {
        box-shadow: 0 4px 20px rgba(6, 182, 212, 0.6);
    }
}

@keyframes shake {
    0%, 100% { transform: translateX(0); }
    10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
    20%, 40%, 60%, 80% { transform: translateX(5px); }
}

/* ═══════════════════════════════════════════════════════════════════
   PREMIUM BUTTON DESIGN - ENTERPRISE GRADE
   ═══════════════════════════════════════════════════════════════════ */

/* All Buttons - Base Styling */
.stButton button {
    background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 50%, #0284c7 100%) !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 16px 40px !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: #ffffff !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 
        0 10px 30px rgba(14, 165, 233, 0.4),
        0 0 20px rgba(6, 182, 212, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    background-size: 200% 200% !important;
    animation: gradientShift 3s ease infinite !important;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Ripple Effect on Click */
.stButton button::after {
    content: '' !important;
    position: absolute !important;
    top: 50% !important;
    left: 50% !important;
    width: 0 !important;
    height: 0 !important;
    border-radius: 50% !important;
    background: rgba(255, 255, 255, 0.5) !important;
    transform: translate(-50%, -50%) !important;
    transition: width 0.6s, height 0.6s !important;
}

.stButton button:active::after {
    width: 300px !important;
    height: 300px !important;
}

/* Shine Effect Sweep */
.stButton button::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: -100% !important;
    width: 50% !important;
    height: 100% !important;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent) !important;
    transition: left 0.7s ease !important;
    transform: skewX(-20deg) !important;
}

.stButton button:hover::before {
    left: 150% !important;
}

/* Hover State - Levitation + Glow */
.stButton button:hover {
    transform: translateY(-6px) scale(1.05) !important;
    box-shadow: 
        0 15px 40px rgba(14, 165, 233, 0.6),
        0 0 50px rgba(6, 182, 212, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.5) !important;
    background: linear-gradient(135deg, #06b6d4 0%, #0ea5e9 50%, #0891b2 100%) !important;
}

/* Active/Click State */
.stButton button:active {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 
        0 8px 20px rgba(14, 165, 233, 0.5),
        inset 0 3px 6px rgba(0, 0, 0, 0.3) !important;
}

/* Primary Button (type="primary") - Extra Premium */
.stButton button[kind="primary"],
.stButton button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 50%, #0ea5e9 100%) !important;
    box-shadow: 
        0 12px 35px rgba(139, 92, 246, 0.5),
        0 0 30px rgba(99, 102, 241, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
}

.stButton button[kind="primary"]:hover,
.stButton button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%) !important;
    box-shadow: 
        0 18px 50px rgba(139, 92, 246, 0.7),
        0 0 60px rgba(99, 102, 241, 0.5),
        inset 0 1px 0 rgba(255, 255, 255, 0.5) !important;
    transform: translateY(-8px) scale(1.08) !important;
}

/* Secondary Button Styling */
.stButton button[kind="secondary"] {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.2) 0%, rgba(6, 182, 212, 0.2) 100%) !important;
    border: 2px solid #0ea5e9 !important;
    color: #0ea5e9 !important;
    backdrop-filter: blur(10px) !important;
}

.stButton button[kind="secondary"]:hover {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.4) 0%, rgba(6, 182, 212, 0.4) 100%) !important;
    border-color: #06b6d4 !important;
    color: #ffffff !important;
}

/* Download Button Special Styling */
.stDownloadButton button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    box-shadow: 
        0 10px 30px rgba(16, 185, 129, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
}

.stDownloadButton button:hover {
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    box-shadow: 
        0 15px 40px rgba(16, 185, 129, 0.6),
        0 0 40px rgba(5, 150, 105, 0.4) !important;
    transform: translateY(-6px) scale(1.05) !important;
}

/* Button Icon Enhancement */
.stButton button svg,
.stDownloadButton button svg {
    filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3)) !important;
    transition: all 0.3s ease !important;
}

.stButton button:hover svg,
.stDownloadButton button:hover svg {
    filter: drop-shadow(0 4px 8px rgba(255, 255, 255, 0.5)) !important;
    transform: scale(1.1) !important;
}

/* ═══════════════════════════════════════════════════════════════════
   WEBRTC CONTROLS - START BUTTON & DEVICE SELECTOR
   ═══════════════════════════════════════════════════════════════════ */

/* WebRTC Container Styling */
.streamlit-webrtc,
[data-testid="stWebRtc"],
.stWebRtc {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%) !important;
    border: 2px solid rgba(14, 165, 233, 0.3) !important;
    border-radius: 20px !important;
    padding: 30px !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: 
        0 10px 40px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
}

/* START Button - Premium Red Design */
button[aria-label*="start" i],
button[aria-label*="START" i],
button:has(span:contains("START")),
.streamlit-webrtc button:first-of-type,
[class*="webrtc"] button:first-of-type {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 50%, #b91c1c 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-weight: 800 !important;
    font-size: 1.1rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #ffffff !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 
        0 8px 25px rgba(239, 68, 68, 0.5),
        0 0 30px rgba(220, 38, 38, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    background-size: 200% 200% !important;
    animation: redPulse 2s ease infinite !important;
    min-width: 120px !important;
}

@keyframes redPulse {
    0%, 100% {
        background-position: 0% 50%;
        box-shadow: 
            0 8px 25px rgba(239, 68, 68, 0.5),
            0 0 30px rgba(220, 38, 38, 0.3);
    }
    50% {
        background-position: 100% 50%;
        box-shadow: 
            0 10px 35px rgba(239, 68, 68, 0.7),
            0 0 50px rgba(220, 38, 38, 0.5);
    }
}

/* START Button Shine Effect */
button[aria-label*="start" i]::before,
.streamlit-webrtc button:first-of-type::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: -100% !important;
    width: 50% !important;
    height: 100% !important;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent) !important;
    transition: left 0.7s ease !important;
    transform: skewX(-20deg) !important;
}

button[aria-label*="start" i]:hover::before,
.streamlit-webrtc button:first-of-type:hover::before {
    left: 150% !important;
}

/* START Button Hover */
button[aria-label*="start" i]:hover,
.streamlit-webrtc button:first-of-type:hover {
    transform: translateY(-4px) scale(1.05) !important;
    box-shadow: 
        0 12px 40px rgba(239, 68, 68, 0.7),
        0 0 60px rgba(220, 38, 38, 0.5),
        inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
    background: linear-gradient(135deg, #dc2626 0%, #b91c1c 50%, #991b1b 100%) !important;
}

/* START Button Active */
button[aria-label*="start" i]:active,
.streamlit-webrtc button:first-of-type:active {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 
        0 6px 20px rgba(239, 68, 68, 0.6),
        inset 0 3px 6px rgba(0, 0, 0, 0.3) !important;
}

/* SELECT DEVICE Button - Dark Premium Design */
button[aria-label*="device" i],
button[aria-label*="SELECT" i],
.streamlit-webrtc button:last-of-type,
[class*="webrtc"] button:last-of-type,
select[aria-label*="device" i] {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%) !important;
    border: 2px solid rgba(14, 165, 233, 0.5) !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: #0ea5e9 !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 
        0 6px 20px rgba(14, 165, 233, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    min-width: 180px !important;
}

/* Device Selector Animated Border */
button[aria-label*="device" i]::before,
.streamlit-webrtc button:last-of-type::before {
    content: '' !important;
    position: absolute !important;
    inset: -2px !important;
    border-radius: 12px !important;
    padding: 2px !important;
    background: linear-gradient(135deg, #0ea5e9, #06b6d4, #8b5cf6, #0ea5e9) !important;
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0) !important;
    -webkit-mask-composite: xor !important;
    mask-composite: exclude !important;
    opacity: 0 !important;
    transition: opacity 0.5s ease !important;
    animation: borderRotate 3s linear infinite !important;
    background-size: 300% 300% !important;
}

button[aria-label*="device" i]:hover::before,
.streamlit-webrtc button:last-of-type:hover::before {
    opacity: 1 !important;
}

/* Device Selector Hover */
button[aria-label*="device" i]:hover,
.streamlit-webrtc button:last-of-type:hover {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.2) 0%, rgba(6, 182, 212, 0.2) 100%) !important;
    border-color: #06b6d4 !important;
    color: #ffffff !important;
    transform: translateY(-4px) scale(1.03) !important;
    box-shadow: 
        0 10px 30px rgba(14, 165, 233, 0.5),
        0 0 40px rgba(6, 182, 212, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
}

/* Device Selector Active */
button[aria-label*="device" i]:active,
.streamlit-webrtc button:last-of-type:active {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 
        0 6px 20px rgba(14, 165, 233, 0.4),
        inset 0 2px 4px rgba(0, 0, 0, 0.2) !important;
}

/* WebRTC Button Container Layout */
.streamlit-webrtc > div,
[class*="webrtc"] > div {
    display: flex !important;
    gap: 16px !important;
    align-items: center !important;
    justify-content: center !important;
    flex-wrap: wrap !important;
    margin: 20px 0 !important;
}

/* ═══════════════════════════════════════════════════════════════════
   SAFE TARGETED STYLING - WEBCAM TAB ONLY
   ═══════════════════════════════════════════════════════════════════ */

/* Premium Video Container Styling */
[data-baseweb="tab-panel"]:nth-child(2) video,
[data-baseweb="tab-panel"]:nth-child(2) .stVideo,
[data-baseweb="tab-panel"]:nth-child(2) iframe {
    border-radius: 20px !important;
    border: 3px solid rgba(14, 165, 233, 0.4) !important;
    box-shadow: 
        0 15px 50px rgba(0, 0, 0, 0.5),
        0 0 40px rgba(14, 165, 233, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.9) 100%) !important;
    backdrop-filter: blur(20px) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
}

/* Video Container Hover Effect */
[data-baseweb="tab-panel"]:nth-child(2) video:hover,
[data-baseweb="tab-panel"]:nth-child(2) .stVideo:hover,
[data-baseweb="tab-panel"]:nth-child(2) iframe:hover {
    border-color: rgba(14, 165, 233, 0.7) !important;
    box-shadow: 
        0 20px 60px rgba(0, 0, 0, 0.6),
        0 0 60px rgba(14, 165, 233, 0.5),
        inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    transform: scale(1.01) !important;
}

/* Animated Border for Video Container */
[data-baseweb="tab-panel"]:nth-child(2) video::before,
[data-baseweb="tab-panel"]:nth-child(2) .stVideo::before {
    content: '' !important;
    position: absolute !important;
    inset: -3px !important;
    border-radius: 20px !important;
    background: linear-gradient(45deg, #0ea5e9, #06b6d4, #8b5cf6, #0ea5e9) !important;
    background-size: 300% 300% !important;
    z-index: -1 !important;
    opacity: 0.5 !important;
    animation: videoBorderGlow 4s linear infinite !important;
    filter: blur(10px) !important;
}

@keyframes videoBorderGlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Webcam Container Background */
[data-baseweb="tab-panel"]:nth-child(2) {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.3) 0%, rgba(30, 41, 59, 0.3) 100%) !important;
    border-radius: 16px !important;
    padding: 20px !important;
}

/* Target only buttons in tab2 (Live Webcam) */
[data-baseweb="tab-panel"]:nth-child(2) button:not([data-testid*="stFileUploader"]):not(.stButton button) {
    position: relative !important;
    overflow: visible !important;
}

/* START Button - Only in webcam section */
[data-baseweb="tab-panel"]:nth-child(2) button[aria-label*="start" i],
[data-baseweb="tab-panel"]:nth-child(2) button:first-of-type:not([data-testid*="baseButton"]) {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-weight: 800 !important;
    font-size: 0.95rem !important;
    letter-spacing: 1.5px !important;
    color: #ffffff !important;
    box-shadow: 
        0 6px 20px rgba(239, 68, 68, 0.5),
        0 0 25px rgba(220, 38, 38, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    animation: startButtonPulse 2s ease-in-out infinite !important;
}

@keyframes startButtonPulse {
    0%, 100% {
        box-shadow: 
            0 6px 20px rgba(239, 68, 68, 0.5),
            0 0 25px rgba(220, 38, 38, 0.3);
    }
    50% {
        box-shadow: 
            0 8px 30px rgba(239, 68, 68, 0.7),
            0 0 40px rgba(220, 38, 38, 0.5);
    }
}

[data-baseweb="tab-panel"]:nth-child(2) button[aria-label*="start" i]:hover,
[data-baseweb="tab-panel"]:nth-child(2) button:first-of-type:not([data-testid*="baseButton"]):hover {
    transform: translateY(-3px) scale(1.03) !important;
    box-shadow: 
        0 10px 35px rgba(239, 68, 68, 0.7),
        0 0 50px rgba(220, 38, 38, 0.5) !important;
    background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
}

[data-baseweb="tab-panel"]:nth-child(2) button[aria-label*="start" i]:active,
[data-baseweb="tab-panel"]:nth-child(2) button:first-of-type:not([data-testid*="baseButton"]):active {
    transform: translateY(-1px) scale(1.01) !important;
}

/* SELECT DEVICE - Text button styling */
[data-baseweb="tab-panel"]:nth-child(2) button:last-of-type:not([data-testid*="baseButton"]),
[data-baseweb="tab-panel"]:nth-child(2) button[aria-label*="device" i] {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.8) 100%) !important;
    border: 2px solid rgba(14, 165, 233, 0.4) !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 1.2px !important;
    color: #0ea5e9 !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 
        0 4px 15px rgba(14, 165, 233, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
}

/* Animated border effect for SELECT DEVICE */
[data-baseweb="tab-panel"]:nth-child(2) button:last-of-type:not([data-testid*="baseButton"])::after,
[data-baseweb="tab-panel"]:nth-child(2) button[aria-label*="device" i]::after {
    content: '' !important;
    position: absolute !important;
    inset: -3px !important;
    border-radius: 10px !important;
    background: linear-gradient(45deg, #0ea5e9, #06b6d4, #8b5cf6, #0ea5e9) !important;
    background-size: 300% 300% !important;
    z-index: -1 !important;
    opacity: 0 !important;
    transition: opacity 0.4s ease !important;
    animation: borderGlow 3s linear infinite !important;
    filter: blur(8px) !important;
}

@keyframes borderGlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

[data-baseweb="tab-panel"]:nth-child(2) button:last-of-type:not([data-testid*="baseButton"]):hover::after,
[data-baseweb="tab-panel"]:nth-child(2) button[aria-label*="device" i]:hover::after {
    opacity: 1 !important;
}

[data-baseweb="tab-panel"]:nth-child(2) button:last-of-type:not([data-testid*="baseButton"]):hover,
[data-baseweb="tab-panel"]:nth-child(2) button[aria-label*="device" i]:hover {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.2) 0%, rgba(6, 182, 212, 0.2) 100%) !important;
    border-color: #06b6d4 !important;
    color: #ffffff !important;
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 
        0 8px 25px rgba(14, 165, 233, 0.5),
        0 0 35px rgba(6, 182, 212, 0.3) !important;
}

[data-baseweb="tab-panel"]:nth-child(2) button:last-of-type:not([data-testid*="baseButton"]):active,
[data-baseweb="tab-panel"]:nth-child(2) button[aria-label*="device" i]:active {
    transform: translateY(-1px) scale(1.01) !important;
}

/* Progress bar animation */
.stProgress > div > div {
    transition: width 0.5s ease-out !important;
}

/* Hide all Streamlit running/status indicators */
[data-testid="stStatusWidget"] {
    display: none !important;
    visibility: hidden !important;
}

.stApp > header {
    display: none !important;
}

[data-testid="stAppViewBlockContainer"] > div:first-child {
    display: none !important;
}

/* Hide running spinner and text */
div[data-testid="stSpinner"],
div[data-testid="stStatus"],
.stSpinner,
[class*="StatusWidget"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
}

/* Hide "Running" text specifically */
div:has(> div[data-testid="stSpinner"]) {
    display: none !important;
}

/* Hide status messages */
[data-testid="stNotification"],
[data-testid="stToast"] {
    display: none !important;
}


/* Smooth container transitions */
[data-testid="stVerticalBlock"] > div {
    animation: fadeInUp 0.4s ease-out !important;
}

/* Hover effect on cards/containers */
[data-testid="stVerticalBlock"] {
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
}

/* Input field focus effects */
input:focus, textarea:focus, select:focus {
    transform: scale(1.02) !important;
    box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.2) !important;
    transition: all 0.3s ease !important;
}

/* Slider smooth animation */
[data-testid="stSlider"] {
    transition: all 0.3s ease !important;
}

/* Expander smooth animation */
[data-testid="stExpander"] {
    transition: all 0.3s ease !important;
}

[data-testid="stExpander"][aria-expanded="true"] {
    animation: expandDown 0.3s ease-out !important;
}

@keyframes expandDown {
    from {
        opacity: 0;
        max-height: 0;
    }
    to {
        opacity: 1;
        max-height: 1000px;
    }
}

    button[data-testid="baseButton-header"][title="Settings"],
    [data-testid="stHeaderActionElements"] button[title="Settings"],
    [data-testid="stHeaderActionElements"] button[aria-label="Settings"] {
        display: none !important;
    }
    
    /* Hide Settings menu item specifically */
    [role="menuitem"]:has-text("Settings"),
    [role="option"]:has-text("Settings"),
    li:has-text("Settings") {
        display: none !important;
    }
    
    /* Additional hiding for any Settings text */
    *:contains("Settings") {
        display: none !important;
    }
    
    /* Hide content when collapsed */
    section[data-testid="stSidebar"] > div {
        background: transparent;
        padding-top: 60px;
        opacity: 0;
        transform: translateX(-20px);
        transition: all 0.4s ease;
        pointer-events: none;
    }
    
    /* Expand sidebar on hover */
    section[data-testid="stSidebar"]:hover {
        width: 300px !important;
        min-width: 300px !important;
    }
    
    /* Show content when expanded */
    section[data-testid="stSidebar"]:hover > div {
        opacity: 1;
        transform: translateX(0);
        pointer-events: auto;
    }
    
    /* Hide Streamlit's default collapse button */
    button[kind="header"] {
        display: none !important;
    }
    
    /* Sidebar Header Styling - Clean, No Glow */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #0ea5e9 !important;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        transition: all 0.3s ease;
    }
    
    section[data-testid="stSidebar"] h1:hover,
    section[data-testid="stSidebar"] h2:hover,
    section[data-testid="stSidebar"] h3:hover {
        color: #06b6d4 !important;
        transform: translateX(5px);
    }
    
    /* Sidebar Text Styling */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #cbd5e1 !important;
        font-weight: 500;
        transition: color 0.3s ease;
    }
    
    section[data-testid="stSidebar"] strong {
        color: #06b6d4 !important;
        font-weight: 700;
    }
    
    /* Expander Styling - Clean, No Glow */
    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        background: rgba(14, 165, 233, 0.1);
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 8px;
        color: #0ea5e9 !important;
        font-weight: 600;
        transition: all 0.3s ease;
        margin: 10px 0;
    }
    
    section[data-testid="stSidebar"] .streamlit-expanderHeader:hover {
        background: rgba(14, 165, 233, 0.2);
        border-color: #06b6d4;
        transform: translateX(5px);
    }
    
    /* Slider Styling - Clean */
    section[data-testid="stSidebar"] .stSlider > div > div > div {
        background: linear-gradient(90deg, #0ea5e9, #06b6d4);
    }
    
    section[data-testid="stSidebar"] .stSlider > div > div > div > div {
        background: #06b6d4;
    }
    
    /* Info/Success/Warning Boxes in Sidebar */
    section[data-testid="stSidebar"] .stAlert {
        background: rgba(14, 165, 233, 0.1);
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 8px;
        color: #cbd5e1 !important;
    }
    
    /* Caption Styling */
    section[data-testid="stSidebar"] .stCaptionContainer {
        color: #94a3b8 !important;
        font-size: 0.85rem;
    }
</style>

<script>
function removeSettingsFromMenu() {
    // More aggressive removal of Settings menu items
    const hideSettings = () => {
        // Target all possible Settings elements
        document.querySelectorAll(
            'button[title="Settings"], ' +
            'button[aria-label="Settings"], ' +
            '[role="menuitem"], ' + 
            'li[role="menuitem"], ' +
            '[data-testid*="Settings"], ' +
            'button[data-testid*="Settings"]'
        ).forEach(element => {
            // Check text content
            if (element.textContent && element.textContent.includes('Settings')) {
                // Remove from DOM completely
                element.remove();
            }
        });
        
        // Also target any list items containing Settings
        document.querySelectorAll('li').forEach(li => {
            if (li.textContent && li.textContent.trim() === 'Settings') {
                li.remove();
            }
        });
        
        // Target any element with Settings in aria-label
        document.querySelectorAll('[aria-label*="Settings"]').forEach(el => {
            el.remove();
        });
    };
    
    // Run immediately and after delay
    hideSettings();
    setTimeout(hideSettings, 100);
    setTimeout(hideSettings, 500);
    setTimeout(hideSettings, 1000);
}

// Run on page load
document.addEventListener('DOMContentLoaded', removeSettingsFromMenu);
// Observe DOM changes to hide dynamically added "Settings" entries
if (typeof MutationObserver !== 'undefined') {
    const __smpObserver = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.addedNodes.length > 0) {
                removeSettingsFromMenu();
            }
        });
    });
    __smpObserver.observe(document.body, { childList: true, subtree: true });
}
// Run periodically to catch dynamically loaded menus
setInterval(removeSettingsFromMenu, 1000);

// Also intercept clicks on the menu button to remove Settings after menu opens
document.addEventListener('click', function(e) {
    // If clicked on menu button (three dots)
    if (e.target && (e.target.matches('[data-testid="stToolbar"] button') || 
                     e.target.closest('[data-testid="stToolbar"] button'))) {
        setTimeout(removeSettingsFromMenu, 50);
        setTimeout(removeSettingsFromMenu, 150);
        setTimeout(removeSettingsFromMenu, 300);
    }
}, true);
</script>
"""

# Apply the combined style + script block
st.markdown(hide_settings_style, unsafe_allow_html=True)

# Initialize session state for analytics
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'total_detections' not in st.session_state:
    st.session_state.total_detections = 0
if 'total_images_processed' not in st.session_state:
    st.session_state.total_images_processed = 0
if 'session_start_time' not in st.session_state:
    st.session_state.session_start_time = datetime.now()

def draw_detections(image, detections, line_thickness=3):
    """Draw optimized bounding boxes and labels with color coding by material type"""
    img = image.copy()
    
    # Define colors for different materials (BGR format - Blue, Green, Red)
    color_map = {
        'plastic': (0, 255, 0),      # Bright green for plastic
        'metal': (255, 0, 0),        # Blue for metal (255 in Blue channel)
        'wood': (0, 165, 255),       # Orange for wood
        'concrete': (128, 128, 128), # Gray for concrete
        'default': (0, 255, 255)     # Yellow for unknown
    }
    
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection['bbox'])
        confidence = detection['confidence']
        class_name = detection['class_name']

        # Get color based on material type
        color = color_map.get(class_name.lower(), color_map['default'])

        # Draw thick bounding box for visibility
        cv2.rectangle(img, (x1, y1), (x2, y2), color, max(line_thickness, 3))

        # Draw optimized label
        label = f"{class_name}: {confidence:.2f}"
        font_scale = 0.7
        thickness = 2
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Ensure label is within image bounds
        label_y = max(y1 - 10, label_height + 10)
        
        # Draw label background with material color
        cv2.rectangle(img, (x1, label_y - label_height - 5), (x1 + label_width + 5, label_y + 5), color, -1)
        
        # Draw label text in black for contrast
        cv2.putText(img, label, (x1 + 2, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    return img

def log_detection(image_name, num_detections, confidence_avg):
    """Log detection to session history"""
    st.session_state.detection_history.append({
        'timestamp': datetime.now(),
        'image_name': image_name,
        'detections': num_detections,
        'avg_confidence': confidence_avg
    })
    st.session_state.total_detections += num_detections
    st.session_state.total_images_processed += 1

def process_webcam_frame(frame, model, confidence, line_thickness):
    """Process webcam frame for detection with robust error handling"""
    if model is None:
        return frame

    try:
        # Ensure frame is in correct format
        if frame is None or frame.size == 0:
            return frame
            
        # Make sure frame is uint8
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        # Ensure frame has correct shape (H, W, 3)
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return frame

        # Run detection with error handling
        detections = detect_plastic(frame, model, confidence)

        # Draw detections if any found
        if detections and len(detections) > 0:
            frame = draw_detections(frame, detections, line_thickness)

        return frame
        
    except Exception as e:
        # Log error but don't crash - return original frame
        print(f"Webcam processing error: {e}")
        return frame

# Main UI

st.markdown('<h1 class="main-header">SMART MARINE PROJECT</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Plastic Waste Detection</p>', unsafe_allow_html=True)

# Load model
model, model_status = load_model()

# Professional Enterprise Sidebar
with st.sidebar:
    # Professional Header
    st.markdown("""
    <div class="sidebar-header">
        <h2 class="sidebar-title">🌊 MARINE AI</h2>
        <p class="sidebar-subtitle">Advanced Plastic Detection System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Detection Configuration Section
    st.markdown("""
    <div class="section-header">
        Detection Configuration
    </div>
    """, unsafe_allow_html=True)
    
    # Professional confidence threshold control
    st.markdown('<p class="control-label">Detection Sensitivity</p>', unsafe_allow_html=True)
    confidence_threshold = st.slider(
        "Detection Sensitivity",
        min_value=0.01,
        max_value=1.0,
        value=0.01,  # Ultra-low default for maximum detection in dense scenes
        step=0.01,
        help="Lower = more sensitive (detects more bottles). Set to 0.01 for maximum detection in marine debris",
        label_visibility="collapsed"
    )
    
    # Display current value professionally
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<p style="color: #94a3b8; font-size: 0.8rem;">Current Threshold</p>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<p class="control-value">{confidence_threshold:.3f}</p>', unsafe_allow_html=True)
    
    # Professional line thickness control
    st.markdown('<p class="control-label">Annotation Thickness</p>', unsafe_allow_html=True)
    line_thickness = st.slider(
        "Annotation Thickness",
        min_value=1,
        max_value=10,
        value=3,
        help="Bounding box line thickness",
        label_visibility="collapsed"
    )
    
    # System Optimizations Section
    st.markdown("""
    <div class="section-header">
        AI Optimizations
    </div>
    """, unsafe_allow_html=True)
    
    # Professional optimization display
    optimizations = [
        "YOLOv8 Neural Network",
        "Marine-Specific Training", 
        "15-Layer Filtering System",
        "Real-Time Processing",
        "Edge Computing Ready"
    ]
    
    for opt in optimizations:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">✓ {opt}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # System Status Section
    st.markdown("""
    <div class="section-header">
        System Status
    </div>
    """, unsafe_allow_html=True)
    
    # Professional status display
    st.markdown(f"""
    <div class="status-section">
        <div class="status-header">🔋 Core Systems</div>
        <div class="status-item">YOLOv8 Neural Engine</div>
        <div class="status-item">Filtering Pipeline</div>
        <div class="status-item">Marine AI Models</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance Metrics
    st.markdown("""
    <div class="section-header">
        Performance Metrics
    </div>
    """, unsafe_allow_html=True)
    
    # Professional metrics display
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">94.2%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Speed</div>
            <div class="metric-value">80 FPS</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Professional footer
    st.markdown("""
    <div style="margin-top: 2rem; padding: 1rem; background: linear-gradient(135deg, #1e293b 0%, #334155 100%); border-radius: 8px; text-align: center;">
        <p style="color: #64748b; font-size: 0.7rem; margin: 0;">SMART MARINE PROJECT</p>
        <p style="color: #0ea5e9; font-size: 0.8rem; font-weight: 600; margin: 0;">Enterprise Edition v2.0</p>
    </div>
    """, unsafe_allow_html=True)

    if not WEBCAM_AVAILABLE:
        st.warning("📹 Webcam features not available. Install: pip install streamlit-webrtc av")

# Main content area
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📸 Single Image", "📹 Live Webcam", "📚 Batch Upload", "📊 Analytics", "🚤 Autonomous Mode"])

with tab1:
    st.markdown(f"""
    <div class="header-container">
        <h2 class="tab-content-header">Single Image Detection</h2>
        <div class="boat-float"></div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'bmp', 'webp'])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if model is not None:
            if st.button("🔍 Detect Plastic Waste", type="primary"):
                # Use less aggressive filtering for single images (no faces expected)
                detections = detect_plastic(np.array(image), model, confidence_threshold, filter_faces=False)

                if detections:
                    # Draw detections on image
                    annotated_image = draw_detections(np.array(image), detections, line_thickness)
                    st.image(annotated_image, caption="Detection Results", use_container_width=True)

                    # Log detection
                    avg_conf = sum(d['confidence'] for d in detections) / len(detections)
                    log_detection(uploaded_file.name, len(detections), avg_conf)

                    # Display results
                    st.success(f"✅ Found {len(detections)} plastic objects!")

                    # Show detailed results
                    with st.expander("📋 Detailed Results"):
                        for i, det in enumerate(detections, 1):
                            st.write(f"**Object {i}:**")
                            st.write(f"- Label: **{det['class_name']}**")
                            st.write(f"- Confidence: {det['confidence']:.2f}")
                            st.write(f"- Bounding Box: {det['bbox']}")
                            st.write("---")
                else:
                    st.info("ℹ️ No plastic objects detected in this image.")
                    log_detection(uploaded_file.name, 0, 0.0)
        else:
            st.error("❌ Model not loaded. Please check the model file.")

with tab2:
    st.markdown('''
    <div class="header-container">
        <h2 class="tab-content-header">Live Webcam Detection</h2>
        <div class="boat-float"></div>
    </div>
    ''', unsafe_allow_html=True)

    # Load webcam-specific model (original YOLOv5 for better detection)
    webcam_model, webcam_status = load_webcam_model()
    
    if not WEBCAM_AVAILABLE:
        st.error("❌ Webcam features not available.")
        st.info("📦 Install required packages:")
        st.code("pip install streamlit-webrtc av", language="bash")
        st.warning("⚠️ Note: Webcam requires HTTPS in production. Use localhost for development.")
    elif webcam_model is None:
        st.error(f"❌ Webcam model not loaded: {webcam_status}")
        st.info("💡 Using original YOLOv5 model for better webcam detection")
    else:
        # Detection is always enabled
        detection_enabled = True

        # WebRTC Configuration
        RTC_CONFIGURATION = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]}
            ]
        })

        # Global frame counter and detection cache for smoothing
        frame_counter = [0]
        last_detections = [None]
        detection_hold_frames = [0]
        
        def video_frame_callback(frame):
            """Optimized video frame processing with detection smoothing"""
            try:
                # Convert frame to numpy array
                img = frame.to_ndarray(format="bgr24")
                frame_counter[0] += 1
                
                # Apply detection if enabled
                if detection_enabled and webcam_model is not None:
                    try:
                        # Run detection every 3 frames for stability
                        if frame_counter[0] % 3 == 0:
                            # Use webcam model (original YOLOv5) for better detection
                            # Ultra-low confidence threshold for webcam (0.01 for all orientations)
                            detections = detect_plastic(img, webcam_model, max(confidence_threshold, 0.01), filter_faces=False)
                            
                            # Cache detections if found
                            if detections:
                                last_detections[0] = detections
                                detection_hold_frames[0] = 10  # Hold for 10 frames
                        
                        # Draw cached detections for smooth display
                        if last_detections[0] and detection_hold_frames[0] > 0:
                            img = draw_detections(img, last_detections[0], line_thickness)
                            detection_hold_frames[0] -= 1
                        elif detection_hold_frames[0] <= 0:
                            last_detections[0] = None  # Clear cache
                        
                    except Exception as e:
                        print(f"Detection error: {e}")
                
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            except Exception as e:
                print(f"Frame processing error: {e}")
                return frame

        # WebRTC Streamer
        webrtc_ctx = webrtc_streamer(
            key="smart-marine-webcam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={
                "video": {
                    "width": {"min": 640, "ideal": 1280, "max": 1920},
                    "height": {"min": 480, "ideal": 720, "max": 1080},
                    "frameRate": {"min": 15, "ideal": 30, "max": 60}
                },
                "audio": False
            },
            async_processing=True,
        )

        # Status display
        if webrtc_ctx.state.playing:
            st.success("🎥 Webcam active!")
        else:
            st.warning("📹 Click 'START' to begin webcam stream")
            
        # Troubleshooting tips
        with st.expander("🔧 Webcam Troubleshooting"):
            st.markdown("""
            **If webcam is not working:**
            
            1. **Browser Permissions**: Allow camera access when prompted
            2. **HTTPS Required**: Use `https://` or `localhost` (not IP addresses)
            3. **Browser Compatibility**: Chrome/Safari work best
            4. **Firewall**: Ensure ports 8501 and WebRTC ports are open
            5. **Multiple Tabs**: Close other tabs using the camera
            
            **For mobile devices:**
            - Use HTTPS (required for camera access)
            - Try different browsers (Chrome, Safari)
            - Ensure good internet connection
            """)

with tab3:
    st.markdown('''
    <div class="header-container">
        <h2 class="tab-content-header">Batch Upload & Processing</h2>
        <div class="boat-float"></div>
    </div>
    ''', unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Choose multiple images...", type=['jpg', 'jpeg', 'png', 'bmp', 'webp'], accept_multiple_files=True)

    if uploaded_files:
        st.write(f"📁 {len(uploaded_files)} files uploaded")

        if model is not None:
            if st.button("🔍 Process All Images", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                all_results = []

                for i, uploaded_file in enumerate(uploaded_files):
                    # Update progress
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {uploaded_file.name}...")

                    # Process image
                    image = Image.open(uploaded_file)
                    # Use less aggressive filtering for batch images (no faces expected)
                    detections = detect_plastic(np.array(image), model, confidence, filter_faces=False)

                    # Log detection
                    if detections:
                        avg_conf = sum(d['confidence'] for d in detections) / len(detections)
                        log_detection(uploaded_file.name, len(detections), avg_conf)
                    else:
                        log_detection(uploaded_file.name, 0, 0.0)

                    all_results.append({
                        'filename': uploaded_file.name,
                        'detections': detections
                    })

                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")

                # Display results
                st.success(f"✅ Processed {len(uploaded_files)} images!")

                # Summary statistics
                total_detections = sum(len(result['detections']) for result in all_results)
                st.info(f"📊 Total plastic objects detected: {total_detections}")

                # Show results for each image
                for result in all_results:
                    with st.expander(f"📷 {result['filename']} - {len(result['detections'])} detections"):
                        if result['detections']:
                            # Show annotated image
                            image = Image.open([f for f in uploaded_files if f.name == result['filename']][0])
                            annotated_image = draw_detections(np.array(image), result['detections'], line_thickness)
                            st.image(annotated_image, caption=f"Detection Results - {result['filename']}", use_container_width=True)

                            # Show detection details
                            for det in result['detections']:
                                st.write(f"- {det.get('class_name', 'plastic')}: {det['confidence']:.2f}")
                        else:
                            st.info("No plastic objects detected")

        else:
            st.error("❌ Model not loaded. Please check the model file.")

with tab4:
    st.markdown('''
    <div class="header-container">
        <h2 class="tab-content-header">Analytics Dashboard</h2>
        <div class="boat-float"></div>
    </div>
    ''', unsafe_allow_html=True)

    # Session Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #0ea5e9, #06b6d4); padding: 20px; border-radius: 12px; text-align: center; box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);'>
            <h3 style='color: white; margin: 0; font-size: 2.5rem;'>🖼️</h3>
            <h2 style='color: white; margin: 10px 0;'>{}</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 0;'>Images Processed</p>
        </div>
        """.format(st.session_state.total_images_processed), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #06b6d4, #0891b2); padding: 20px; border-radius: 12px; text-align: center; box-shadow: 0 4px 12px rgba(6, 182, 212, 0.3);'>
            <h3 style='color: white; margin: 0; font-size: 2.5rem;'>🎯</h3>
            <h2 style='color: white; margin: 10px 0;'>{}</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 0;'>Total Detections</p>
        </div>
        """.format(st.session_state.total_detections), unsafe_allow_html=True)
    
    with col3:
        avg_per_image = st.session_state.total_detections / st.session_state.total_images_processed if st.session_state.total_images_processed > 0 else 0
        st.markdown("""
        <div style='background: linear-gradient(135deg, #0891b2, #0e7490); padding: 20px; border-radius: 12px; text-align: center; box-shadow: 0 4px 12px rgba(8, 145, 178, 0.3);'>
            <h3 style='color: white; margin: 0; font-size: 2.5rem;'>📊</h3>
            <h2 style='color: white; margin: 10px 0;'>{:.1f}</h2>
        </div>
        """.format(avg_per_image), unsafe_allow_html=True)
    
    with col4:
        session_duration = (datetime.now() - st.session_state.session_start_time).total_seconds() / 60
        st.markdown("""
        <div class="metric-container">
            <p class="metric-label">Session Time</p>
            <p class="metric-value">{:.0f}m</p>
        </div>
        """.format(session_duration), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Section
    if st.session_state.detection_history:
        # Create DataFrame from history
        df = pd.DataFrame(st.session_state.detection_history)
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Detection Timeline
            st.markdown("### 📈 Detection Timeline")
            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['detections'],
                mode='lines+markers',
                name='Detections',
                line=dict(color='#0ea5e9', width=3),
                marker=dict(size=8, color='#06b6d4'),
                fill='tozeroy',
                fillcolor='rgba(14, 165, 233, 0.2)'
            ))
            fig_timeline.update_layout(
                plot_bgcolor='rgba(15, 23, 42, 0.8)',
                paper_bgcolor='rgba(15, 23, 42, 0.8)',
                font=dict(color='#cbd5e1'),
                xaxis=dict(
                    title='Time',
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    showgrid=True
                ),
                yaxis=dict(
                    title='Number of Detections',
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    showgrid=True
                ),
                hovermode='x unified',
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        with col_right:
            # Confidence Distribution
            st.markdown("### 🎯 Confidence Distribution")
            confidences = [entry['avg_confidence'] for entry in st.session_state.detection_history if entry['avg_confidence'] > 0]
            
            if confidences:
                fig_conf = go.Figure()
                fig_conf.add_trace(go.Histogram(
                    x=confidences,
                    nbinsx=20,
                    marker=dict(
                        color='#0ea5e9',
                        line=dict(color='#06b6d4', width=1)
                    ),
                    name='Confidence'
                ))
                fig_conf.update_layout(
                    plot_bgcolor='rgba(15, 23, 42, 0.8)',
                    paper_bgcolor='rgba(15, 23, 42, 0.8)',
                    font=dict(color='#cbd5e1'),
                    xaxis=dict(
                        title='Confidence Score',
                        gridcolor='rgba(255, 255, 255, 0.1)',
                        showgrid=True
                    ),
                    yaxis=dict(
                        title='Frequency',
                        gridcolor='rgba(255, 255, 255, 0.1)',
                        showgrid=True
                    ),
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig_conf, use_container_width=True)
            else:
                st.info("No confidence data available yet")

        # Detection History Table
        st.markdown("### 📋 Recent Detection History")
        
        # Format the dataframe for display
        display_df = df.copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
        display_df['avg_confidence'] = display_df['avg_confidence'].apply(lambda x: f"{x:.2f}" if x > 0 else "N/A")
        display_df = display_df.rename(columns={
            'timestamp': 'Time',
            'image_name': 'Image',
            'detections': 'Detections',
            'avg_confidence': 'Avg Confidence'
        })
        
        # Show last 10 entries
        st.dataframe(
            display_df.tail(10).iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

        # Download Options
        st.markdown("### 💾 Export Data")
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            # Export as CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_export2:
            # Export as JSON
            json_str = df.to_json(orient='records', date_format='iso')
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

        # Clear History Button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Detection History", type="secondary"):
            st.session_state.detection_history = []
            st.session_state.total_detections = 0
            st.session_state.total_images_processed = 0
            st.session_state.session_start_time = datetime.now()
            st.success("✅ History cleared!")
            st.rerun()

    else:
        # No data yet
        st.info("""
        ### 📊 No Data Yet
        
        Start detecting plastic waste to see analytics here!
        
        **Available Analytics:**
        - 📈 Detection timeline
        - 🎯 Confidence distribution
        - 📋 Detection history table
        - 💾 Export data (CSV/JSON)
        
        Process images in the **Single Image** or **Batch Upload** tabs to populate this dashboard.
        """)

# ============================================================================
# TAB 5: AUTONOMOUS VESSEL MODE
# ============================================================================

with tab5:
    st.markdown('''
    <div class="header-container">
        <h2 class="tab-content-header">🚤 Autonomous Vessel Navigation</h2>
        <div class="boat-float"></div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Check if vessel modules are available
    if not VESSEL_MODULES_AVAILABLE or not MAPPING_AVAILABLE:
        st.error("""
        **🚨 Vessel modules not fully available**
        
        Missing dependencies for autonomous vessel functionality.
        """)
        
        st.markdown("### 📦 Install Required Dependencies")
        st.code("""
        pip install streamlit-folium folium geopy pyyaml
        """)
        
        if not VESSEL_MODULES_AVAILABLE:
            st.warning("**vessel_modules** directory not found or incomplete")
        if not MAPPING_AVAILABLE:
            st.warning("**Mapping libraries** not installed (folium, geopy)")
        
        st.info("""
        **Alternative:** Use the simplified simulation below while dependencies are installed.
        """)
        
        # Fallback to simple simulation
        st.markdown("### 🌊 Simple Vessel Simulator")
        
        # Initialize simple simulation state
        if 'vessel_x' not in st.session_state:
            st.session_state.vessel_x = 50
            st.session_state.vessel_y = 50
            st.session_state.plastics_collected = 0
            st.session_state.vessel_heading = 0
        
        # Simple controls
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Turn Left"):
                st.session_state.vessel_heading = (st.session_state.vessel_heading - 45) % 360
                st.rerun()
        with col2:
            if st.button("⬆️ Forward"):
                st.session_state.vessel_y = max(10, st.session_state.vessel_y - 10)
                st.rerun()
        with col3:
            if st.button("➡️ Turn Right"):
                st.session_state.vessel_heading = (st.session_state.vessel_heading + 45) % 360
                st.rerun()
        
        st.metric("Position", f"({st.session_state.vessel_x}, {st.session_state.vessel_y})")
        st.metric("Heading", f"{st.session_state.vessel_heading}°")
        
    else:
        # Full autonomous vessel implementation
        st.markdown("### 🗺️ GPS Navigation & Autonomous Collection")
        
        # Initialize vessel simulator
        if 'vessel_simulator' not in st.session_state:
            try:
                # Load configuration
                config_path = os.path.join(os.path.dirname(__file__), 'vessel_modules', 'vessel_config.yaml')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                else:
                    # Default configuration - DEEP OCEAN WATER ONLY
                    config = {
                        'map_center_lat': 12.9500,  # Deep Bay of Bengal - NO LAND
                        'map_center_lon': 80.3500,  # Far from coast - PURE OCEAN
                        'boat_speed_mps': 1.5,
                        'detection_range_m': 50,
                        'collection_range_m': 2,
                        'spawn_plastic_count': 8  # Fewer for realistic ocean scenario
                    }
                
                st.session_state.vessel_simulator = VesselSimulator(config)
                st.session_state.collection_counter = CollectionCounter()
                st.session_state.autopilot_active = False
                st.session_state.last_update_time = time.time()
                
            except Exception as e:
                st.error(f"Failed to initialize vessel simulator: {e}")
                st.stop()
        
        # Mode selection
        mode = st.selectbox(
            "🎮 Select Operation Mode",
            ["🖥️ Simulation Mode", "🔧 Hardware Mode (Raspberry Pi)"],
            help="Simulation mode works without hardware. Hardware mode requires Raspberry Pi setup."
        )
        
        if mode == "🖥️ Simulation Mode":
            
            # Simulation controls
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("▶️ Start Autopilot", type="primary"):
                    st.session_state.autopilot_active = True
                    st.success("🤖 Autopilot activated!")
                    st.rerun()
            
            with col2:
                if st.button("⏸️ Stop Autopilot"):
                    st.session_state.autopilot_active = False
                    st.info("⏹️ Autopilot stopped")
                    st.rerun()
            
            with col3:
                if st.button("🔄 Reset Mission"):
                    st.session_state.vessel_simulator = VesselSimulator(st.session_state.vessel_simulator.config)
                    st.session_state.collection_counter = CollectionCounter()
                    st.session_state.autopilot_active = False
                    st.success("🔄 Mission reset!")
                    st.rerun()
            
            with col4:
                if st.button("🔄 Update Map"):
                    st.rerun()
            
            # Autopilot status indicator with mission info - FIXED POSITION
            simulator = st.session_state.vessel_simulator
            
            # Calculate stats once to prevent layout shifts
            current_uncollected = len([p for p in simulator.plastics if not p.collected])
            current_collected = len([p for p in simulator.plastics if p.collected])
            
            if st.session_state.autopilot_active:
                if current_uncollected > 0:
                    st.info(f"🤖 **Autopilot Status:** ACTIVE - Navigating to collect {current_uncollected} remaining debris. 📹 Camera FOV: 60° cone. Click '🔄 Update Map' to see movement.")
                else:
                    st.success("🎉 **Mission Complete!** All plastic debris collected successfully!")
            else:
                st.info(f"⏸️ **Autopilot Status:** STOPPED - {current_collected} collected, {current_uncollected} remaining. 📹 Camera has limited 60° field of view.")
            
            # Update simulation - IMPROVED AUTOPILOT
            if st.session_state.autopilot_active:
                current_time = time.time()
                dt = current_time - st.session_state.last_update_time
                st.session_state.last_update_time = current_time
                
                # Update vessel simulation with better logic
                simulator = st.session_state.vessel_simulator
                
                # First, try to collect any nearby plastic
                collected = simulator.try_collect_plastic()
                if collected:
                    st.session_state.collection_counter.add_collection(
                        gps_lat=simulator.boat_lat,
                        gps_lon=simulator.boat_lon,
                        confidence=0.85
                    )
                    # Show collection success
                    if 'last_collection_time' not in st.session_state:
                        st.session_state.last_collection_time = 0
                    if current_time - st.session_state.last_collection_time > 3:
                        st.success(f"🗑️ Collected plastic debris #{collected.id}!")
                        st.session_state.last_collection_time = current_time
                
                # Find all uncollected plastics
                uncollected_plastics = [p for p in simulator.plastics if not p.collected]
                
                if uncollected_plastics:
                    # Find closest plastic (not just visible ones)
                    closest_plastic = None
                    min_distance = float('inf')
                    
                    for plastic in uncollected_plastics:
                        distance = simulator._calculate_distance(
                            simulator.boat_lat, simulator.boat_lon,
                            plastic.lat, plastic.lon
                        )
                        if distance < min_distance:
                            min_distance = distance
                            closest_plastic = plastic
                    
                    if closest_plastic:
                        # Calculate heading to closest plastic
                        target_heading = simulator._calculate_bearing(
                            simulator.boat_lat, simulator.boat_lon,
                            closest_plastic.lat, closest_plastic.lon
                        )
                        
                        # Move towards the plastic
                        simulator.update(dt * 2, 'forward', target_heading)  # Faster movement
                else:
                    # All plastics collected - stop autopilot
                    st.session_state.autopilot_active = False
                    st.success("🎉 Mission Complete! All plastic debris collected!")
            
            # Create GPS map
            simulator = st.session_state.vessel_simulator
            
            # Create folium map - OCEAN CHART ONLY
            m = folium.Map(
                location=[simulator.boat_lat, simulator.boat_lon],
                zoom_start=17,  # Close zoom to see boat and plastic details clearly
                tiles=None  # No default tiles
            )
            
            # Add ONLY ocean chart - no other options
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
                attr='Esri Ocean Chart',
                name='Ocean Chart',
                overlay=False,
                control=False  # No layer control needed
            ).add_to(m)
            
            # Add vessel marker
            folium.Marker(
                [simulator.boat_lat, simulator.boat_lon],
                popup=f"🚤 Vessel\nHeading: {simulator.boat_heading:.1f}°",
                icon=folium.Icon(color='blue', icon='ship', prefix='fa')
            ).add_to(m)
            
            # Add camera field of view cone - REALISTIC CAMERA VIEW
            import math
            camera_range_m = 50  # Larger camera detection range for better visibility
            camera_fov_deg = 60  # Camera field of view (from config)
            
            # Calculate camera cone boundaries
            left_bearing = (simulator.boat_heading - camera_fov_deg/2) % 360
            right_bearing = (simulator.boat_heading + camera_fov_deg/2) % 360
            
            # Convert to lat/lon points for the cone
            def bearing_to_point(lat, lon, bearing, distance_m):
                """Convert bearing and distance to lat/lon point"""
                # Approximate conversion
                meters_per_deg_lat = 111000
                meters_per_deg_lon = 111000 * math.cos(math.radians(lat))
                
                bearing_rad = math.radians(bearing)
                delta_lat = (distance_m * math.cos(bearing_rad)) / meters_per_deg_lat
                delta_lon = (distance_m * math.sin(bearing_rad)) / meters_per_deg_lon
                
                return [lat + delta_lat, lon + delta_lon]
            
            # Create camera cone polygon
            cone_points = [
                [simulator.boat_lat, simulator.boat_lon],  # Boat position (apex)
                bearing_to_point(simulator.boat_lat, simulator.boat_lon, left_bearing, camera_range_m),
                bearing_to_point(simulator.boat_lat, simulator.boat_lon, right_bearing, camera_range_m),
                [simulator.boat_lat, simulator.boat_lon]  # Back to boat
            ]
            
            # Add camera field of view visualization - LARGER AND DARKER
            folium.Polygon(
                locations=cone_points,
                color='darkblue',  # Darker border
                weight=4,  # Thicker border
                fillColor='blue',  # Darker fill
                fillOpacity=0.4,  # More opaque
                popup=f"📹 Camera FOV: {camera_fov_deg}°\nRange: {camera_range_m}m"
            ).add_to(m)
            
            # Add camera cone outline lines for better visibility
            folium.PolyLine(
                locations=[[simulator.boat_lat, simulator.boat_lon], 
                          bearing_to_point(simulator.boat_lat, simulator.boat_lon, left_bearing, camera_range_m)],
                color='darkblue',
                weight=3,
                opacity=0.8
            ).add_to(m)
            
            folium.PolyLine(
                locations=[[simulator.boat_lat, simulator.boat_lon], 
                          bearing_to_point(simulator.boat_lat, simulator.boat_lon, right_bearing, camera_range_m)],
                color='darkblue',
                weight=3,
                opacity=0.8
            ).add_to(m)
            
            # Add plastic markers - FLOATING ON WATER
            uncollected_plastics = [p for p in simulator.plastics if not p.collected]
            collected_plastics = [p for p in simulator.plastics if p.collected]
            
            # Show uncollected plastics in red/orange - LARGER for visibility
            for plastic in uncollected_plastics:
                folium.CircleMarker(
                    [plastic.lat, plastic.lon],
                    radius=12,  # Larger radius for better visibility
                    popup=f"🗑️ Floating Plastic Waste\nID: {plastic.id}\nSize: {plastic.size:.1f}m",
                    color='red',
                    weight=4,
                    fillColor='orange',
                    fillOpacity=0.9
                ).add_to(m)
            
            # Show collected plastics in green (faded) - LARGER for visibility
            for plastic in collected_plastics:
                folium.CircleMarker(
                    [plastic.lat, plastic.lon],
                    radius=10,  # Larger radius for better visibility
                    popup=f"✅ Collected Plastic\nID: {plastic.id}",
                    color='green',
                    weight=3,
                    fillColor='lightgreen',
                    fillOpacity=0.5
                ).add_to(m)
            
            # Show target line if autopilot is active
            if st.session_state.autopilot_active and uncollected_plastics:
                # Find closest plastic to show target line
                closest_plastic = min(uncollected_plastics, 
                    key=lambda p: simulator._calculate_distance(
                        simulator.boat_lat, simulator.boat_lon, p.lat, p.lon
                    )
                )
                
                # Draw line to target - THICKER for visibility
                folium.PolyLine(
                    locations=[[simulator.boat_lat, simulator.boat_lon], 
                              [closest_plastic.lat, closest_plastic.lon]],
                    color='blue',
                    weight=5,  # Thicker line for better visibility
                    opacity=0.8,
                    popup="🎯 Navigation Path"
                ).add_to(m)
            
            # Add camera detected plastics - ONLY WITHIN FOV CONE
            visible_plastics = simulator.get_visible_plastics()
            for plastic in visible_plastics:
                folium.CircleMarker(
                    [plastic['lat'], plastic['lon']],
                    radius=14,  # Larger radius for camera detected items
                    popup=f"📹 CAMERA DETECTED\nDistance: {plastic['distance']:.1f}m\nBearing: {plastic['bearing']:.1f}°\n(Within {camera_fov_deg}° FOV)",
                    color='yellow',
                    weight=4,
                    fillColor='yellow',
                    fillOpacity=0.9
                ).add_to(m)
            
            # No layer control needed - Ocean Chart only
            
            # Statistics dashboard - FIXED POSITION (before map)
            st.markdown("### 📊 Mission Statistics")
            
            # Use consistent variables to prevent layout shifts
            total_plastics = len(simulator.plastics)
            collected_count_stats = len([p for p in simulator.plastics if p.collected])
            remaining_count_stats = len([p for p in simulator.plastics if not p.collected])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🗑️ Collected", collected_count_stats)
            
            with col2:
                st.metric("📍 Remaining", remaining_count_stats)
            
            with col3:
                st.metric(
                    "📍 GPS Position",
                    f"{simulator.boat_lat:.4f}, {simulator.boat_lon:.4f}"
                )
            
            with col4:
                st.metric("🧭 Heading", f"{simulator.boat_heading:.1f}°")
            
            # Map container - separate from other content
            with st.container():
                st.markdown("---")
                st.markdown("### 🗺️ Deep Ocean GPS Navigation - Pure Water Environment")
                st.info("🌊 **Ocean Chart View**: Displaying marine navigation chart with water depths and ocean features only.")
                map_data = st_folium(m, width=700, height=450)
                
                # Force spacing after map
                st.markdown("<br><br><br>", unsafe_allow_html=True)
                
            
            
        else:
            # Hardware mode
            st.warning("""
            **🔧 Hardware Mode**
            
            This mode requires:
            - Raspberry Pi 4 or Jetson Nano
            - GPS Module (Neo-6M)
            - Motor drivers and servos
            - Camera module
            
            Please ensure all hardware is connected and configured.
            """)
            
            st.info("Hardware integration coming soon! Use Simulation Mode for testing.")
        
        # Manual refresh only - no automatic blinking
        # Users can click "🔄 Update Map" to see autopilot progress
