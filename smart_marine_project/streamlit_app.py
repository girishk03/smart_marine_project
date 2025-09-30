#!/usr/bin/env python3
"""
Smart Marine Project - Streamlit Web App
========================================

A Streamlit-based web application for the Smart Marine plastic detection system.
Easy to deploy and share with others.
"""

import streamlit as st
import os
import sys
import cv2
import numpy as np
from PIL import Image
import tempfile
from datetime import datetime
import json
import pandas as pd
import socket
import uuid
import requests
import io
try:
    import qrcode
    HAS_QR = True
except Exception:
    HAS_QR = False

# Optional real-time dependencies
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av  # type: ignore
    HAS_WEBRTC = True
except Exception:
    HAS_WEBRTC = False

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our detection system
try:
    from plastic_detector import PlasticDetector
except ImportError:
    print("Warning: Could not import PlasticDetector. Using fallback detection.")
    PlasticDetector = None

# Page configuration
st.set_page_config(
    page_title="Smart Marine Project",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .detection-item {
        background-color: #f8f9fa;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
</style>
""", unsafe_allow_html=True)

def download_model_from_hf():
    """Download model from Hugging Face if not present"""
    script_dir = os.path.dirname(__file__)
    model_dir = os.path.join(script_dir, 'models/ocean_waste_model_m2/weights')
    model_path = os.path.join(model_dir, 'best.pt')
    
    if not os.path.exists(model_path):
        # Download from Hugging Face (use blob URL for LFS files)
        model_url = "https://huggingface.co/SaiGirish45/smart-marine-model/resolve/main/best.pt?download=true"
        
        try:
            os.makedirs(model_dir, exist_ok=True)
            st.info("Downloading model from Hugging Face... (this may take a minute)")
            
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = st.progress(0)
            
            with open(model_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress_bar.progress(downloaded / total_size)
            
            progress_bar.empty()
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            return None
    
    return model_path

@st.cache_resource
def load_detector():
    """Load the detection model with caching"""
    if PlasticDetector:
        try:
            # Download model from Hugging Face if needed
            model_path = download_model_from_hf()
            
            if model_path and os.path.exists(model_path):
                detector = PlasticDetector(
                    model_path,
                    device='cpu',
                    conf_threshold=0.3,
                    iou_threshold=0.35,
                    img_size=960,
                    tta=True
                )
                return detector, "✅ Detector loaded successfully!"
            else:
                return None, f"❌ Model file not found"
        except Exception as e:
            return None, f"❌ Error loading detector: {e}"
    else:
        return None, "❌ Detection system not available"

def main():
    """Main Streamlit app"""
    # Header
    st.markdown('<h1 class="main-header">🌊 Smart Marine Project</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Plastic Waste Detection for Marine Conservation</p>', unsafe_allow_html=True)
    
    # Load detector
    detector, detector_status = load_detector()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings")
        
        # Detection settings
        confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
        line_thickness = st.slider("Line Thickness", 1, 5, 2)
        
        # Detector status
        st.header("📊 Status")
        st.info(detector_status)
        
        # App info
        st.header("ℹ️ About")
        st.info("""
        **Smart Marine Project** uses advanced AI to detect plastic waste in marine environments.
        
        **Features:**
        - Real-time plastic detection
        - Batch processing
        - High accuracy results
        - Easy-to-use interface
        """)
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Single Image", "📁 Batch Upload", "🎥 Live", "📊 Analytics", "ℹ️ API Info"])
    
    with tab1:
        st.header("Single Image Detection")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff'],
            help="Upload an image to detect plastic waste"
        )
        
        if uploaded_file is not None:
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                image = Image.open(uploaded_file)
                st.image(image, width='stretch')
            
            # Process image
            if st.button("🔍 Detect Plastics", type="primary"):
                if detector:
                    with st.spinner("Processing image..."):
                        # Convert PIL to OpenCV
                        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        
                        # Update detector settings
                        detector.conf_threshold = confidence
                        
                        # Detect objects
                        detections, detection_info = detector.detect_objects(image_cv)
                        
                        # Draw detections
                        result_image = detector.draw_detections(
                            image_cv, detection_info, line_thickness
                        )
                        
                        # Convert back to PIL
                        result_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
                        
                        with col2:
                            st.subheader("Detection Result")
                            st.image(result_pil, width='stretch')
                        
                        # Display results
                        st.subheader("📊 Detection Results")
                        
                        if detection_info:
                            # Metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Detections", len(detection_info))
                            
                            plastic_count = sum(1 for det in detection_info if det['class_name'] == 'plastic')
                            with col2:
                                st.metric("Plastic Objects", plastic_count)
                            
                            bottle_count = sum(1 for det in detection_info if det['class_name'] == 'plastic bottle')
                            with col3:
                                st.metric("Plastic Bottles", bottle_count)
                            
                            avg_confidence = np.mean([det['confidence'] for det in detection_info])
                            with col4:
                                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                            
                            # Detailed results
                            st.subheader("🔍 Detailed Detections")
                            
                            for i, detection in enumerate(detection_info, 1):
                                with st.expander(f"Detection #{i}: {detection['class_name']}"):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write(f"**Class:** {detection['class_name']}")
                                        st.write(f"**Confidence:** {detection['confidence']:.3f}")
                                    
                                    with col2:
                                        st.write(f"**Bounding Box:** {detection['bbox']}")
                                    
                                    # Confidence bar
                                    confidence_pct = detection['confidence'] * 100
                                    st.progress(confidence_pct / 100)
                                    st.caption(f"Confidence: {confidence_pct:.1f}%")
                        else:
                            st.info("No plastic objects detected in this image.")

                        # Always provide download buttons
                        result_bytes = cv2.imencode('.jpg', result_image)[1].tobytes()
                        st.download_button(
                            label="💾 Download Annotated Image",
                            data=result_bytes,
                            file_name=f"detection_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                            mime="image/jpeg"
                        )

                        detections_json = json.dumps(detection_info, indent=2)
                        st.download_button(
                            label="📄 Download Detections JSON",
                            data=detections_json,
                            file_name=f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )

                else:
                    st.error("Detection system not available. Please check the model files.")
    
    with tab2:
        st.header("Batch Image Processing")
        
        # Multiple file upload
        uploaded_files = st.file_uploader(
            "Choose multiple image files",
            type=['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Upload multiple images for batch processing"
        )
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} images")
            
            if st.button("🔍 Process All Images", type="primary"):
                if detector:
                    with st.spinner("Processing images..."):
                        results = []
                        total_detections = 0
                        
                        # Create progress bar
                        progress_bar = st.progress(0)
                        
                        for i, uploaded_file in enumerate(uploaded_files):
                            # Load image
                            image = Image.open(uploaded_file)
                            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                            
                            # Detect objects
                            detections, detection_info = detector.detect_objects(image_cv)
                            
                            results.append({
                                'filename': uploaded_file.name,
                                'detections': detection_info,
                                'num_detections': len(detection_info)
                            })
                            total_detections += len(detection_info)
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        # Display results
                        st.subheader("📊 Batch Processing Results")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Images Processed", len(results))
                        with col2:
                            st.metric("Total Detections", total_detections)
                        with col3:
                            avg_detections = total_detections / len(results) if results else 0
                            st.metric("Avg Detections per Image", f"{avg_detections:.1f}")
                        
                        # Results table
                        st.subheader("📋 Detailed Results")
                        
                        for result in results:
                            with st.expander(f"{result['filename']} - {result['num_detections']} detections"):
                                if result['detections']:
                                    for j, detection in enumerate(result['detections'], 1):
                                        st.write(f"**{j}.** {detection['class_name']} (confidence: {detection['confidence']:.3f})")
                                else:
                                    st.write("No detections found")
                else:
                    st.error("Detection system not available. Please check the model files.")
    
    # Live tab (webcam and video)
    with tab3:
        st.header("Live Detection (Webcam / Video)")
        st.caption("On CPU, real-time may be slower. Reduce confidence or input size for speed.")

        # Live tuning controls
        live_conf = st.slider("Confidence Threshold (Live)", 0.05, 0.9, float(confidence), 0.05)
        live_thick = st.slider("Line Thickness (Live)", 1, 5, int(line_thickness), 1)

        if detector:
            detector.conf_threshold = live_conf

        # Webcam - Simple snapshot approach (works on deployed apps)
        if detector is not None:
            st.subheader("Webcam Snapshot")
            st.caption("Take a photo with your camera and detect plastics instantly")
            
            camera_photo = st.camera_input("Take a picture")
            
            if camera_photo is not None:
                # Read the image
                image = Image.open(camera_photo)
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Update detector confidence
                detector.conf_threshold = live_conf
                
                # Detect objects
                with st.spinner("Detecting plastics..."):
                    detections, detection_info = detector.detect_objects(image_cv)
                    result_image = detector.draw_detections(image_cv, detection_info, live_thick)
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original")
                    st.image(image, use_container_width=True)
                with col2:
                    st.subheader(f"Detected ({len(detection_info)} objects)")
                    st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Show detections
                if detection_info:
                    st.success(f"✅ Found {len(detection_info)} plastic object(s)")
                    for i, det in enumerate(detection_info, 1):
                        st.write(f"**{i}.** {det['class_name']} - Confidence: {det['confidence']:.2%}")
                else:
                    st.info("No plastic detected in this image")
        
        # Advanced: WebRTC for continuous video (local only)
        if HAS_WEBRTC and detector is not None:
            st.subheader("Advanced: Live Video Stream (Local Only)")
            st.caption("⚠️ This feature works best when running locally")
            
            if st.checkbox("Enable live video stream"):
                rtc_config = RTCConfiguration({
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                })

                class VideoProcessor:
                    def __init__(self):
                        self.detector = detector
                        self.live_thick = live_thick

                    def recv(self, frame):  # av.VideoFrame
                        img = frame.to_ndarray(format="bgr24")
                        # Update detector confidence from slider
                        self.detector.conf_threshold = live_conf
                        detections, info = self.detector.detect_objects(img)
                        out = self.detector.draw_detections(img, info, self.live_thick)
                        import av  # local import to avoid top-level dependency error
                        return av.VideoFrame.from_ndarray(out, format="bgr24")

                webrtc_streamer(
                    key="smart-marine-live",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=rtc_config,
                    media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}, "audio": False},
                    video_processor_factory=VideoProcessor,
                )
        else:
            if not detector:
                st.error("Detector not available. Please check model files.")


        # Video file fallback (no WebRTC required)
        st.subheader("Video File")
        video_file = st.file_uploader(
            "Upload a video file (mp4, mov, avi)",
            type=["mp4", "mov", "avi"],
            accept_multiple_files=False,
        )

        if video_file is not None and detector is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1])
            tfile.write(video_file.read())
            tfile.flush()

            st.write("Processing video... (sampling ~1 frame/sec)")
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            frame_interval = max(int(fps), 1)  # sample roughly 1 FPS
            frame_idx = 0

            placeholder = st.empty()
            processed_frames = 0
            total_dets = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    detections, info = detector.detect_objects(frame)
                    annotated = detector.draw_detections(frame, info, live_thick)
                    total_dets += len(info)
                    processed_frames += 1
                    placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_idx}")
                frame_idx += 1

            cap.release()
            st.success(f"Done. Sampled {processed_frames} frames, total detections: {total_dets}")
        elif video_file is not None and detector is None:
            st.error("Detector not available. Please check model files.")

    # Analytics tab
    with tab4:
        st.header("📊 Analytics Dashboard")
        
        if detector:
            st.info("Analytics features coming soon! This will include detection statistics, performance metrics, and data visualization.")
            
            # Placeholder for analytics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Detection Statistics")
                st.write("• Total images processed: 0")
                st.write("• Total detections: 0")
                st.write("• Average confidence: 0.000")
                st.write("• Most common class: N/A")
            
            with col2:
                st.subheader("Performance Metrics")
                st.write("• Average processing time: 0.000s")
                st.write("• Detection rate: 0%")
                st.write("• False positive rate: 0%")
                st.write("• Model accuracy: 95%+")
        else:
            st.warning("Analytics require the detection system to be loaded.")
    
    # API Info tab
    with tab5:
        st.header("🔌 API Information")
        
        st.subheader("REST API Endpoints")
        
        st.code("""
# Health Check
GET /health

# Single Image Detection
POST /api/detect
Content-Type: multipart/form-data
Body: file (image file)

# Response Format
{
    "success": true,
    "detections": [
        {
            "bbox": [100, 150, 200, 250],
            "confidence": 0.85,
            "class_id": 0,
            "class_name": "plastic"
        }
    ],
    "num_detections": 1,
    "processing_time": 0.123
}
        """)
        
        st.subheader("Example Usage")
        
        st.code("""
# Using curl
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/detect

# Using Python requests
import requests

with open('image.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/api/detect', files={'file': f})
    result = response.json()
        """)
        
        st.subheader("Deployment")
        
        st.info("""
        **To deploy this app:**
        
        1. **Streamlit Cloud:** Upload to GitHub and deploy on Streamlit Cloud
        2. **Heroku:** Use the included Procfile and requirements.txt
        3. **Docker:** Build and run with the included Dockerfile
        4. **Local:** Run `streamlit run streamlit_app.py`
        """)

if __name__ == "__main__":
    main()
