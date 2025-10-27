#!/usr/bin/env python3
"""
Smart Marine Project - REST API Server
======================================

A FastAPI-based REST API for the Smart Marine plastic detection system.
Provides endpoints for integration with other applications.
"""

import os
import sys
import json
import base64
from datetime import datetime
from typing import List, Optional
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import io
import uvicorn

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our detection system
try:
    from plastic_detector import PlasticDetector
except Exception:
    print("Warning: Could not import PlasticDetector. Using fallback detection.")
    PlasticDetector = None

# FastAPI imports
try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
    from fastapi.responses import JSONResponse, HTMLResponse, Response
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    print("Error: FastAPI not installed. Please install with: pip install fastapi uvicorn pydantic")
    sys.exit(1)

# Initialize FastAPI app
app = FastAPI(
    title="Smart Marine Project API",
    description="AI-Powered Plastic Waste Detection API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector = None
MOBILE_ROOMS = {}

# Pydantic models
class DetectionResult(BaseModel):
    bbox: List[float]
    confidence: float
    class_id: int
    class_name: str

class DetectionResponse(BaseModel):
    success: bool
    detections: List[DetectionResult]
    num_detections: int
    processing_time: float
    image_size: Optional[List[int]] = None

class BatchDetectionResponse(BaseModel):
    success: bool
    results: List[DetectionResponse]
    total_files: int
    total_detections: int
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    detector_loaded: bool
    timestamp: str
    version: str

@app.on_event("startup")
async def startup_event():
    """Initialize the detector on startup"""
    global detector
    if PlasticDetector:
        try:
            # Resolve model path relative to this file
            script_dir = os.path.dirname(__file__)
            model_path = os.path.join(script_dir, 'models/ocean_waste_model_m2/weights/best.pt')
            if os.path.exists(model_path):
                detector = PlasticDetector(model_path, device='cpu', conf_threshold=0.3)
                print("✅ Smart Marine detector loaded successfully!")
            else:
                print(f"❌ Model file not found at: {model_path}")
        except Exception as e:
            print(f"❌ Error loading detector: {e}")
    else:
        print("❌ Detection system not available")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Smart Marine Project API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# ---------------------------
# Mobile camera support
# ---------------------------
MOBILE_HTML = """
<!doctype html>
<html>
  <head>
    <meta name=viewport content="width=device-width, initial-scale=1">
    <title>Smart Marine - Mobile Camera</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 1rem; }
      video, canvas { width: 100%; max-width: 480px; border-radius: 8px; }
      .row { display:flex; gap:12px; align-items:center; }
      button { padding:10px 16px; border:none; border-radius:8px; background:#1f77b4; color:white; font-weight:600 }
    </style>
  </head>
  <body>
    <h2>Smart Marine - Mobile Camera</h2>
    <p>Room: <code id="room"></code></p>
    <video id="video" autoplay playsinline></video>
    <div class="row">
      <button id="start">Start</button>
      <button id="stop" disabled>Stop</button>
    </div>
    <canvas id="canvas" style="display:none"></canvas>
    <script>
      const qs = new URLSearchParams(location.search);
      const room = qs.get('room') || 'default';
      document.getElementById('room').textContent = room;
      const video = document.getElementById('video');
      const canvas = document.getElementById('canvas');
      let timer = null;
      async function start() {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio:false });
          video.srcObject = stream;
          document.getElementById('start').disabled = true;
          document.getElementById('stop').disabled = false;
          timer = setInterval(captureAndSend, 800);
        } catch(e) { alert('Camera error: ' + e); }
      }
      function stop() {
        if (video.srcObject) video.srcObject.getTracks().forEach(t=>t.stop());
        clearInterval(timer); timer=null;
        document.getElementById('start').disabled = false;
        document.getElementById('stop').disabled = true;
      }
      async function captureAndSend(){
        const w = video.videoWidth; const h = video.videoHeight;
        if (!w || !h) return;
        canvas.width = w; canvas.height = h;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, w, h);
        const blob = await new Promise(r=>canvas.toBlob(r, 'image/jpeg', 0.7));
        fetch(`/mobile_frame?room=${room}`, { method:'POST', body: blob, headers: { 'Content-Type': 'application/octet-stream' } });
      }
      document.getElementById('start').onclick = start;
      document.getElementById('stop').onclick = stop;
    </script>
  </body>
 </html>
"""

@app.get("/mobile", response_class=HTMLResponse)
async def mobile_page(room: str = "default"):
    return HTMLResponse(content=MOBILE_HTML, media_type="text/html")

@app.post("/mobile_frame")
async def mobile_frame(room: str, request: Request):
    """Receive a JPEG frame from mobile, run detection, store latest annotated JPEG for polling."""
    global detector
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="Empty body")
    # Decode image
    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
    # Run detection if available
    annotated = img
    if detector is not None:
        dets, info = detector.detect_objects(img)
        annotated = detector.draw_detections(img, info, line_thickness=2)
    # Encode and store
    ok, buf = cv2.imencode('.jpg', annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Encode failed")
    MOBILE_ROOMS[room] = buf.tobytes()
    return JSONResponse({"success": True, "bytes": len(MOBILE_ROOMS[room])})

@app.get("/mobile_latest.jpg")
async def mobile_latest(room: str):
    data = MOBILE_ROOMS.get(room)
    if not data:
        raise HTTPException(status_code=404, detail="No frame yet for this room")
    return Response(content=data, media_type="image/jpeg")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if detector else "unhealthy",
        detector_loaded=detector is not None,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/detect", response_model=DetectionResponse)
async def detect_plastics(
    file: UploadFile = File(...),
    confidence: float = Form(0.3),
    line_thickness: int = Form(2)
):
    """
    Detect plastic waste in a single image
    
    - **file**: Image file (JPG, PNG, GIF, BMP, TIFF)
    - **confidence**: Confidence threshold (0.1-1.0)
    - **line_thickness**: Line thickness for bounding boxes (1-5)
    """
    if not detector:
        raise HTTPException(status_code=503, detail="Detection system not available")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Update detector settings
        detector.conf_threshold = confidence
        
        # Detect objects
        start_time = datetime.now()
        detections, detection_info = detector.detect_objects(image_cv)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Convert to response format
        detection_results = []
        for det in detection_info:
            detection_results.append(DetectionResult(
                bbox=det['bbox'],
                confidence=det['confidence'],
                class_id=det['class_id'],
                class_name=det['class_name']
            ))
        
        return DetectionResponse(
            success=True,
            detections=detection_results,
            num_detections=len(detection_results),
            processing_time=processing_time,
            image_size=[image_cv.shape[1], image_cv.shape[0]]  # width, height
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect_with_image", response_model=dict)
async def detect_with_image(
    file: UploadFile = File(...),
    confidence: float = Form(0.3),
    line_thickness: int = Form(2)
):
    """
    Detect plastic waste and return annotated image
    
    - **file**: Image file (JPG, PNG, GIF, BMP, TIFF)
    - **confidence**: Confidence threshold (0.1-1.0)
    - **line_thickness**: Line thickness for bounding boxes (1-5)
    """
    if not detector:
        raise HTTPException(status_code=503, detail="Detection system not available")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Update detector settings
        detector.conf_threshold = confidence
        
        # Detect objects
        start_time = datetime.now()
        detections, detection_info = detector.detect_objects(image_cv)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Draw detections
        result_image = detector.draw_detections(image_cv, detection_info, line_thickness)
        
        # Convert result to base64
        result_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        result_pil.save(buffer, format='JPEG')
        result_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Convert to response format
        detection_results = []
        for det in detection_info:
            detection_results.append({
                "bbox": det['bbox'],
                "confidence": det['confidence'],
                "class_id": det['class_id'],
                "class_name": det['class_name']
            })
        
        return {
            "success": True,
            "detections": detection_results,
            "num_detections": len(detection_results),
            "processing_time": processing_time,
            "image_size": [image_cv.shape[1], image_cv.shape[0]],
            "result_image": f"data:image/jpeg;base64,{result_base64}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/batch_detect", response_model=BatchDetectionResponse)
async def batch_detect(
    files: List[UploadFile] = File(...),
    confidence: float = Form(0.3),
    line_thickness: int = Form(2)
):
    """
    Detect plastic waste in multiple images
    
    - **files**: List of image files (JPG, PNG, GIF, BMP, TIFF)
    - **confidence**: Confidence threshold (0.1-1.0)
    - **line_thickness**: Line thickness for bounding boxes (1-5)
    """
    if not detector:
        raise HTTPException(status_code=503, detail="Detection system not available")
    
    if len(files) > 50:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 50 files allowed per batch")
    
    try:
        results = []
        total_detections = 0
        start_time = datetime.now()
        
        # Update detector settings
        detector.conf_threshold = confidence
        
        for file in files:
            # Validate file type
            if not file.content_type.startswith('image/'):
                continue
            
            try:
                # Read image
                image_data = await file.read()
                image = Image.open(io.BytesIO(image_data))
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Detect objects
                detections, detection_info = detector.detect_objects(image_cv)
                
                # Convert to response format
                detection_results = []
                for det in detection_info:
                    detection_results.append(DetectionResult(
                        bbox=det['bbox'],
                        confidence=det['confidence'],
                        class_id=det['class_id'],
                        class_name=det['class_name']
                    ))
                
                results.append(DetectionResponse(
                    success=True,
                    detections=detection_results,
                    num_detections=len(detection_results),
                    processing_time=0.0,  # Individual processing time not tracked
                    image_size=[image_cv.shape[1], image_cv.shape[0]]
                ))
                
                total_detections += len(detection_results)
                
            except Exception as e:
                # Add error result for this file
                results.append(DetectionResponse(
                    success=False,
                    detections=[],
                    num_detections=0,
                    processing_time=0.0,
                    image_size=None
                ))
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchDetectionResponse(
            success=True,
            results=results,
            total_files=len(results),
            total_detections=total_detections,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch detection failed: {str(e)}")

@app.get("/classes")
async def get_classes():
    """Get available detection classes"""
    return {
        "classes": [
            {"id": 0, "name": "plastic", "description": "General plastic waste"},
            {"id": 1, "name": "plastic bottle", "description": "Plastic bottles specifically"}
        ]
    }

@app.get("/stats")
async def get_stats():
    """Get detection statistics (placeholder)"""
    return {
        "total_requests": 0,
        "total_detections": 0,
        "average_processing_time": 0.0,
        "most_common_class": "plastic",
        "uptime": "0:00:00"
    }

@app.post("/validate_image")
async def validate_image(file: UploadFile = File(...)):
    """Validate if an image can be processed"""
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Check image properties
        width, height = image.size
        format = image.format
        mode = image.mode
        
        return {
            "valid": True,
            "width": width,
            "height": height,
            "format": format,
            "mode": mode,
            "size_bytes": len(image_data)
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }

def main():
    """Main function to run the API server"""
    print("🌊 Smart Marine Project API Server")
    print("=" * 40)
    print("🚀 Starting API server...")
    print("📱 API Documentation: http://localhost:8000/docs")
    print("🔧 Health Check: http://localhost:8000/health")
    print("=" * 40)
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
