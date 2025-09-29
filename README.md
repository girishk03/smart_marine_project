# 🌊 Smart Marine Project

AI-Powered Plastic Waste Detection for Marine Conservation

## 📋 Overview

The Smart Marine Project uses YOLOv5-based deep learning to detect plastic waste in marine environments. It provides multiple interfaces:
- **CLI**: Command-line detection for single images or batch processing
- **Streamlit Web App**: Interactive web interface with live webcam support
- **REST API**: FastAPI-based endpoints for integration

## 🚀 Features

- Real-time plastic detection via webcam
- Single image and batch processing
- Video file analysis
- High accuracy results with configurable confidence thresholds
- REST API for external integrations
- Download annotated images and detection results (JSON/CSV)

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- macOS, Linux, or Windows

### Setup

1. **Clone/Navigate to the project**
   ```bash
   cd /Users/saigirish050704/Desktop/smart_mairine_project
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # .venv\Scripts\activate   # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r smart_marine_project/requirements.txt
   pip install fastapi uvicorn pydantic streamlit streamlit-webrtc av 'qrcode[pil]' requests ultralytics thop
   ```

## 🎯 Usage

### 1. Command Line Interface (CLI)

#### Single Image Detection
```bash
python3 smart_marine_project/scripts/run_detection.py \
  --input smart_marine_project/static/uploads/20250929_021229_plastic_940.jpg \
  --output smart_marine_project/results/result.jpg \
  --mode balanced \
  --device cpu
```

#### Batch Processing
```bash
python3 smart_marine_project/scripts/run_detection.py \
  --input smart_marine_project/static/uploads \
  --output smart_marine_project/results/batch \
  --mode accurate \
  --device cpu \
  --save-results
```

#### CLI Options
- `--input`: Path to image file or directory
- `--output`: Output path for annotated image(s)
- `--mode`: Detection mode (`fast`, `balanced`, `accurate`)
- `--conf`: Confidence threshold (0.1-1.0, default: 0.3)
- `--thickness`: Bounding box line thickness (1-5, default: 2)
- `--device`: Device to use (`cpu`, `cuda`, `mps`)
- `--save-results`: Save detection results as JSON

### 2. Streamlit Web App

```bash
streamlit run smart_marine_project/streamlit_app.py
```

Then open your browser to http://localhost:8501

**Features:**
- **Single Image**: Upload and detect plastics in images
- **Batch Upload**: Process multiple images at once
- **Live**: Real-time webcam detection (requires `streamlit-webrtc` and `av`)
- **Analytics**: View detection statistics (coming soon)
- **API Info**: REST API documentation

**Live Webcam:**
- Adjust confidence threshold and line thickness in real-time
- Press F11 for browser fullscreen mode

### 3. REST API Server

```bash
uvicorn smart_marine_project.api_server:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000/docs for interactive API documentation.

**Key Endpoints:**
- `GET /health` - Check API status
- `POST /detect` - Detect plastics in a single image
- `POST /detect_with_image` - Get detections + annotated image
- `POST /batch_detect` - Process multiple images
- `GET /classes` - List detection classes

**Example with curl:**
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@path/to/image.jpg" \
  -F "confidence=0.3"
```

## 📁 Project Structure

```
smart_mairine_project/
├── smart_marine_project/
│   ├── src/
│   │   └── plastic_detector.py      # Core detection engine
│   ├── scripts/
│   │   └── run_detection.py         # CLI tool
│   ├── models/
│   │   └── ocean_waste_model_m2/
│   │       └── weights/
│   │           └── best.pt          # YOLOv5 model weights
│   ├── static/
│   │   └── uploads/                 # Sample images
│   ├── results/                     # Output directory
│   ├── streamlit_app.py             # Streamlit web interface
│   ├── api_server.py                # FastAPI REST server
│   └── requirements.txt             # Python dependencies
├── yolov5/                          # YOLOv5 repository
└── .venv/                           # Virtual environment
```

## 🔧 Configuration

### Detection Modes
- **Fast**: Lower confidence (0.3), faster processing
- **Balanced**: Good balance of speed and accuracy (default)
- **Accurate**: Higher confidence (0.5), stricter detections

### Confidence Threshold
- Lower (0.2-0.3): More detections, may include false positives
- Medium (0.3-0.4): Balanced
- Higher (0.5-0.7): Fewer but more confident detections

### Device Selection
- `cpu`: Works on all systems (slower)
- `cuda`: NVIDIA GPU (faster, requires CUDA)
- `mps`: Apple Silicon GPU (macOS only)

## 🎨 Detection Classes

The model detects two classes:
1. **plastic** - General plastic waste
2. **plastic bottle** - Plastic bottles specifically

## 📊 Output Formats

### Annotated Images
- Bounding boxes with class labels and confidence scores
- Saved as JPEG/PNG

### JSON Results
```json
{
  "detections": [
    {
      "class_name": "plastic bottle",
      "confidence": 0.85,
      "bbox": [100, 150, 200, 250],
      "class_id": 1
    }
  ]
}
```

### CSV (Batch Processing)
- Filename, detection index, class, confidence, bbox

## 🐛 Troubleshooting

### "Detection system not available"
- Ensure all dependencies are installed: `pip install ultralytics thop`
- Check model weights exist at: `smart_marine_project/models/ocean_waste_model_m2/weights/best.pt`

### Webcam not working
- Install webcam dependencies: `pip install streamlit-webrtc av`
- Allow camera permissions in your browser
- Try a different browser (Chrome/Safari recommended)

### Low detection accuracy
- Adjust confidence threshold (try 0.25-0.35)
- Ensure good lighting on objects
- Use "accurate" mode for stricter results

### Port already in use
```bash
# Kill existing processes
pkill -f "uvicorn smart_marine_project.api_server:app"
pkill -f "streamlit run"

# Or use a different port
uvicorn smart_marine_project.api_server:app --port 8001
```

## 🚀 Quick Start Commands

### All-in-One Setup
```bash
# Navigate to project
cd /Users/saigirish050704/Desktop/smart_mairine_project

# Activate venv
source .venv/bin/activate

# Run Streamlit (recommended for beginners)
streamlit run smart_marine_project/streamlit_app.py
```

### Development Mode
```bash
# Terminal 1: API Server
source .venv/bin/activate
uvicorn smart_marine_project.api_server:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Streamlit
source .venv/bin/activate
streamlit run smart_marine_project/streamlit_app.py
```

## 📝 Notes

- Model trained on marine plastic waste dataset
- Best results with clear, well-lit images
- CPU inference is slower; GPU recommended for real-time use
- Webcam requires HTTPS on mobile browsers (use localhost or tunnel)

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Ensure model weights are present
4. Check terminal output for error messages

## 📄 License

This project uses YOLOv5 (GPL-3.0 license) and is intended for educational and research purposes.

---

**Built with ❤️ for Marine Conservation**
