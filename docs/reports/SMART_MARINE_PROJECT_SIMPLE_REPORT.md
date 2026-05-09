# SMART MARINE PROJECT
## AI-Powered Plastic Waste Detection System for Marine Conservation

**Project Type:** Mini Project / Final Year Project (AI + Computer Vision)

---

## ABSTRACT
Marine plastic pollution is a serious environmental problem. Manual monitoring of plastic waste on beaches and oceans is slow and costly. This project uses **Artificial Intelligence (AI)** and **Computer Vision** to automatically detect plastic waste in images and live camera frames. The system uses a YOLO-based object detection model to draw bounding boxes around plastic objects, show confidence scores, store detection history, and provide analytics. An advanced module also demonstrates how an autonomous vessel could navigate towards detected plastic locations (simulation mode).

---

## 1. INTRODUCTION
### 1.1 Problem Statement
Plastic waste enters oceans and rivers and harms marine life. We need a fast method to detect plastic waste so cleanup and monitoring become easier.

### 1.2 Proposed Solution
We built a software system that:
- Detects plastic waste in images
- Detects plastic waste in real-time using a webcam
- Processes multiple images (batch)
- Shows analytics (counts, graphs, history)
- Demonstrates an autonomous vessel simulation that can navigate to plastic points

---

## 2. OBJECTIVES
- To detect plastic waste using a deep learning object detection model
- To provide a simple user interface (Streamlit web app)
- To support single image, batch, and live webcam detection
- To record detection history and provide analytics
- To demonstrate a basic autonomous navigation workflow (simulation)

---

## 3. TECHNOLOGY STACK
### 3.1 Programming Language
- Python

### 3.2 Libraries / Tools
- Streamlit (Web UI)
- OpenCV (Image processing)
- PyTorch / Ultralytics YOLO (Deep learning inference)
- NumPy, Pandas (Data handling)
- Plotly (Analytics graphs)
- streamlit-webrtc + av (Live webcam)
- Folium + streamlit-folium + geopy + pyyaml (Autonomous map simulation)

---

## 4. SYSTEM ARCHITECTURE (HIGH LEVEL)
### 4.1 Block Diagram
```
User (Browser)
   |
   v
Streamlit App (UI)
   |
   v
Detection Engine (YOLO model)
   |
   v
Results
- Boxes + confidence
- Logs (session history)
- Analytics charts
- Export (CSV/JSON)
```

### 4.2 Data Flow
1. Input image/webcam frame is collected
2. Image is converted to the correct format (numpy array)
3. YOLO model runs inference
4. Detections are filtered/mapped to plastic-only labels
5. Bounding boxes are drawn
6. Results are displayed and stored in session analytics

---

## 5. MODULE-WISE EXPLANATION (IMPORTANT FILES)

## 5.1 Main Application (Root Level)
### 5.1.1 `reliable_web_app.py`
**Purpose:** Main Streamlit UI and the most feature-rich app.

**Why it is important:**
- It contains the complete UI with **5 tabs**.
- It loads the YOLO model and calls detection functions.
- It stores analytics in `st.session_state`.

### 5.1.2 `plastic_detector.py`
**Purpose:** Core detection logic (YOLO pipeline) in a standalone detector class.

**What it does:**
- Loads model weights
- Prepares images (resize/letterbox)
- Runs inference
- Applies NMS
- Converts raw detections to a simplified format (plastic/plastic bottle)

## 5.2 Autonomous Vessel Module
### 5.2.1 `vessel_modules/`
**Purpose:** Autonomous navigation + simulation modules used in the Autonomous tab.

Important files:
- `vessel_modules/simulator.py` → creates a digital ocean + simulated plastics + boat movement
- `vessel_modules/camera_module.py` → estimates target position (left/right/center) and navigation command
- `vessel_modules/gps_navigation.py` → navigation math and heading/distance logic
- `vessel_modules/object_counter.py` → logs collected plastics and exports CSV/JSON
- `vessel_modules/vessel_config.yaml` → configuration values

## 5.3 Package Version (Optional / Alternate Interface)
### 5.3.1 `smart_marine_project/streamlit_app.py`
**Purpose:** Another Streamlit UI (alternate). Includes cloud fallback detector.

### 5.3.2 `smart_marine_project/api_server.py`
**Purpose:** FastAPI server exposing REST endpoints (integration use).

---

## 6. STREAMLIT USER INTERFACE (WHY 5 TABS?)
The main UI in `reliable_web_app.py` uses 5 tabs because each tab solves a different real-world need.

### Tab 1: Single Image Detection
**Why:** easiest testing + demo.
- Upload image
- Detect plastic
- Show bounding boxes and details

### Tab 2: Live Webcam Detection
**Why:** real-time demo and monitoring.
- Uses webcam frames
- Shows live detection

### Tab 3: Batch Upload
**Why:** process many images quickly.
- Upload multiple images
- Process all images and show per-image results

### Tab 4: Analytics
**Why:** convert detection results into measurable stats.
- Total images processed
- Total detections
- Graphs (timeline + confidence distribution)
- Export CSV/JSON

### Tab 5: Autonomous Mode
**Why:** show how detection can support a real cleanup robot.
- Simulation mode (no hardware)
- GPS map view
- Autopilot demonstration
- Collection logging

---

## 7. HOW TO RUN THE PROJECT
### 7.1 Install Dependencies
```bash
pip install -r requirements.txt
```

### 7.2 Run the Main App
```bash
streamlit run reliable_web_app.py
```

---

## 8. RESULTS (WHAT OUTPUT DO WE GET?)
- Annotated images (boxes + labels)
- Detection counts
- Confidence scores
- Analytics dashboard
- CSV/JSON export of detection history

---

## 9. LIMITATIONS
- Accuracy depends on lighting, image quality, and training data.
- Webcam mode requires browser permissions and `streamlit-webrtc`.
- Autonomous mode is simulation unless hardware is set up.

---

## 10. CONCLUSION
This project demonstrates how AI can help marine conservation by automatically detecting plastic waste using a YOLO-based model. The Streamlit UI makes it easy for any user to run detections on images, batch data, and live camera streams, while analytics provide measurable insight. The autonomous simulation demonstrates a possible next step toward real-world cleanup automation.
