#!/usr/bin/env python3
"""
Smart Marine Project - Web Interface
====================================

A Flask-based web interface for the Smart Marine plastic detection system.
Upload images and get instant plastic detection results.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our detection system
try:
    from plastic_detector import PlasticDetector
except ImportError:
    print("Warning: Could not import PlasticDetector. Using fallback detection.")
    PlasticDetector = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'smart_marine_project_2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Initialize detector
detector = None
if PlasticDetector:
    try:
        model_path = 'models/ocean_waste_model_m2/weights/best.pt'
        if os.path.exists(model_path):
            detector = PlasticDetector(model_path, conf_threshold=0.3)
            print("✅ Smart Marine detector loaded successfully!")
        else:
            print("❌ Model file not found. Please check the model path.")
    except Exception as e:
        print(f"❌ Error loading detector: {e}")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_directories():
    """Create necessary directories"""
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and detection"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image
        if detector:
            # Use our custom detector
            result = detector.process_image(filepath, None, line_thickness=2)
            
            # Create result image
            image = cv2.imread(filepath)
            result_image = detector.draw_detections(image, result['detections'], line_thickness=2)
            
            # Save result
            result_filename = f"result_{filename}"
            result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
            cv2.imwrite(result_path, result_image)
            
            # Prepare response
            response = {
                'success': True,
                'original_image': f"/static/uploads/{filename}",
                'result_image': f"/static/results/{result_filename}",
                'detections': result['detections'],
                'num_detections': result['num_detections'],
                'processing_time': f"{result['processing_time']:.3f}s"
            }
        else:
            # Fallback: just return the uploaded image
            response = {
                'success': True,
                'original_image': f"/static/uploads/{filename}",
                'result_image': f"/static/uploads/{filename}",
                'detections': [],
                'num_detections': 0,
                'processing_time': "0.000s",
                'message': 'Detection system not available. Image uploaded successfully.'
            }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/batch_upload', methods=['POST'])
def batch_upload():
    """Handle batch file upload and detection"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    results = []
    total_detections = 0
    
    try:
        for file in files:
            if file and allowed_file(file.filename):
                # Save uploaded file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Process image
                if detector:
                    result = detector.process_image(filepath, None, line_thickness=2)
                    
                    # Create result image
                    image = cv2.imread(filepath)
                    result_image = detector.draw_detections(image, result['detections'], line_thickness=2)
                    
                    # Save result
                    result_filename = f"result_{filename}"
                    result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
                    cv2.imwrite(result_path, result_image)
                    
                    results.append({
                        'filename': file.filename,
                        'original_image': f"/static/uploads/{filename}",
                        'result_image': f"/static/results/{result_filename}",
                        'detections': result['detections'],
                        'num_detections': result['num_detections']
                    })
                    total_detections += result['num_detections']
                else:
                    results.append({
                        'filename': file.filename,
                        'original_image': f"/static/uploads/{filename}",
                        'result_image': f"/static/uploads/{filename}",
                        'detections': [],
                        'num_detections': 0
                    })
        
        response = {
            'success': True,
            'results': results,
            'total_files': len(results),
            'total_detections': total_detections
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Batch processing failed: {str(e)}'}), 500

@app.route('/api/detect', methods=['POST'])
def api_detect():
    """API endpoint for detection"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        if detector:
            result = detector.process_image(filepath, None, line_thickness=2)
            return jsonify({
                'success': True,
                'detections': result['detections'],
                'num_detections': result['num_detections'],
                'processing_time': result['processing_time']
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Detection system not available'
            })
            
    except Exception as e:
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'detector_loaded': detector is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    ensure_directories()
    print("🌊 Smart Marine Project Web Interface")
    print("=" * 40)
    print("🚀 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🔧 Detector status:", "✅ Loaded" if detector else "❌ Not available")
    print("=" * 40)
    
    app.run(host="0.0.0.0", port=5051, debug=True)
