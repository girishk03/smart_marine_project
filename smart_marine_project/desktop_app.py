#!/usr/bin/env python3
"""
Smart Marine Project - Desktop GUI
==================================

A desktop application using tkinter for the Smart Marine plastic detection system.
Easy-to-use interface for image upload and detection.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import font
import os
import sys
import threading
from PIL import Image, ImageTk
import cv2
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our detection system
try:
    from plastic_detector import PlasticDetector
except ImportError:
    print("Warning: Could not import PlasticDetector. Using fallback detection.")
    PlasticDetector = None

class SmartMarineApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Marine Project - Plastic Detection")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize detector
        self.detector = None
        self.current_image = None
        self.result_image = None
        
        # Setup GUI
        self.setup_gui()
        self.load_detector()
        
    def setup_gui(self):
        """Setup the GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🌊 Smart Marine Project", 
                               font=('Arial', 20, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        subtitle_label = ttk.Label(main_frame, text="AI-Powered Plastic Waste Detection", 
                                  font=('Arial', 12))
        subtitle_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Upload button
        self.upload_btn = ttk.Button(control_frame, text="📁 Upload Image", 
                                   command=self.upload_image, style='Accent.TButton')
        self.upload_btn.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Batch upload button
        self.batch_btn = ttk.Button(control_frame, text="📁 Upload Multiple", 
                                  command=self.upload_batch)
        self.batch_btn.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Detection settings
        settings_frame = ttk.LabelFrame(control_frame, text="Detection Settings", padding="10")
        settings_frame.grid(row=2, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Confidence threshold
        ttk.Label(settings_frame, text="Confidence:").grid(row=0, column=0, sticky=tk.W)
        self.confidence_var = tk.DoubleVar(value=0.3)
        self.confidence_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, 
                                        variable=self.confidence_var, orient=tk.HORIZONTAL)
        self.confidence_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        self.confidence_label = ttk.Label(settings_frame, text="0.3")
        self.confidence_label.grid(row=0, column=2, padx=(5, 0))
        
        # Line thickness
        ttk.Label(settings_frame, text="Line Thickness:").grid(row=1, column=0, sticky=tk.W)
        self.thickness_var = tk.IntVar(value=2)
        self.thickness_scale = ttk.Scale(settings_frame, from_=1, to=5, 
                                       variable=self.thickness_var, orient=tk.HORIZONTAL)
        self.thickness_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        self.thickness_label = ttk.Label(settings_frame, text="2")
        self.thickness_label.grid(row=1, column=2, padx=(5, 0))
        
        # Detection button
        self.detect_btn = ttk.Button(control_frame, text="🔍 Detect Plastics", 
                                   command=self.detect_plastics, state='disabled')
        self.detect_btn.grid(row=3, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Save result button
        self.save_btn = ttk.Button(control_frame, text="💾 Save Result", 
                                 command=self.save_result, state='disabled')
        self.save_btn.grid(row=4, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var)
        self.status_label.grid(row=5, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.grid(row=6, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Image display area
        image_frame = ttk.LabelFrame(main_frame, text="Image Display", padding="10")
        image_frame.grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(image_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Original image tab
        self.original_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.original_frame, text="Original Image")
        
        self.original_canvas = tk.Canvas(self.original_frame, bg='white', width=500, height=400)
        self.original_canvas.pack(expand=True, fill='both')
        
        # Result image tab
        self.result_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.result_frame, text="Detection Result")
        
        self.result_canvas = tk.Canvas(self.result_frame, bg='white', width=500, height=400)
        self.result_canvas.pack(expand=True, fill='both')
        
        # Results panel
        results_frame = ttk.LabelFrame(main_frame, text="Detection Results", padding="10")
        results_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Results text
        self.results_text = tk.Text(results_frame, height=8, width=80)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        # Bind scale events
        self.confidence_scale.configure(command=self.update_confidence)
        self.thickness_scale.configure(command=self.update_thickness)
        
    def load_detector(self):
        """Load the detection model"""
        if PlasticDetector:
            try:
                model_path = 'models/ocean_waste_model_m2/weights/best.pt'
                if os.path.exists(model_path):
                    self.detector = PlasticDetector(model_path, conf_threshold=0.3)
                    self.status_var.set("✅ Detector loaded successfully!")
                else:
                    self.status_var.set("❌ Model file not found")
            except Exception as e:
                self.status_var.set(f"❌ Error loading detector: {e}")
        else:
            self.status_var.set("❌ Detection system not available")
    
    def update_confidence(self, value):
        """Update confidence threshold display"""
        self.confidence_label.config(text=f"{float(value):.2f}")
    
    def update_thickness(self, value):
        """Update line thickness display"""
        self.thickness_label.config(text=str(int(float(value))))
    
    def upload_image(self):
        """Upload a single image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.gif *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.load_image(file_path)
    
    def upload_batch(self):
        """Upload multiple images"""
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.gif *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_paths:
            self.process_batch(file_paths)
    
    def load_image(self, file_path):
        """Load and display an image"""
        try:
            # Load image
            self.current_image = cv2.imread(file_path)
            if self.current_image is None:
                messagebox.showerror("Error", "Could not load image")
                return
            
            # Display original image
            self.display_image(self.current_image, self.original_canvas)
            
            # Enable detection button
            self.detect_btn.config(state='normal')
            self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {e}")
    
    def display_image(self, image, canvas):
        """Display image on canvas"""
        # Convert BGR to RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Resize to fit canvas
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Calculate scaling factor
            scale_w = canvas_width / pil_image.width
            scale_h = canvas_height / pil_image.height
            scale = min(scale_w, scale_h, 1.0)  # Don't scale up
            
            new_width = int(pil_image.width * scale)
            new_height = int(pil_image.height * scale)
            
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, image=photo, anchor=tk.CENTER)
        canvas.image = photo  # Keep a reference
    
    def detect_plastics(self):
        """Detect plastics in the current image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please upload an image first")
            return
        
        if not self.detector:
            messagebox.showerror("Error", "Detection system not available")
            return
        
        # Start detection in a separate thread
        self.detect_btn.config(state='disabled')
        self.progress.start()
        self.status_var.set("Detecting plastics...")
        
        thread = threading.Thread(target=self.run_detection)
        thread.daemon = True
        thread.start()
    
    def run_detection(self):
        """Run detection in background thread"""
        try:
            # Update detector settings
            self.detector.conf_threshold = self.confidence_var.get()
            
            # Run detection
            result = self.detector.process_image(
                None,  # We'll pass the image directly
                None,  # No output path
                self.thickness_var.get()
            )
            
            # Process the image directly
            detections, detection_info = self.detector.detect_objects(self.current_image)
            
            # Draw detections
            self.result_image = self.detector.draw_detections(
                self.current_image.copy(), 
                detection_info, 
                self.thickness_var.get()
            )
            
            # Update GUI in main thread
            self.root.after(0, self.detection_complete, detection_info)
            
        except Exception as e:
            self.root.after(0, self.detection_error, str(e))
    
    def detection_complete(self, detections):
        """Handle detection completion"""
        self.progress.stop()
        self.detect_btn.config(state='normal')
        self.save_btn.config(state='normal')
        
        # Display result image
        self.display_image(self.result_image, self.result_canvas)
        
        # Update results text
        self.update_results_text(detections)
        
        # Update status
        self.status_var.set(f"Detection complete! Found {len(detections)} objects")
        
        # Switch to result tab
        self.notebook.select(1)
    
    def detection_error(self, error_msg):
        """Handle detection error"""
        self.progress.stop()
        self.detect_btn.config(state='normal')
        self.status_var.set(f"Detection failed: {error_msg}")
        messagebox.showerror("Detection Error", error_msg)
    
    def update_results_text(self, detections):
        """Update the results text area"""
        self.results_text.delete(1.0, tk.END)
        
        if not detections:
            self.results_text.insert(tk.END, "No plastic objects detected.\n")
            return
        
        # Count by class
        plastic_count = sum(1 for det in detections if det['class_name'] == 'plastic')
        bottle_count = sum(1 for det in detections if det['class_name'] == 'plastic bottle')
        
        # Summary
        self.results_text.insert(tk.END, f"Detection Summary:\n")
        self.results_text.insert(tk.END, f"Total objects found: {len(detections)}\n")
        self.results_text.insert(tk.END, f"Plastic objects: {plastic_count}\n")
        self.results_text.insert(tk.END, f"Plastic bottles: {bottle_count}\n")
        self.results_text.insert(tk.END, f"\nDetailed Results:\n")
        self.results_text.insert(tk.END, f"{'='*50}\n")
        
        # Individual detections
        for i, detection in enumerate(detections, 1):
            self.results_text.insert(tk.END, f"\nDetection #{i}:\n")
            self.results_text.insert(tk.END, f"  Class: {detection['class_name']}\n")
            self.results_text.insert(tk.END, f"  Confidence: {detection['confidence']:.3f}\n")
            self.results_text.insert(tk.END, f"  Bounding Box: {detection['bbox']}\n")
    
    def save_result(self):
        """Save the detection result"""
        if self.result_image is None:
            messagebox.showwarning("Warning", "No result to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Detection Result",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG files", "*.jpg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.result_image)
                messagebox.showinfo("Success", f"Result saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving result: {e}")
    
    def process_batch(self, file_paths):
        """Process multiple images"""
        if not self.detector:
            messagebox.showerror("Error", "Detection system not available")
            return
        
        self.progress.start()
        self.status_var.set(f"Processing {len(file_paths)} images...")
        
        thread = threading.Thread(target=self.run_batch_detection, args=(file_paths,))
        thread.daemon = True
        thread.start()
    
    def run_batch_detection(self, file_paths):
        """Run batch detection in background thread"""
        try:
            results = []
            total_detections = 0
            
            for i, file_path in enumerate(file_paths):
                # Load image
                image = cv2.imread(file_path)
                if image is None:
                    continue
                
                # Detect objects
                detections, detection_info = self.detector.detect_objects(image)
                
                # Draw detections
                result_image = self.detector.draw_detections(image, detection_info, self.thickness_var.get())
                
                # Save result
                result_path = f"batch_result_{i+1}_{os.path.basename(file_path)}"
                cv2.imwrite(result_path, result_image)
                
                results.append({
                    'filename': os.path.basename(file_path),
                    'detections': detection_info,
                    'num_detections': len(detection_info)
                })
                total_detections += len(detection_info)
                
                # Update progress
                progress = (i + 1) / len(file_paths) * 100
                self.root.after(0, lambda p=progress: self.status_var.set(f"Processing... {p:.1f}%"))
            
            # Update GUI
            self.root.after(0, self.batch_complete, results, total_detections)
            
        except Exception as e:
            self.root.after(0, self.batch_error, str(e))
    
    def batch_complete(self, results, total_detections):
        """Handle batch processing completion"""
        self.progress.stop()
        self.status_var.set(f"Batch processing complete! {total_detections} total detections")
        
        # Update results text
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Batch Processing Results:\n")
        self.results_text.insert(tk.END, f"{'='*50}\n")
        self.results_text.insert(tk.END, f"Total images processed: {len(results)}\n")
        self.results_text.insert(tk.END, f"Total detections: {total_detections}\n\n")
        
        for result in results:
            self.results_text.insert(tk.END, f"File: {result['filename']}\n")
            self.results_text.insert(tk.END, f"  Detections: {result['num_detections']}\n")
            for det in result['detections']:
                self.results_text.insert(tk.END, f"    - {det['class_name']} ({det['confidence']:.3f})\n")
            self.results_text.insert(tk.END, "\n")
    
    def batch_error(self, error_msg):
        """Handle batch processing error"""
        self.progress.stop()
        self.status_var.set(f"Batch processing failed: {error_msg}")
        messagebox.showerror("Batch Processing Error", error_msg)

def main():
    """Main function"""
    root = tk.Tk()
    
    # Configure style
    style = ttk.Style()
    style.theme_use('clam')
    
    # Create custom style for accent button
    style.configure('Accent.TButton', foreground='white', background='#0078d4')
    
    # Create and run app
    app = SmartMarineApp(root)
    
    print("🌊 Smart Marine Project Desktop App")
    print("=" * 40)
    print("🚀 Starting desktop application...")
    print("📱 GUI is now available!")
    print("=" * 40)
    
    root.mainloop()

if __name__ == '__main__':
    main()
