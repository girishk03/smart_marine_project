# Smart Marine Project - Usage Guide 🚀

## Quick Start

### 1. **Simple Detection (Recommended)**
```bash
cd smart_marine_project
python3 scripts/simple_detect.py --input path/to/image.jpg --output results/
```

### 2. **Batch Processing**
```bash
python3 scripts/simple_detect.py --input path/to/images/ --output results/ --conf 0.3 --thickness 1
```

### 3. **High Accuracy Mode**
```bash
python3 scripts/simple_detect.py --input path/to/image.jpg --output results/ --conf 0.5 --thickness 2
```

## 🎯 What You Get

### **Detection Results**
- **Images with bounding boxes** around detected plastic objects
- **2 simplified classes**: "plastic" and "plastic bottle"
- **Confidence scores** for each detection
- **Clean, professional visualization**

### **Output Location**
Results are saved to: `yolov5/runs/detect/smart_marine_detection/`

## 📊 Detection Modes

| Mode | Command | Confidence | Use Case |
|------|---------|------------|----------|
| **Fast** | `--conf 0.3 --thickness 1` | 30% | Quick processing |
| **Balanced** | `--conf 0.3 --thickness 1` | 30% | General purpose |
| **Accurate** | `--conf 0.5 --thickness 2` | 50% | High accuracy |

## 🔧 Parameters

- `--input`: Path to image or directory
- `--output`: Output directory (results will be in yolov5/runs/detect/)
- `--conf`: Confidence threshold (0.0-1.0)
- `--thickness`: Bounding box line thickness (1-5)

## 📁 Project Structure

```
smart_marine_project/
├── src/                          # Source code
├── models/                       # Trained models
│   └── ocean_waste_model_m2/    # Your trained model
├── configs/                      # Configuration files
├── scripts/                      # Utility scripts
│   ├── simple_detect.py         # Main detection script
│   ├── run_detection.py         # Advanced detection
│   └── quick_test.py            # Test script
├── results/                      # Sample results
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```

## 🚀 Examples

### **Single Image**
```bash
python3 scripts/simple_detect.py --input test.jpg --output results/
```

### **Directory of Images**
```bash
python3 scripts/simple_detect.py --input images/ --output results/ --conf 0.3
```

### **High Accuracy**
```bash
python3 scripts/simple_detect.py --input image.jpg --output results/ --conf 0.5 --thickness 2
```

## ✅ Your Project is Ready!

You now have a complete **Smart Marine Project** with:
- ✅ **Simplified 2-class detection** (plastic, plastic bottle)
- ✅ **Multiple detection modes** (fast, balanced, accurate)
- ✅ **Easy-to-use scripts**
- ✅ **Professional documentation**
- ✅ **Organized project structure**

## 🎯 Next Steps

1. **Test the detection**: Run the simple detection script
2. **Customize settings**: Edit config files as needed
3. **Process your images**: Use batch processing for multiple images
4. **Analyze results**: Check the detection results in the output folder

**Happy detecting! 🌊**
