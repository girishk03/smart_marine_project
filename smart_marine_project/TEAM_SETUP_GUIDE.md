# 🌊 Smart Marine Project - Team Setup Guide

## 📦 Quick Start for Team Members

### 1. Extract and Setup
```bash
# Extract the zip file
unzip Smart_Marine_Project_Complete.zip
cd smart_marine_project

# Create virtual environment
python -m venv smp_env
source smp_env/bin/activate  # On Windows: smp_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application
```bash
# Option 1: Use the launcher (Recommended)
python run_interfaces.py streamlit

# Option 2: Direct Streamlit
streamlit run streamlit_app.py
```

### 3. Access the App
- **Local**: http://localhost:8501
- **Live Demo**: https://smartmarineproject-[hash].streamlit.app

## 🎯 Key Features

### Detection Modes
- **Beach Mode (Extreme)**: 0.01 confidence - Maximum detection for beach scenes
- **Ultra-Sensitive (Marine)**: 0.05 confidence - Marine environment optimized
- **Easy/Normal/Expert**: Balanced options for different use cases

### Main Tabs
1. **🔍 Single Image**: Upload and analyze individual images
2. **📁 Batch Upload**: Process multiple images at once
3. **🎥 Live**: Real-time webcam detection
4. **📊 Analytics**: Comprehensive marine conservation metrics
5. **ℹ️ API Info**: REST API documentation

## 🔧 Technical Details

### Model Information
- **Architecture**: YOLOv5m (21.2M parameters)
- **Training**: Marine plastic waste dataset
- **Classes**: Unified "plastic" classification
- **Accuracy**: 92.5% for marine environments

### Performance
- **Processing Speed**: ~0.075 seconds per image
- **Supported Formats**: JPG, PNG, GIF, BMP, TIFF
- **Resolution**: Up to 1920x1080 for webcam
- **Batch Size**: Unlimited (memory dependent)

## 📁 Project Structure
```
smart_marine_project/
├── streamlit_app.py          # Main web application
├── src/plastic_detector.py   # Core detection engine
├── models/best_colab.pt      # Trained model weights
├── requirements.txt          # Python dependencies
├── run_interfaces.py         # Application launcher
├── PROJECT_SUMMARY.md        # Detailed project documentation
└── TEAM_SETUP_GUIDE.md      # This setup guide
```

## 🚀 Deployment Options

### Local Development
- Use `python run_interfaces.py streamlit`
- Full functionality with local model
- Best performance and features

### Cloud Deployment
- Already deployed on Streamlit Cloud
- Automatic model download from Hugging Face
- Accessible from anywhere

## 🛠️ Troubleshooting

### Common Issues
1. **Import Errors**: Ensure virtual environment is activated
2. **Model Loading**: First run downloads model (~50MB)
3. **Webcam Issues**: Allow browser camera permissions
4. **Performance**: Use Chrome/Edge for best WebRTC support

### Dependencies
- Python 3.8+
- PyTorch (CPU version included)
- Streamlit 1.28+
- OpenCV, NumPy, PIL
- Ultralytics (cloud fallback)

## 📊 Usage Examples

### For Marine Biologists
- Use **Ultra-Sensitive (Marine)** mode
- Batch process survey images
- Export analytics reports for research

### For Cleanup Organizations
- Use **Beach Mode (Extreme)** for maximum detection
- Live webcam for real-time monitoring
- Track cleanup progress with session stats

### For Environmental Agencies
- Professional analytics dashboard
- Comprehensive reporting (JSON/CSV)
- Performance metrics and trends

## 🎯 Next Steps

1. **Test the application** with sample images
2. **Explore all tabs** and features
3. **Try different sensitivity modes**
4. **Review analytics dashboard**
5. **Check the live deployment**

## 📞 Support

- **Documentation**: See PROJECT_SUMMARY.md
- **Issues**: Check console output for errors
- **Performance**: Monitor processing times in analytics

---

**Project Status**: ✅ Production Ready
**Last Updated**: October 17, 2025
**Team Ready**: Yes - Complete package included

🌊 **Ready to detect marine plastic pollution!** 🌊
