# Smart Marine Project - Interface Guide 🌊

## 🎯 **Multiple Interface Options**

Your Smart Marine Project now has **4 different interfaces** to choose from, each designed for different use cases:

---

## 🌐 **1. Web Interface (Flask)**

**Best for:** Sharing with others, web deployment, easy access

### **Features:**
- ✅ **Drag & drop** image upload
- ✅ **Batch processing** of multiple images
- ✅ **Real-time detection** results
- ✅ **Professional UI** with Bootstrap
- ✅ **API documentation** built-in

### **How to Run:**
```bash
cd smart_marine_project
python3 web_app.py
```
**Access:** http://localhost:5000

### **Screenshots:**
- Clean, modern web interface
- Side-by-side image comparison
- Detailed detection statistics
- Batch upload capabilities

---

## 🖥️ **2. Desktop GUI (Tkinter)**

**Best for:** Personal use, offline work, full control

### **Features:**
- ✅ **Native desktop app** (works offline)
- ✅ **Real-time settings** adjustment
- ✅ **Image preview** before/after
- ✅ **Batch processing** with progress bar
- ✅ **Save results** directly

### **How to Run:**
```bash
cd smart_marine_project
python3 desktop_app.py
```

### **Features:**
- Adjustable confidence threshold
- Line thickness control
- Image comparison tabs
- Detailed results display
- Save detection results

---

## 🌊 **3. Streamlit Web App**

**Best for:** Data scientists, easy deployment, sharing

### **Features:**
- ✅ **Interactive widgets** and controls
- ✅ **Real-time visualization**
- ✅ **Analytics dashboard**
- ✅ **Easy deployment** to cloud
- ✅ **Professional documentation**

### **How to Run:**
```bash
cd smart_marine_project
streamlit run streamlit_app.py
```
**Access:** http://localhost:8501

### **Features:**
- Tabbed interface (Single, Batch, Analytics, API)
- Interactive sliders for settings
- Real-time image processing
- Analytics and statistics
- Built-in API documentation

---

## 🔌 **4. REST API Server**

**Best for:** Integration, automation, mobile apps

### **Features:**
- ✅ **RESTful API** endpoints
- ✅ **JSON responses**
- ✅ **Batch processing** support
- ✅ **Auto-generated docs** (Swagger)
- ✅ **Easy integration**

### **How to Run:**
```bash
cd smart_marine_project
python3 api_server.py
```
**Access:** 
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### **Endpoints:**
- `POST /detect` - Single image detection
- `POST /detect_with_image` - Detection with annotated image
- `POST /batch_detect` - Batch processing
- `GET /health` - Health check
- `GET /classes` - Available classes

---

## 🚀 **Quick Start Guide**

### **Option 1: Easy Launcher**
```bash
cd smart_marine_project
python3 run_interfaces.py [interface]
```

**Available interfaces:**
- `web` - Flask web interface
- `desktop` - Desktop GUI
- `streamlit` - Streamlit web app
- `api` - REST API server
- `all` - Show all options

### **Option 2: Direct Launch**
```bash
# Web Interface
python3 web_app.py

# Desktop App
python3 desktop_app.py

# Streamlit App
streamlit run streamlit_app.py

# API Server
python3 api_server.py
```

---

## 📋 **Installation Requirements**

### **Install All Dependencies:**
```bash
pip install -r requirements_interfaces.txt
```

### **Individual Interface Requirements:**

**Web Interface (Flask):**
```bash
pip install flask werkzeug
```

**Desktop GUI (Tkinter):**
```bash
# No additional packages needed (tkinter is built-in)
```

**Streamlit App:**
```bash
pip install streamlit
```

**REST API:**
```bash
pip install fastapi uvicorn python-multipart
```

---

## 🎯 **Which Interface Should You Use?**

| Use Case | Recommended Interface | Why |
|----------|----------------------|-----|
| **Sharing with team** | Web Interface | Easy access, professional UI |
| **Personal use** | Desktop GUI | Full control, works offline |
| **Data analysis** | Streamlit App | Interactive, great for exploration |
| **Mobile app** | REST API | Easy integration, JSON responses |
| **Quick testing** | Desktop GUI | Fastest to start |
| **Production deployment** | Web Interface | Most robust, scalable |

---

## 🔧 **Configuration**

### **Detection Settings:**
- **Confidence Threshold:** 0.1 - 1.0 (default: 0.3)
- **Line Thickness:** 1 - 5 (default: 2)
- **Model Path:** `models/ocean_waste_model_m2/weights/best.pt`

### **Interface Settings:**
- **Web Interface:** Port 5000
- **Streamlit App:** Port 8501
- **API Server:** Port 8000
- **Desktop GUI:** Native window

---

## 📱 **Mobile Access**

### **Web Interface:**
- Responsive design works on mobile
- Touch-friendly drag & drop
- Optimized for small screens

### **API Integration:**
- Use with mobile apps
- RESTful endpoints
- JSON responses

---

## 🌐 **Deployment Options**

### **Web Interface:**
- **Heroku:** Easy deployment
- **AWS:** Scalable hosting
- **Docker:** Containerized deployment

### **Streamlit App:**
- **Streamlit Cloud:** Free hosting
- **Heroku:** Easy deployment
- **AWS/GCP:** Enterprise hosting

### **API Server:**
- **Railway:** Simple deployment
- **AWS Lambda:** Serverless
- **Docker:** Containerized

---

## 🆘 **Troubleshooting**

### **Common Issues:**

**1. "Module not found" errors:**
```bash
pip install -r requirements_interfaces.txt
```

**2. "Model file not found":**
- Check that `models/ocean_waste_model_m2/weights/best.pt` exists
- Verify file permissions

**3. "Port already in use":**
- Change ports in the respective files
- Kill existing processes: `lsof -ti:5000 | xargs kill`

**4. "Detection not working":**
- Check model file exists
- Verify image format (JPG, PNG, etc.)
- Check confidence threshold

---

## 🎉 **Your Smart Marine Project is Ready!**

You now have **4 complete interfaces** for your plastic detection system:

✅ **Web Interface** - Professional web app  
✅ **Desktop GUI** - Native desktop application  
✅ **Streamlit App** - Interactive data science app  
✅ **REST API** - Integration-ready API  

**Choose the interface that best fits your needs and start detecting plastic waste! 🌊**
