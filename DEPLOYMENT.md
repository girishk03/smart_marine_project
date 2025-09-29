# 🚀 Deployment Guide - Smart Marine Project

## Option 1: Streamlit Community Cloud (FREE & Recommended)

### Prerequisites
- GitHub account
- Git installed

### Step-by-Step Deployment

1. **Initialize Git Repository**
   ```bash
   cd /Users/saigirish050704/Desktop/smart_mairine_project
   git init
   git add .
   git commit -m "Initial commit - Smart Marine Project"
   ```

2. **Create GitHub Repository**
   - Go to https://github.com/new
   - Name: `smart-marine-project`
   - Make it Public
   - Don't initialize with README (we already have one)
   - Click "Create repository"

3. **Push to GitHub**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/smart-marine-project.git
   git branch -M main
   git push -u origin main
   ```

4. **Deploy on Streamlit Cloud**
   - Visit https://share.streamlit.io
   - Click "Sign in with GitHub"
   - Click "New app"
   - Repository: `YOUR_USERNAME/smart-marine-project`
   - Branch: `main`
   - Main file path: `smart_marine_project/streamlit_app.py`
   - Click "Deploy!"

5. **Your Permanent URL**
   ```
   https://YOUR_USERNAME-smart-marine-project.streamlit.app
   ```

### ⚠️ Important Notes for Streamlit Cloud

**Large Model File Issue:**
The `best.pt` model file (~40MB) may cause issues. Solutions:

**Option A: Use Git LFS (Large File Storage)**
```bash
# Install Git LFS
brew install git-lfs  # macOS
# or download from https://git-lfs.github.com

# Initialize Git LFS
git lfs install

# Track large files
git lfs track "*.pt"
git add .gitattributes
git add smart_marine_project/models/ocean_waste_model_m2/weights/best.pt
git commit -m "Add model with Git LFS"
git push
```

**Option B: Download Model on Startup**
Upload `best.pt` to Google Drive or Hugging Face, then modify `streamlit_app.py` to download it on first run.

---

## Option 2: Hugging Face Spaces (FREE)

### Steps

1. **Create Account**
   - Go to https://huggingface.co/join
   - Sign up (free)

2. **Create New Space**
   - Click your profile → "New Space"
   - Name: `smart-marine-project`
   - License: Apache 2.0
   - SDK: Streamlit
   - Click "Create Space"

3. **Upload Files**
   - Upload all files from `smart_marine_project/`
   - Or connect your GitHub repo

4. **Add Dependencies**
   Create `requirements.txt` in root:
   ```
   torch>=1.9.0
   torchvision>=0.10.0
   opencv-python>=4.5.0
   numpy>=1.21.0
   Pillow>=8.3.0
   streamlit>=1.20.0
   streamlit-webrtc>=0.45.0
   av>=10.0.0
   qrcode[pil]>=7.3.0
   requests>=2.27.0
   pandas>=1.3.0
   ultralytics>=8.0.0
   thop>=0.1.1
   ```

5. **Your Permanent URL**
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/smart-marine-project
   ```

---

## Option 3: Railway.app (FREE tier)

### Steps

1. **Sign Up**
   - Go to https://railway.app
   - Sign in with GitHub

2. **New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your `smart-marine-project` repo

3. **Configure**
   - Railway auto-detects Python
   - Add start command: `streamlit run smart_marine_project/streamlit_app.py --server.port $PORT`

4. **Environment Variables** (if needed)
   ```
   PORT=8501
   ```

5. **Deploy**
   - Click "Deploy"
   - Get your URL from the deployment

---

## Option 4: Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY smart_marine_project/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install streamlit streamlit-webrtc av qrcode[pil] ultralytics thop

# Copy application
COPY . .

# Expose port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "smart_marine_project/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Deploy to Cloud Run (Google Cloud)

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/smart-marine
gcloud run deploy smart-marine --image gcr.io/PROJECT_ID/smart-marine --platform managed
```

---

## 🎯 Recommended Approach

**For Quick & Free Deployment:**
→ **Streamlit Community Cloud** (easiest, free, perfect for demos)

**For ML Projects with GPU:**
→ **Hugging Face Spaces** (free GPU, ML-focused)

**For Production with API:**
→ **Railway.app** or **Google Cloud Run** (scalable, reliable)

---

## 📱 Sharing Your App

Once deployed, share your permanent URL:
```
https://your-app-name.streamlit.app
```

Anyone can access it without installation!

---

## 🔧 Troubleshooting Deployment

### "Module not found" errors
- Ensure all dependencies are in `requirements.txt`
- Check Python version compatibility

### Model file too large
- Use Git LFS
- Or download model on startup from cloud storage

### Memory issues
- Streamlit Cloud has 1GB RAM limit
- Consider using a smaller model or cloud platform with more resources

### Webcam not working on deployed app
- WebRTC requires HTTPS (Streamlit Cloud provides this)
- Some browsers block camera on non-localhost
- Mobile browsers may need additional permissions

---

## 💡 Tips

1. **Test locally first** before deploying
2. **Use environment variables** for sensitive data
3. **Monitor logs** in deployment platform
4. **Set up custom domain** (optional, usually paid feature)
5. **Enable auto-deploy** from GitHub for continuous deployment

---

**Need help?** Check the platform-specific documentation:
- Streamlit Cloud: https://docs.streamlit.io/streamlit-community-cloud
- Hugging Face: https://huggingface.co/docs/hub/spaces
- Railway: https://docs.railway.app
