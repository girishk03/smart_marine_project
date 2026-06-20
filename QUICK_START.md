# Quick Start

This guide covers the supported local Streamlit workflow. Historical commands that referenced missing training and evaluation scripts have been removed.

## Install

```bash
git clone https://github.com/girishk03/smart_marine_project.git
cd smart_marine_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Run

```bash
streamlit run reliable_web_app.py
```

Open `http://localhost:8501`.

The primary application uses Ultralytics YOLOv8n. On first use, Ultralytics may download `yolov8n.pt`, so network access is required. The app heuristically treats selected COCO container classes as possible debris; it does not determine whether an object is made of plastic.

## Test

```bash
pip install pytest
pytest -q smart_marine_project/tests
```

The suite currently has known failures involving missing model weights and benchmark input validation. See the README for the observed result and CI limitations.

## Optional Workflows

- Webcam mode requires browser camera permission.
- Batch mode accepts multiple supported image files.
- Analytics exports the current session history as CSV or JSON.
- Vessel navigation is a software simulation and does not control physical hardware.

Historical YOLOv5m plots remain in `smart_marine_project/models/ocean_waste_model_m2/`, but the dataset, weights, and complete training workflow are not included. Do not use those artifacts as evidence of current runtime accuracy.
