# Model decisions

## Problem framing

This project targets **assistive detection** for marine and near‑shore debris from:

- drone survey imagery (post-flight)
- fixed / handheld cameras (near real-time review)

The goal is to reduce human screening time while keeping a human-in-the-loop for operational decisions.

## Why a YOLO-family detector

Constraints that favor a YOLO-style one-stage detector:

- latency matters for interactive review / live preview
- deployment often occurs on limited hardware (laptop in the field; sometimes edge devices)
- mature tooling for training, export, and inference

## Why YOLOv5 in this repo (current state)

YOLOv5 is used because:

- it is widely adopted with stable inference code paths
- model sizes (s/m/l/x) allow speed/accuracy tuning
- integration with PyTorch + OpenCV is straightforward

If starting fresh today, **YOLOv8** (Ultralytics) or other modern detectors may be preferred for:

- stronger baseline performance on many datasets
- improved export paths (depending on your target)

The repo keeps YOLOv5 because it matches the existing training artifacts and scripts.

## Model size selection

Current README references YOLOv5s; other project docs reference YOLOv5m.

Recommended way to describe this professionally:

- document the exact checkpoint(s) used in production-like runs
- keep smaller models for live preview; keep larger models for post-flight processing

Example policy:

- **Live preview:** yolov5s / small image size / higher confidence threshold
- **Post-flight batch:** yolov5m or larger / full resolution / tuned thresholds

## Evaluation plan (what to report)

### Dataset splits

Report:

- train/val/test split strategy
- whether test locations are **held out** (preferred)
- camera altitude ranges and weather conditions included

### Metrics

At minimum:

- mAP@0.5 and mAP@0.5:0.95
- precision / recall per class
- confusion trends (common false positives)

Operationally useful metrics:

- false positives per 100 images
- false negatives found during manual audit

### Latency / throughput benchmarks

Benchmark table template (fill with your numbers):

| Hardware | Device | Input size | Batch | Avg latency (ms/img) | FPS | Notes |
|---|---|---:|---:|---:|---:|---|
| MacBook (spec) | CPU | 640 | 1 |  |  | |
| Laptop GPU (spec) | CUDA | 640 | 1 |  |  | |
| Edge (spec) | CPU/GPU | 640 | 1 |  |  | |

## Thresholding policy

Thresholds are operational knobs. Document:

- default confidence threshold and why
- how thresholds change for:
  - glare / sun reflections
  - turbid water
  - higher altitude (smaller objects)

## Model versioning (recommended)

For production-like use, treat weights as immutable artifacts:

- store weights with a version tag (e.g. `weights/best_v1.0.0.pt`)
- record checksum (sha256) and training config
- log the version tag in every inference run

## Known gaps

This repo currently does not provide a complete, reproducible training pipeline with dataset lineage. If you want a production story, add:

- dataset manifest + licensing
- training script + pinned dependencies
- experiment tracking (W&B/MLflow) and model registry
