# ONNX benchmarking (docs-only)

This document outlines an **optional** path for improving inference throughput using an optimized runtime.

This repo currently runs inference via PyTorch/YOLOv5 code paths. For real-time targets (e.g., 30+ FPS), teams often evaluate:

- ONNX export + ONNX Runtime
- TensorRT (NVIDIA) from ONNX

This doc provides a reproducible evaluation checklist without changing the project code.

## What you can (and cannot) claim

- You *can* claim: a repeatable plan to evaluate ONNX-based acceleration and to record benchmark artifacts.
- You *cannot* claim: production real-time performance until you actually run and record the GPU/ONNX benchmarks.

## Export model to ONNX (YOLOv5)

YOLOv5 includes export tooling.

From the repo root:

```bash
python3 smart_mairine_project/yolov5/export.py \
  --weights smart_mairine_project/smart_marine_project/models/ocean_waste_model_m2/weights/best.pt \
  --img 640 \
  --batch 1 \
  --include onnx
```

This should produce an `.onnx` file alongside the weights (or in the default export directory depending on YOLOv5 version).

## Install ONNX Runtime

CPU:

```bash
pip3 install onnxruntime
```

GPU (NVIDIA/CUDA):

```bash
pip3 install onnxruntime-gpu
```

Notes:

- The GPU package requires compatible NVIDIA drivers and CUDA libraries.
- Use a Python environment dedicated to benchmarking to avoid dependency conflicts.

## Benchmarking methodology

Use the existing `scripts/benchmark_inference.py` as the baseline for CPU/GPU PyTorch performance.

For ONNX Runtime benchmarking, keep the methodology consistent:

- same input source (webcam/video)
- same input resolution
- record p50/p95/p99 and peak RSS
- record environment details (hardware, driver/toolkit versions)

Benchmark table template:

| Runtime | Device | Resolution | FPS | p50 | p95 | p99 | Peak RSS | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PyTorch | CPU | 640 |  |  |  |  |  | |
| PyTorch | CUDA | 640 |  |  |  |  |  | |
| ONNX RT | CPU | 640 |  |  |  |  |  | |
| ONNX RT | CUDA | 640 |  |  |  |  |  | |

## TensorRT (optional)

If you have an NVIDIA deployment target, consider TensorRT:

- export to ONNX
- build TensorRT engine
- benchmark end-to-end latency

This typically yields the best throughput but requires the most platform-specific setup.

## Recommended repo artifacts

To make this portfolio/engineering-review friendly, commit:

- the benchmark JSON reports (or a summarized table)
- the exact export command used
- environment info (driver/CUDA versions)

Avoid committing large `.onnx` / engine binaries unless you have a dedicated release artifact mechanism.
