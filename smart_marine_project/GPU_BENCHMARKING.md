# GPU benchmarking

This document describes how to benchmark Smart Marine on GPU.

## Why GPU

CPU benchmarks are useful for demos and batch review, but real-time drone feeds often require GPU acceleration to reach 30+ FPS at practical resolutions.

## Local GPU benchmarking (recommended)

Prerequisites:

- NVIDIA GPU
- CUDA-capable PyTorch build

Run:

```bash
python3 scripts/benchmark_inference.py --webcam 0 --device cuda
```

Or on a video file:

```bash
python3 scripts/benchmark_inference.py --video /path/to/drone.mp4 --device cuda
```

Tips:

- start with `--img-size 640`, then test `416` and `320`
- report p50/p95/p99 and peak RSS

## Docker GPU benchmarking (CUDA)

This repo ships a CPU-focused Dockerfile.

To benchmark with GPU inside Docker, you need:

- NVIDIA Container Toolkit installed on the host
- a CUDA base image

High-level approach:

1. Create a CUDA Dockerfile variant (example outline):

- base image: `nvidia/cuda:<version>-runtime-ubuntu22.04`
- install Python + pip
- install CUDA-enabled PyTorch
- install `requirements.txt`

2. Run with GPU access:

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e SMART_MARINE_MODEL_PATH=/app/models/ocean_waste_model_m2/weights/best.pt \
  -v "$(pwd)/models:/app/models" \
  smart-marine:cuda
```

3. Benchmark inside the container:

```bash
python scripts/benchmark_inference.py --video /path/in/container.mp4 --device cuda
```

Note: exact CUDA/PyTorch versions must match your GPU driver and toolkit.
