#!/usr/bin/env python3

import argparse
import json
import os
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import psutil
except Exception:
    psutil = None


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(__file__))


def _import_detector():
    import sys

    src_path = os.path.join(_repo_root(), "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from plastic_detector import PlasticDetector

    return PlasticDetector


def _default_model_path() -> str:
    return os.path.join(_repo_root(), "models", "ocean_waste_model_m2", "weights", "best.pt")


def _hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
    }

    try:
        import torch

        info["torch_version"] = getattr(torch, "__version__", None)
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch_version"] = None
        info["cuda_available"] = None

    try:
        info["opencv_version"] = cv2.__version__
    except Exception:
        info["opencv_version"] = None

    return info


def _percentiles(values: List[float], ps: List[float]) -> Dict[str, float]:
    if not values:
        return {f"p{int(p)}": 0.0 for p in ps}
    vs = sorted(values)

    def pct(p: float) -> float:
        k = (len(vs) - 1) * (p / 100.0)
        f = int(np.floor(k))
        c = int(np.ceil(k))
        if f == c:
            return float(vs[int(k)])
        return float(vs[f] + (vs[c] - vs[f]) * (k - f))

    return {f"p{int(p)}": pct(p) for p in ps}


def _bench_frames(
    detector,
    frames: List[np.ndarray],
    warmup: int,
    max_frames: Optional[int],
) -> Dict[str, Any]:
    # Warmup
    for _ in range(max(0, warmup)):
        detector.detect_objects(frames[0])

    latencies_ms: List[float] = []
    detections_per_frame: List[int] = []

    rss_samples_bytes: List[int] = []
    proc = None
    if psutil is not None:
        try:
            proc = psutil.Process(os.getpid())
        except Exception:
            proc = None

    n = len(frames) if max_frames is None else min(len(frames), max_frames)

    start_total = time.time()
    for i in range(n):
        frame = frames[i]

        if proc is not None:
            try:
                rss_samples_bytes.append(int(proc.memory_info().rss))
            except Exception:
                pass

        t0 = time.time()
        _, info = detector.detect_objects(frame)
        t1 = time.time()
        latencies_ms.append((t1 - t0) * 1000.0)
        detections_per_frame.append(len(info) if isinstance(info, list) else 0)
    elapsed_total = time.time() - start_total

    fps = (n / elapsed_total) if elapsed_total > 0 else 0.0

    rss_before = rss_samples_bytes[0] if rss_samples_bytes else None
    rss_after = rss_samples_bytes[-1] if rss_samples_bytes else None
    rss_peak = max(rss_samples_bytes) if rss_samples_bytes else None

    return {
        "frames_processed": n,
        "total_seconds": elapsed_total,
        "fps": fps,
        "latency_ms": {
            "mean": statistics.mean(latencies_ms) if latencies_ms else 0.0,
            "stdev": statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
            **_percentiles(latencies_ms, [50, 90, 95, 99]),
        },
        "detections_per_frame": {
            "mean": statistics.mean(detections_per_frame) if detections_per_frame else 0.0,
            "max": max(detections_per_frame) if detections_per_frame else 0,
        },
        "resources": {
            "psutil_available": psutil is not None,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "rss_peak_bytes": rss_peak,
        },
    }


def _bench_stream(
    detector,
    cap: "cv2.VideoCapture",
    warmup: int,
    max_seconds: float,
    max_frames: Optional[int],
    sample_every_n: int,
) -> Dict[str, Any]:
    # Warmup
    for _ in range(max(0, warmup)):
        ok, frame = cap.read()
        if not ok:
            break
        detector.detect_objects(frame)

    latencies_ms: List[float] = []
    detections_per_frame: List[int] = []
    errors = 0

    rss_samples_bytes: List[int] = []
    proc = None
    if psutil is not None:
        try:
            proc = psutil.Process(os.getpid())
        except Exception:
            proc = None

    start_total = time.time()
    frames_processed = 0
    while True:
        if max_frames is not None and frames_processed >= max_frames:
            break
        if (time.time() - start_total) >= max_seconds:
            break

        ok, frame = cap.read()
        if not ok:
            break

        if proc is not None and (frames_processed % max(1, sample_every_n) == 0):
            try:
                rss_samples_bytes.append(int(proc.memory_info().rss))
            except Exception:
                pass

        t0 = time.time()
        try:
            _, info = detector.detect_objects(frame)
            detections_per_frame.append(len(info) if isinstance(info, list) else 0)
        except Exception:
            errors += 1
        t1 = time.time()
        latencies_ms.append((t1 - t0) * 1000.0)

        frames_processed += 1

    elapsed_total = time.time() - start_total
    fps = (frames_processed / elapsed_total) if elapsed_total > 0 else 0.0

    rss_before = rss_samples_bytes[0] if rss_samples_bytes else None
    rss_after = rss_samples_bytes[-1] if rss_samples_bytes else None
    rss_peak = max(rss_samples_bytes) if rss_samples_bytes else None

    return {
        "frames_processed": frames_processed,
        "total_seconds": elapsed_total,
        "fps": fps,
        "latency_ms": {
            "mean": statistics.mean(latencies_ms) if latencies_ms else 0.0,
            "stdev": statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
            **_percentiles(latencies_ms, [50, 90, 95, 99]),
        },
        "detections_per_frame": {
            "mean": statistics.mean(detections_per_frame) if detections_per_frame else 0.0,
            "max": max(detections_per_frame) if detections_per_frame else 0,
        },
        "errors": {
            "count": int(errors),
            "rate": (float(errors) / float(frames_processed)) if frames_processed > 0 else 0.0,
        },
        "resources": {
            "psutil_available": psutil is not None,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "rss_peak_bytes": rss_peak,
            "rss_samples_count": len(rss_samples_bytes),
            "rss_samples_every_n_frames": int(sample_every_n),
        },
    }


def _load_video_frames(source: str, max_frames: int, stride: int) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    if not os.path.exists(source):
        raise FileNotFoundError(
            f"Video file not found: {source}. "
            "Pass a real file path (e.g. /Users/<you>/Videos/drone.mp4)."
        )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video source: {source}. "
            "If this is a valid file, your OpenCV build may lack the codec for that container. "
            "Try converting to H.264 MP4 (or MOV) and rerun."
        )

    meta: Dict[str, Any] = {
        "source": source,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "input_fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
    }

    frames: List[np.ndarray] = []
    i = 0
    grabbed = 0
    while grabbed < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if stride <= 1 or (i % stride == 0):
            frames.append(frame)
            grabbed += 1
        i += 1

    cap.release()
    return frames, meta


def _load_webcam_frames(device_index: int, max_frames: int) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam device: {device_index}")

    meta: Dict[str, Any] = {
        "device_index": device_index,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "input_fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
    }

    frames: List[np.ndarray] = []
    for _ in range(max_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)

    cap.release()
    return frames, meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Smart Marine - Inference Benchmark")

    parser.add_argument("--model", type=str, default=_default_model_path(), help="Path to model weights")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference (cpu/cuda/auto)")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.3, help="IoU threshold")
    parser.add_argument("--img-size", type=int, default=640, help="Inference image size")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Video file path (or camera URL) to benchmark")
    group.add_argument("--webcam", type=int, help="Webcam device index (e.g. 0)")

    parser.add_argument("--max-frames", type=int, default=200, help="Max frames to benchmark")
    parser.add_argument("--stride", type=int, default=1, help="For video: take every Nth frame")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")

    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="If set, run a time-based stress benchmark for this many seconds (reads frames from the source continuously)",
    )
    parser.add_argument(
        "--rss-sample-every-n",
        type=int,
        default=10,
        help="During --max-seconds stress runs, sample RSS every N frames (requires psutil)",
    )

    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Write JSON report to this path (default: ./results/benchmark_<timestamp>.json)",
    )

    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    PlasticDetector = _import_detector()

    detector = PlasticDetector(
        model_path=model_path,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        img_size=args.img_size,
        debug_mode=False,
    )

    cap = None
    frames = None
    source_meta = None
    if args.video:
        video_path = os.path.abspath(args.video)
        if args.max_seconds is not None:
            cap = cv2.VideoCapture(args.video)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video source: {args.video}")
            source_meta = {
                "source": args.video,
                "resolved_path": video_path,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                "input_fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
                "stride": args.stride,
            }
        else:
            frames, source_meta = _load_video_frames(args.video, max_frames=args.max_frames, stride=args.stride)
        source = {"type": "video", **(source_meta or {}), "stride": args.stride, "resolved_path": video_path}
    else:
        if args.max_seconds is not None:
            cap = cv2.VideoCapture(args.webcam)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open webcam device: {args.webcam}")
            source_meta = {
                "device_index": args.webcam,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                "input_fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            }
        else:
            frames, source_meta = _load_webcam_frames(args.webcam, max_frames=args.max_frames)
        source = {"type": "webcam", **(source_meta or {})}

    if args.max_seconds is None:
        if not frames:
            raise RuntimeError("No frames read from source")

    cpu_before = None
    cpu_after = None
    if psutil is not None:
        try:
            cpu_before = float(psutil.cpu_percent(interval=0.1))
        except Exception:
            cpu_before = None

    if args.max_seconds is not None:
        try:
            bench = _bench_stream(
                detector,
                cap,
                warmup=args.warmup,
                max_seconds=float(args.max_seconds),
                max_frames=args.max_frames,
                sample_every_n=int(args.rss_sample_every_n),
            )
        finally:
            if cap is not None:
                cap.release()
    else:
        bench = _bench_frames(detector, frames, warmup=args.warmup, max_frames=len(frames))

    if psutil is not None:
        try:
            cpu_after = float(psutil.cpu_percent(interval=0.1))
        except Exception:
            cpu_after = None

    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "source": source,
        "config": {
            "model_path": model_path,
            "device": args.device,
            "conf_threshold": args.conf,
            "iou_threshold": args.iou,
            "img_size": args.img_size,
            "warmup": args.warmup,
            "max_seconds": args.max_seconds,
            "rss_sample_every_n": args.rss_sample_every_n,
        },
        "hardware": _hardware_info(),
        "system": {
            "psutil_available": psutil is not None,
            "cpu_percent_before": cpu_before,
            "cpu_percent_after": cpu_after,
        },
        "results": bench,
    }

    if args.report:
        report_path = args.report
    else:
        out_dir = os.path.join(_repo_root(), "results")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(out_dir, f"benchmark_{ts}.json")

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({"report": report_path, "fps": report["results"]["fps"], "p50_ms": report["results"]["latency_ms"]["p50"]}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
