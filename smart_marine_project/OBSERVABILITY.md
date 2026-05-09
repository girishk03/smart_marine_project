# Observability

This document describes a production-minded observability plan for Smart Marine.

## Goals

- make inference runs **auditable** (what model/config produced what result)
- detect **quality regressions** (domain shift, threshold changes)
- measure **performance** (latency, failure rates)
- support field ops workflows (export artifacts, traceability)

## What to log (structured)

Log one JSON line per event (stdout or file), with a stable schema.

### Run metadata (once per run)

- `run_id` (UUID)
- `timestamp`
- `operator` (optional)
- `mission_id` (optional; drone flight identifier)
- `model_version` (weights tag + checksum)
- `device` (cpu/cuda + device name)
- `config` (confidence threshold, IoU, input size)

### Per-image inference event

- `run_id`, `image_id`, `image_path` (or hash)
- `width`, `height`
- `num_detections`
- `detections` summary:
  - top-k classes + confidences
  - bbox areas (for “small object” analysis)
- `latency_ms`
- `error` (if failed)

### Error event

- exception type + message
- stack trace (optional)
- whether the error was recoverable

## Metrics to track

### Reliability

- images processed / minute
- error rate (by exception type)
- percent of images with 0 detections (by location/time)

### Performance

- p50 / p95 latency per image
- throughput (FPS) for live mode

### Quality proxies (no labels required)

- confidence distribution drift (per class)
- detection count distribution drift
- average bbox area drift (proxy for altitude/zoom changes)

## Drift detection (practical approach)

Without ground-truth labels in the field, use proxy drift alerts:

- sudden drop in mean confidence
- spike in “plastic” detections in reflective conditions
- large rise in 0-detection rate compared to recent missions

## Artifact management

For each run, persist:

- run config JSON
- raw outputs (JSON detections)
- annotated images
- summary report (CSV or JSON)

Suggested directory layout:

- `runs/<run_id>/config.json`
- `runs/<run_id>/detections.jsonl`
- `runs/<run_id>/annotated/`
- `runs/<run_id>/summary.json`

## Alerting (optional)

If deployed for regular operations:

- alert on sustained error rate > threshold
- alert on extreme drift signals (as above)
- alert when model weights checksum changes unexpectedly

## Privacy and safety

- avoid storing personally identifiable data
- if capturing coastal areas with people, implement a redaction policy for exports

## Minimal implementation checklist

- add a `run_id` and `model_version` field to every output
- emit JSONL logs for run + per-image events
- generate a run summary at the end (counts, latency percentiles)
