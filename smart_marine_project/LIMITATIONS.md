# Limitations

This system is designed for **assistive detection** and will produce errors. This document lists practical limitations and mitigations for a drone operator workflow.

## Environmental limitations

- **Glare / specular reflections** can cause false positives.
- **Turbidity / foam / waves** can hide debris (false negatives).
- **Low light / motion blur** reduces small-object recall.

Mitigation:

- standardize capture settings (altitude, shutter speed, angle)
- run a calibration pass per environment (threshold tuning)

## Domain shift

Performance can degrade when moving across:

- new coastlines / water color
- different cameras / lenses
- different altitudes (object scale changes)

Mitigation:

- keep a small labeled “field validation set” per region
- periodically fine-tune and re-benchmark

## Operational risks

- **False positives** waste time and reduce trust.
- **False negatives** may under-prioritize a cleanup zone.

Mitigation:

- human review step before acting on detections
- use conservative SOPs (e.g., “review top-N hotspots manually”)

## Model and data limitations

- class taxonomy is simplified; ambiguous objects (seaweed, driftwood, reflections) remain challenging
- dataset coverage may not include extreme conditions (storms, heavy foam)

Mitigation:

- document training data coverage and known failure modes
- expand dataset with hard negatives

## Hardware constraints

- CPU-only inference may not sustain real-time throughput at high resolution.

Mitigation:

- use smaller input sizes for live preview
- batch process post-flight on a GPU machine when possible

## Quality assurance gaps (current repo)

- no guaranteed reproducible training pipeline is included
- limited automated tests for model quality regression

Mitigation:

- add a fixed validation set and a regression check (mAP + latency) before releasing new weights

## Safety note

Do not use this project as the sole basis for safety-critical navigation or autonomous intervention.
