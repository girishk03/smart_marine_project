# Dataset and Model Provenance

This document separates facts supported by repository artifacts from metadata that has not been independently verified.

## Runtime Model

| Field | Status |
| --- | --- |
| Model | Ultralytics YOLOv8n (`yolov8n.pt`) |
| Loading method | Loaded by the Ultralytics Python package; may auto-download when absent |
| Training domain | General-purpose COCO object detection |
| Repository weight file | Not included |
| Project-specific evaluation | Not included |
| Material classification | Not supported; selected object classes are treated heuristically as possible debris |

Ultralytics software and model licensing must be reviewed for the intended deployment. Ultralytics publishes open-source and commercial licensing options; the project MIT license does not replace those terms.

## Historical YOLOv5m Experiment

The directory `smart_marine_project/models/ocean_waste_model_m2/` contains configuration, plots, batch previews, and `results.csv` from a 50-epoch YOLOv5m run.

| Artifact | Present |
| --- | --- |
| Training configuration | Yes |
| Loss and metric history | Yes |
| Plots and batch previews | Yes |
| Trained weights | No |
| Dataset snapshot | No |
| Dataset manifest/checksums | No |
| Complete runnable training scripts | No |

The best value recorded in `results.csv` is approximately `0.20765` mAP@0.5 and `0.14404` mAP@0.5:0.95. The file does not contain a 92% accuracy result. Because the weights and dataset are absent, the experiment cannot be reproduced from a clean clone.

## Dataset Metadata

`data_plastic_only.yaml` records the following metadata:

- Workspace: `smart-marine-project`
- Project: `plastic-only-detection`
- Version: `1`
- Declared class: `plastic`
- Locally declared license: `CC BY 4.0`

The repository does not include a canonical source URL, export metadata, attribution statement, immutable manifest, image inventory, or dataset files. Therefore:

- the original source and ownership are **unverified**;
- the locally declared CC BY 4.0 license is **not independently verified**;
- citation and attribution requirements cannot be confirmed from this repository;
- historical claims of 477 training and 141 validation images are not reproducible and are intentionally omitted from the README.

Before publishing or redistributing the dataset, locate the original Roboflow project/export record and preserve its exact URL, authorship, version, license, attribution, and file checksums.

## Benchmark Provenance

Six JSON files under `smart_marine_project/results/` record CPU timing observations on an Apple system. They reference a custom weight path outside this repository and report zero detections per frame. They support only a limited observation of timing in that recorded environment; they do not validate detection accuracy, useful throughput, GPU performance, or current YOLOv8n runtime behavior.

## License Boundaries

The root MIT license applies only to project-authored source code and documentation. It does not grant rights to:

- Roboflow-hosted or third-party dataset content;
- Ultralytics software or pretrained model weights;
- COCO images or annotations;
- other third-party packages, assets, or reports.

Review each upstream license before redistribution, deployment, or commercial use.
