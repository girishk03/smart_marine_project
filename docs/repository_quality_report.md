# Repository Quality Report

## Scope

This engineering pass follows the credibility-focused documentation cleanup. It fixes verified error-handling bugs, makes CI execute the test suite, and removes generated artifacts without changing detection thresholds, class mapping, confidence boosting, or model inference behavior.

## Test Failures Fixed

| Failure | Root cause | Resolution |
| --- | --- | --- |
| Missing weights unit test | The deployment detector passed an explicit nonexistent path to Ultralytics, swallowed the resulting exception, and returned a detector with no model. | Explicit user-provided paths now raise `FileNotFoundError` before model loading. |
| Missing weights processing test | Same constructor behavior as the unit failure. | Covered by the same input validation fix. |
| Missing benchmark video | The benchmark validated its default model path before validating the user-provided video path. | Video existence is validated before optional model availability. |
| Invalid benchmark video | The benchmark likewise failed on absent weights before attempting to open the invalid video. | Video readability is validated first and returns the actionable source error. |

Local suite after the fixes: **7 passed, 3 skipped**. The three skips require historical custom weights that are intentionally absent from the repository.

## Files Removed

- 866 generated simplified-detection images.
- 7 generated files from `results/detected/`.
- 7 uploaded image copies from `static/uploads/`.
- 10 unreferenced report graphs containing unsupported or synthetic comparisons.

README screenshots under `docs/screenshots/` remain intact. Six historical benchmark JSON reports remain because the README uses them as qualified timing evidence. Historical YOLOv5m configuration, metric logs, and training plots also remain intact.

The cleanup removes roughly 950 MB from the checked-out main tree after merge. Existing Git history still contains the deleted blobs; reducing clone size further would require a separately coordinated history rewrite.

## CI Changes

- Installs dependencies through `requirements-dev.txt`.
- Compiles and imports the Streamlit entry point and deployment detector.
- Starts Streamlit headlessly and verifies `/_stcore/health` before shutdown.
- Runs `pytest -q smart_marine_project/tests` on pushes and pull requests.
- Does not claim webcam, browser, GPU, model-quality, physical-vessel, or deployment coverage.
- Does not add linting because no established lint configuration currently exists.

## Ignore and Container Hygiene

`.gitignore` now consistently excludes Python/tool caches, secrets, virtual environments, local datasets, model weights, training outputs, generated uploads, detections, benchmarks, reports, logs, and temporary files.

`.dockerignore` excludes development metadata and large non-runtime artifacts while preserving the root application, detector modules, vessel simulator, configuration, and dependency files required by the Streamlit startup command.

## Remaining Limitations

- Historical custom weights and dataset remain unavailable, so three weight-dependent tests stay skipped.
- CI verifies process startup and health, but not interactive Streamlit rendering or first-run model download.
- FastAPI uses a deprecated startup-event API and emits warnings.
- The large deleted blobs remain in Git history until an explicitly coordinated history rewrite.
- The repository still has overlapping root and package detector implementations that should be consolidated separately.

## Score Estimate

- Before PR #1: **4.6/10**
- After credibility cleanup: **8.0/10**
- After this engineering cleanup: **8.8/10**

The remaining gap is primarily reproducibility: licensed data, versioned weights, calibrated evaluation, and hardware-independent integration tests are still needed.
