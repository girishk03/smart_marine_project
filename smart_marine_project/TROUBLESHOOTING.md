# Troubleshooting

This document lists common setup and runtime issues for Smart Marine and how to resolve them.

## CLI: `python` not found

On macOS, `python` may not exist by default.

Use:

```bash
python3 --version
```

…and run scripts with `python3`.

## Running scripts from the wrong directory

If you run from the workspace root (`windsurf-project-4/`), scripts inside Smart Marine must be referenced with their full path.

Example:

```bash
python3 smart_mairine_project/smart_marine_project/scripts/benchmark_inference.py --webcam 0
```

## Benchmarking: `Video file not found`

The benchmark script requires a real file path.

- Verify the file exists:

```bash
ls -la "/path/to/video.mp4"
```

- Tip: drag-and-drop the video file into Terminal to paste the correct path.

## Benchmarking: OpenCV can’t open a valid video file

If the file exists but OpenCV fails to open it, it is usually a codec/container issue.

Mitigations:

- Convert the file to H.264 MP4 and retry.
- Ensure `ffmpeg` is installed locally (or use the Docker image which installs `ffmpeg`).

## Webcam errors / camera permissions

Symptoms:

- webcam opens but returns no frames
- OpenCV cannot open device `0`

Mitigations:

- System Settings → Privacy & Security → Camera → allow your terminal/IDE
- Try different indices:

```bash
python3 smart_mairine_project/smart_marine_project/scripts/benchmark_inference.py --webcam 1
```

## Model weights not found

Symptoms:

- `FileNotFoundError: Model weights not found: ...best.pt`

Mitigations:

- Confirm weights exist:

```bash
ls -la smart_mairine_project/smart_marine_project/models/ocean_waste_model_m2/weights/
```

- If you’re using Docker, mount weights (the default `.dockerignore` excludes weights):

```bash
docker run --rm -p 8000:8000 \
  -e SMART_MARINE_MODEL_PATH=/app/models/ocean_waste_model_m2/weights/best.pt \
  -v "$(pwd)/models:/app/models" \
  smart-marine:local
```

## Pytest collects unwanted archived scripts

If you run `pytest` from the repo root without configuration, it may pick up archived/backup scripts under `_archive/` or `smart_marine_project_backup/`.

Mitigation:

- Run from the workspace root using the provided `pytest.ini` (already configured)
- Or run tests from the Smart Marine project root.

## API returns 503 (detector unavailable)

This means the API process started but did not load the model.

Checks:

- Confirm `SMART_MARINE_MODEL_PATH` points to an existing `.pt` file
- Check logs for `model_file_not_found` or `model_load_failed`
- Verify file permissions inside Docker (bind mount is readable)
