# Demo media guide

This guide explains how to generate portfolio-friendly demo assets (screenshots and GIFs) for Smart Marine.

## Recommended assets

- Screenshot: raw frame (input) and annotated output (side-by-side)
- GIF: 5–10 seconds of live detection (webcam or video)
- Sample JSON: one `results/benchmark_*.json` and one detection JSON output

## Screenshot workflow

1. Run inference on a representative image (good lighting + challenging lighting).
2. Save:
   - the raw input image
   - the annotated output image
3. Create a side-by-side comparison image using any editor (Preview, Canva, Figma).

## GIF workflow (macOS)

### Option A: Record a short screen capture

1. Start your Streamlit app or any UI that shows detections.
2. Press `Shift + Command + 5`.
3. Choose **Record Selected Portion**.
4. Record 5–10 seconds.
5. Convert the `.mov` to a GIF.

### Option B: Convert `.mov` to `.gif`

Use any online converter or a local tool.

If you use `ffmpeg` locally (recommended), a typical conversion is:

```bash
ffmpeg -i input.mov -vf "fps=12,scale=960:-1:flags=lanczos" -loop 0 demo.gif
```

## Where to place assets

Suggested structure:

- `docs/media/demo.gif`
- `docs/media/before_after.png`
- `docs/media/sample_output.json`

Then link them from `README.md`.
