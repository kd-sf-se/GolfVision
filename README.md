# GolfVision

GolfVision is a Python computer vision tool that analyzes a golf swing video from iPhone and writes a single annotated output file: `swing_analyze.mp4`.

It uses only:
- MediaPipe Tasks API
- OpenCV

## What it does

- Auto-corrects iPhone rotation so output is upright.
- Runs pose estimation on every frame.
- Uses all 33 pose landmarks.
- Draws the full skeleton with thick, bright lines and landmark dots.
- Computes and overlays real-time metrics for:
  - Lead elbow angle
  - Trail elbow angle
  - Hip rotation
  - Shoulder rotation

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python analyze_swing.py --input /path/to/iphone_swing.mov --handedness right
```

Optional args:
- `--output` (default: `swing_analyze.mp4`)
- `--model-path` (default: `.models/pose_landmarker_full.task`)

The script downloads the default MediaPipe pose model automatically if it is not present.
