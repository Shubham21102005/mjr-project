# Crowd Panic Detection System

Detect when a crowd shifts from normal movement into **panic** from any video
feed — webcam, phone camera, or a video file — and raise an alert in real time.
Ships as both a standalone CLI viewer and a multi-camera web dashboard with live
alarms.

![status](https://img.shields.io/badge/status-demo-blue) ![python](https://img.shields.io/badge/python-3.11%2B-informational)

## How it works

Every frame passes through the same four-stage pipeline:

1. **Person detection** — YOLOv8 finds people in the frame.
2. **Optical flow** — dense Farnebäck flow measures how every pixel is moving.
3. **Metrics** — flow + detections reduce to five *resolution-independent*
   numbers:
   - `avg_speed` — crowd movement speed (% of frame diagonal / frame)
   - `direction_variance` — how chaotic the directions are (0 = coherent, 1 = chaotic)
   - `flow_entropy` — disorder of the motion field (nats)
   - `crowd_density` — fraction of the frame occupied by people
   - `person_count`
4. **Classification** — a trained model (or a threshold-vote fallback) labels each
   frame, and a sliding window with hysteresis smooths it into a stable
   **NORMAL** / **PANIC** decision so a single noisy frame can't trigger a false
   alarm.

Because the metrics are normalized, one model/threshold set works across sources
of any resolution.

## Install

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

Requires: `ultralytics`, `opencv-python`, `numpy`, `scipy` (see
`requirements.txt`). YOLOv8 weights (`yolov8n.pt`) are included.

## Usage

### CLI viewer

```bash
# Webcam
venv/bin/python detect.py --source 0

# Video file (press Q to quit; --debug-flow overlays flow arrows)
venv/bin/python detect.py --source demo_crowd_30s.avi
```

### Web dashboard (multi-camera, live alarm)

```bash
PANIC_SOURCES="live_crowd=demo_crowd_30s.avi,panic=panic_test.avi" \
  venv/bin/uvicorn server:app --host 127.0.0.1 --port 8000
```

Then open <http://localhost:8000>. Each "room" is one feed; the grid shows every
feed's annotated stream plus live metrics over WebSocket. A tile turns red and
sounds an alarm while its feed is in PANIC. Configure rooms with the
`PANIC_SOURCES` env var (`name=source`, comma-separated; a source is a webcam
index or a file path).

### Render an annotated clip (works on any video)

```bash
venv/bin/python visualize.py --clip your_video.mp4 --out viz/out.mp4
```

Overlays a motion heatmap, person boxes, live metrics, and the NORMAL/PANIC
label — good for offline review or a presentation.

## Training the classifier

```bash
# Trains on a synthetic calm->panic sweep + real calm footage -> panic_model.pkl
venv/bin/python train_classifier.py

# Fold in real footage when you have it (e.g. UMN panic clips)
venv/bin/python train_classifier.py --extra-panic path/to/panic.avi
```

If `panic_model.pkl` is absent, the detector falls back to a transparent
threshold vote (a frame is panic when ≥2 of 4 metrics exceed their thresholds),
so it works with zero setup.

## Evaluation

```bash
venv/bin/python evaluate.py --dataset datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test
```

Reports precision / recall / F1 / accuracy against frame-level ground truth.

## Project layout

| Area | Files |
|---|---|
| Core pipeline | `metrics.py`, `classifier.py` |
| CLI viewer | `detect.py` (owns `extract_boxes` + `draw_hud`, reused by the server) |
| Web app | `server.py` (FastAPI), `dashboard.html` |
| Training | `train_classifier.py`, `synth.py` (synthetic clip generator) |
| Evaluation | `evaluate.py`, `visualize.py` (annotated renders) |
| Test assets | `make_test_video.py`, demo videos, `yolov8n.pt` |

## Limitations

- **Not yet validated on real panic footage.** Thresholds and the shipped model
  are calibrated on synthetic clips plus a small real *calm* demo clip. The UMN
  Unusual Crowd Activity dataset is the right source for real validation.
- **UCSD is a mismatched benchmark** — its anomalies are non-pedestrian objects
  (bikes, carts), not crowd panic; `evaluate.py` uses it for a reproducible
  baseline only.
- The dashboard uses `ws://`, intended for local / trusted-network use.

## License

For educational / demonstration use.
