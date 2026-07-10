# CLAUDE.md

Guidance for working in this repository.

## What this is

A **Crowd Panic Detection System**: a computer-vision pipeline that watches a
video feed (webcam, phone camera, or file) and flags when a crowd shifts from
normal movement into panic. It ships as both a standalone CLI viewer and a
multi-camera FastAPI web dashboard.

## The pipeline

Every frame flows through the same four stages, defined once and reused by the
CLI, the server, training, and evaluation:

1. **Person detection** — YOLOv8 (`yolov8n.pt`) finds people; `extract_boxes`
   (in `detect.py`) filters by confidence.
2. **Optical flow** — Farnebäck dense flow on a downscaled grayscale frame
   (`flow_scale=0.5`) measures per-pixel motion.
3. **Metrics** (`metrics.py::compute_metrics`) — flow + boxes reduce to 5
   **resolution-independent** numbers: `avg_speed` (% of frame diagonal/frame),
   `direction_variance` (circular variance, 0–1), `crowd_density` (occupancy
   fraction), `flow_entropy` (nats, max ln36≈3.58), `person_count`.
4. **Classification** (`classifier.py::PanicClassifier`) — per-frame panic vote
   → sliding window → hysteresis-smoothed `NORMAL`/`PANIC` label.

## Key design invariants — keep these true

- **Resolution independence.** All metrics are normalized so one threshold/model
  set works across any source resolution and `flow_scale`. Don't introduce
  raw-pixel quantities into `compute_metrics`.
- **Single source of truth for features.** `train_classifier.py`,
  `evaluate.py`, `detect.py`, and `server.py` all call the *same*
  `compute_metrics`, so training and inference features can't drift. The
  classifier's `FEATURES` list (in `train_classifier.py`) defines which metrics
  the model consumes — the pickle stores it alongside the model.
- **No pipeline duplication.** `server.py` imports `extract_boxes` and
  `draw_hud` from `detect.py`. Extend shared logic in place; don't copy it.
- **Two classifier backends.** If `panic_model.pkl` exists it's used
  (`trained-model`); otherwise a threshold vote (≥2 of 4 metrics over threshold)
  keeps the detector working with zero setup (`threshold-vote`). Both feed the
  same hysteresis window (`enter_fraction=0.6`, `exit_fraction=0.3`).

## Commands

Use the project venv (`venv/bin/python`). Install: `pip install -r requirements.txt`.

```bash
# CLI viewer (webcam, or a file). Q to quit; --debug-flow overlays flow arrows.
venv/bin/python detect.py --source 0
venv/bin/python detect.py --source panic_test.avi

# Web dashboard (multi-room). Then open http://localhost:8000
PANIC_SOURCES="cam1=panic_test.avi,webcam=0" \
  venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000

# Train the classifier -> panic_model.pkl (synthetic sweep + real calm footage;
# add real panic footage with --extra-panic).
venv/bin/python train_classifier.py --extra-panic path/to/umn_panic.avi

# Evaluate against UCSD Ped1 ground truth (mismatched benchmark — see below).
venv/bin/python evaluate.py --dataset datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test

# Regenerate the calm→panic synthetic test clip (panic_test.avi).
venv/bin/python make_test_video.py

# Render an annotated debug video of the detector on a clip.
venv/bin/python visualize.py --clip panic_test.avi --out viz/panic.mp4
```

## Files

| Area | Files |
|---|---|
| Core pipeline | `metrics.py`, `classifier.py` |
| CLI viewer | `detect.py` (owns `extract_boxes` + `draw_hud`, reused by server) |
| Web app | `server.py` (FastAPI: one worker thread per room, MJPEG + WebSocket), `dashboard.html` |
| Training | `train_classifier.py`, `synth.py` (synthetic clip generator) |
| Evaluation | `evaluate.py`, `visualize.py` (annotated debug renders) |
| Test assets | `make_test_video.py`, demo videos, `yolov8n.pt` |

## Known gaps / caveats

- **Calibration is not validated on real panic footage.** Thresholds and the
  trained model are calibrated on synthetic clips + a small real *calm* demo
  clip. UMN Unusual Crowd Activity is the correct dataset for real validation.
- **UCSD is a mismatched benchmark.** UCSD anomalies are non-pedestrian objects
  (bikes, carts), not crowd panic. `evaluate.py` runs there for a reproducible
  numeric baseline only — expect low recall.
- **Dashboard uses `ws://`** (see `dashboard.html`), which breaks under HTTPS.
- Ignored / not committed: `datasets/`, `venv/`, `panic_test.avi`, `viz/`,
  `__pycache__/` (see `.gitignore`).
