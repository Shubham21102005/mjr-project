# CLAUDE.md

Guidance for working in this repository.

## What this is

A **Crowd Panic Detection System**: a computer-vision pipeline that watches a
video feed (uploaded clip, webcam, phone camera, or file) and flags when a crowd
shifts from normal movement into panic. It ships as both a standalone CLI viewer
and a FastAPI web dashboard that analyses uploaded clips and monitors live feeds.

## The pipeline

Every frame flows through the same four stages, defined once and reused by the
CLI, the server, clip analysis, training, and evaluation:

1. **Person detection** — YOLOv8 (`yolov8n.pt`) finds people; `extract_boxes`
   (in `detect.py`) filters by confidence.
2. **Optical flow** — Farnebäck dense flow on a downscaled grayscale frame
   (`flow_scale=0.5`) measures per-pixel motion.
3. **Metrics** (`metrics.py::compute_metrics`) — flow + boxes reduce to 5
   **resolution-independent** numbers: `avg_speed` (% of frame diagonal/frame),
   `direction_variance` (circular variance, 0–1), `crowd_density` (occupancy
   fraction), `flow_entropy` (nats, max ln36≈3.58), `person_count`.
4. **Classification** (`classifier.py::PanicClassifier`) — per-frame panic vote
   → sliding window → hysteresis-smoothed `NORMAL`/`PANIC` label. The vote comes
   from one of three backends; the default, `scene-baseline` (`baseline.py`),
   scores each frame against what *this* feed looks like when calm.

## Key design invariants — keep these true

- **Resolution independence.** All metrics are normalized so one threshold/model
  set works across any source resolution and `flow_scale`. Don't introduce
  raw-pixel quantities into `compute_metrics`.
- **Resolution independence is NOT scene independence — decide scene-relatively.**
  Normalising by frame diagonal cancels pixel count but not camera distance or
  field of view, so absolute thresholds do not transfer between cameras (real
  UMN panic is *slower*, frame-relative, than real calm in a closer shot). The
  detector must therefore compare a frame against that feed's own baseline, not
  against a global constant. Don't reintroduce a fixed cross-camera threshold as
  the primary decision rule.
- **Single source of truth for features.** `train_classifier.py`,
  `evaluate.py`, `detect.py`, `analyze.py`, and `server.py` all call the *same*
  `compute_metrics`, so training and inference features can't drift. The
  classifier's `FEATURES` list (in `train_classifier.py`) defines which metrics
  the model consumes — the pickle stores it alongside the model.
- **No pipeline duplication.** `server.py` and `analyze.py` import
  `extract_boxes` and `draw_hud` from `detect.py`. Extend shared logic in place;
  don't copy it.
- **Three classifier backends**, selectable with the `PANIC_BACKEND` env var or
  `PanicClassifier(backend=...)`. All feed the same hysteresis window
  (`enter_fraction=0.6`, `exit_fraction=0.3`):
  - `scene-baseline` (**default**) — robust z against this feed's own calm
    history. The only backend that detects real panic.
  - `trained-model` — the pickled sklearn model. Recall 0.000 on real UMN
    footage; kept for comparison and reproducibility.
  - `threshold-vote` — fixed thresholds, ≥2 of 4, no model file needed.
- **A broken model must never fail quietly.** A `panic_model.pkl` that exists but
  won't unpickle is a broken install, not a choice to use thresholds.
  `PanicClassifier` records `load_error`, exposes `degraded`, and prints a loud
  warning; the dashboard surfaces it. Don't reduce that back to a silent
  fallback.
- **One pipeline, two ways to feed it.** `RoomBase.process(frame)` in `server.py`
  is the only implementation of the live pipeline; `PullRoom` (server opens a file
  or webcam index) and `PushRoom` (browser pushes JPEGs over `WS /ingest/{room}`)
  differ only in where frames come from. Don't fork `process()`.
- **Pushed frames drop, never queue.** `PushRoom.submit()` overwrites a single
  `pending` slot. The pipeline runs ~15 fps on CPU; queueing a faster sender would
  make the annotated view drift further behind real time every second. Dropping the
  stale frame bounds latency and just lowers the effective rate.
- **`/video/{room}` must stay an ASYNC generator.** As a sync generator it was
  iterated through Starlette's threadpool ~25x/sec per viewer, which starved the
  event loop badly enough that an ingest socket missed its keepalive pongs and the
  server killed the upload mid-stream. It also only emits when `latest_jpeg`
  actually changed. Don't revert either property.
- **Room-scoped endpoints must re-resolve the room each iteration.** `/video/{room}`
  and `/ws/{room}` used to bind `WORKERS.get(room)` once at connect; a push room
  usually does not exist yet when a page first asks for it, so the connection stayed
  dead forever. Look the room up inside the loop.
- **Annotated video must be browser-playable.** Result clips go through
  `video_writer.py`, which pipes frames to system ffmpeg (`libx264`, `yuv420p`,
  `+faststart`). OpenCV's H.264 writer is unreliable (openh264 DLL is commonly
  missing or version-mismatched) and its `mp4v` fallback produces MPEG-4 Part 2,
  which browsers won't play. Don't swap this back to a bare `cv2.VideoWriter`.

## Commands

Use the project venv. On Windows that's `venv\Scripts\python`; on macOS/Linux
`venv/bin/python`. Written below as `PY`.

Install: `PY -m pip install -r requirements.txt`.
**ffmpeg must be on PATH** for the dashboard's annotated clips.

```bash
# Web dashboard. Then open http://localhost:8000
PY -m uvicorn server:app --host 127.0.0.1 --port 8000
#   /          upload a clip + watch live feeds
#   /camera    stream THIS device's camera into a room and run the model on it

# ...with live rooms (opt-in; empty by default). Windows PowerShell:
#   $env:PANIC_SOURCES="lobby=demo_crowd_30s.avi,webcam=0"
PANIC_SOURCES="lobby=demo_crowd_30s.avi,webcam=0" PY -m uvicorn server:app --port 8000

# CLI viewer (webcam, or a file). Q to quit; --debug-flow overlays flow arrows.
PY detect.py --source 0
PY detect.py --source demo_crowd_30s.avi

# Train the classifier -> panic_model.pkl (synthetic sweep + real calm footage).
PY train_classifier.py

# Fold in UMN real escapes with PER-FRAME labels, holding out scene 3 as a test
# set. NOTE: this currently makes the model worse overall — see Known gaps.
# Do NOT use --extra-panic for UMN; that would label its calm 85% as panic.
PY train_classifier.py --umn datasets/umn/Crowd-Activity-All.avi --umn-holdout-scene 3

# Recover UMN per-frame ground truth from its burned-in banner.
PY umn_labels.py datasets/umn/Crowd-Activity-All.avi

# Evaluate against UCSD Ped1 ground truth (mismatched benchmark — see below).
PY evaluate.py --dataset datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test

# Regenerate the calm→panic synthetic test clip (panic_test.avi).
PY make_test_video.py

# Render an annotated debug video of the detector on a clip.
PY visualize.py --clip panic_test.avi --out viz/panic.mp4
```

## Files

| Area | Files |
|---|---|
| Core pipeline | `metrics.py`, `classifier.py`, `baseline.py` (scene-relative scoring) |
| CLI viewer | `detect.py` (owns `extract_boxes` + `draw_hud`, reused by server/analyze) |
| Web app | `server.py` (FastAPI: upload jobs + one worker thread per live room, MJPEG + WebSocket), `dashboard.html`, `camera.html` (browser capture page) |
| Clip analysis | `analyze.py` (offline pass over an upload), `video_writer.py` (ffmpeg/libx264) |
| Training | `train_classifier.py`, `synth.py` (synthetic clip generator) |
| Evaluation | `evaluate.py`, `visualize.py` (annotated debug renders), `umn_labels.py` (UMN per-frame ground truth) |
| Test assets | `make_test_video.py`, demo videos, `yolov8n.pt` |

## The dashboard's two modes

- **Upload** (`POST /api/upload` → `GET /api/jobs/{id}` → `GET /api/jobs/{id}/video`).
  A background thread runs `analyze.analyze_video`, which returns the panic
  segments, a downsampled per-frame metric series for the timeline, and an
  annotated MP4. One analysis at a time (`ANALYSIS_SEM`) — concurrent jobs would
  just make every progress bar lie. This is the demo path, since real panic can't
  be staged for a live camera.
- **Live**, two kinds of room:
  - *pull* — `PANIC_SOURCES` names files or webcam indices the server opens itself.
    Opt-in and empty by default, so there is no permanently dead tile.
  - *push* — a browser at `/camera` captures its own camera and pushes JPEGs over
    `WS /ingest/{room}`. The room is created on first connect and reaped
    `ROOM_IDLE_GRACE` (60s) after the last client leaves. That grace period is
    load-bearing: tearing the room down on a page refresh would discard the scene
    baseline and force another 90-frame warm-up.

  Either kind is watched with `GET /rooms` (polled by the dashboard, so new rooms
  appear without a reload), `/video/{room}` for annotated MJPEG and `WS /ws/{room}`
  for labels and metrics. `POST /api/rooms/{room}/recalibrate` throws away a feed's
  learned baseline — the escape hatch when warm-up happened while the scene was
  already busy.

  **`getUserMedia` needs a secure context.** `http://localhost` qualifies, so
  own-camera use needs no TLS. A phone hitting `http://<lan-ip>:8000` does NOT, and
  the camera fails *silently* rather than prompting — `camera.html` detects this and
  says so. Serving over HTTPS is a prerequisite for the multi-device/QR phase.

## Known gaps / caveats

- **Cross-scene detection is solved by `scene-baseline`; the absolute backends
  are not.** Measured on UMN Crowd-Activity-All.avi (11 real escapes, ground
  truth from `umn_labels.py`), per-frame, each scene scored independently:

  | scene | trained-model F1 | scene-baseline F1 | Prec | Recall |
  |---|---|---|---|---|
  | 1 | 0.000 | 0.845 | 0.777 | 0.925 |
  | 2 | 0.000 | 0.591 | 0.645 | 0.545 |
  | 3 *(held out from all tuning)* | 0.000 | 0.610 | 0.439 | **1.000** |

  On the held-out scene it raises exactly 3 alarms for 3 real escapes. The
  synthetic demo is unchanged (calm FP 0/119, panic TP 115/120 — identical to
  the trained model), so the upload demo still behaves as before.
- **Why the absolute backends fail** — measured per-frame means:

  | group | avg_speed | direction_variance | flow_entropy |
  |---|---|---|---|
  | UMN real PANIC | 0.397 | 0.210 | 1.408 |
  | demo_crowd real calm | 0.520 | 0.276 | 1.659 |
  | synthetic PANIC (training class 1) | 2.669 | 0.744 | 3.266 |

  Real panic is **slower and less disordered than real calm**, because UMN is a
  wide shot and demo_crowd a closer one. All three features overlap at p5-p95
  between those groups, so no global threshold can separate them. Retraining on
  UMN only inverts the failure (recall 1.000 on a held-out scene, but then 159 of
  199 real calm frames flagged); that experiment is kept at
  `datasets/umn/panic_model.umn-trained.pkl` and is deliberately not shipped.
- **`scene-baseline` tuning was honest.** `k=4.0`, `alpha=0.15` and the
  max-combination were tuned on UMN scenes 1 and 2 only; scene 3 was never
  looked at during tuning. Don't re-tune against scene 3 or the numbers above
  stop meaning anything.
- **Remaining `scene-baseline` costs.** (a) A 90-frame warm-up during which it
  cannot fire — it must see calm before it can call something abnormal.
  (b) Precision is modest: one false alarm in 399 frames of `demo_crowd_30s.avi`
  (a real 0.5s transient, peak z 4.81 against k=4.0). Raise `k` to trade recall
  for fewer false alarms. (c) It assumes the feed *starts* calm; a camera that
  is already showing panic at startup will learn panic as normal.
- **UCSD is a mismatched benchmark.** UCSD anomalies are non-pedestrian objects
  (bikes, carts), not crowd panic. `evaluate.py` runs there for a reproducible
  numeric baseline only — expect low recall.
- **Rooms are server-pulled, not device-pushed.** `project-summary.md` calls for
  devices joining a shared room ID and pushing their camera in; what exists is a
  fixed list of sources the server opens at startup. There is no ingest endpoint
  and the page never captures a camera. Alert delivery is likewise unbuilt — the
  alarm is a beep in whichever browser has the dashboard open.
- **Multi-camera throughput.** ~15 fps per stream on CPU, and rooms share one
  YOLO model with no frame-skipping, so N rooms split that budget.
- Ignored / not committed: `datasets/`, `venv/`, `uploads/`, `panic_test.avi`,
  `viz/`, `__pycache__/` (see `.gitignore`).
