# Crowd Panic Detection System

Detect when a crowd shifts from normal movement into **panic** from any video
feed — an uploaded clip, a webcam, or a video file — and raise an alert in real
time. Ships as a standalone CLI viewer and a web dashboard that both analyses
uploaded clips and monitors live feeds.

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
4. **Classification** — each frame is scored against **what this feed looks like
   when it's calm**, and a sliding window with hysteresis smooths that into a
   stable **NORMAL** / **PANIC** decision so a single noisy frame can't trigger a
   false alarm.

### Why the decision is scene-relative

Absolute thresholds don't transfer between cameras. Measured on real footage:
UMN crowd escapes average `avg_speed` **0.397**, while the calm `demo_crowd`
clip averages **0.520** — real panic in a wide overhead shot is *slower*,
frame-relative, than real calm in a closer shot. Normalising by frame diagonal
cancels resolution but not camera distance or field of view.

So instead of asking *"is this fast?"* the detector asks *"is this much faster
and more disordered than normal **for this camera**?"* — a robust z-score
(median + MAD) against a rolling baseline of the feed's own calm frames. This is
also what the crowd-anomaly literature does; see
[Bird et al., *Holistic Features For Real-Time Crowd Behaviour Anomaly
Detection*](https://arxiv.org/pdf/1606.05310), which trains on each scene's own
normal period and reports its cross-scene variant as clearly inferior.

Three backends are available via `PANIC_BACKEND`:
`scene-baseline` (default, the only one that detects real panic),
`trained-model`, and `threshold-vote`.

## Install

```bash
python -m venv venv

# Windows
venv\Scripts\pip install -r requirements.txt
# macOS / Linux
venv/bin/pip install -r requirements.txt
```

**ffmpeg is required for the web dashboard.** Annotated result clips are encoded
with `libx264` so they play in the browser; OpenCV's own H.264 writer is
unreliable across machines. Install ffmpeg and make sure it's on `PATH`
(`winget install Gyan.FFmpeg`, `brew install ffmpeg`, or
`apt install ffmpeg`). The CLI viewer does not need it.

YOLOv8 weights (`yolov8n.pt`) are included in the repo.

Throughout the rest of this README, `PY` means `venv\Scripts\python` on Windows
and `venv/bin/python` on macOS/Linux.

## Web dashboard

```bash
# Windows
venv\Scripts\python -m uvicorn server:app --host 127.0.0.1 --port 8000
# macOS / Linux
venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Then open <http://localhost:8000>.

### Analyse an uploaded clip — the main demo path

Drop a video onto the page and hit **Analyse**. Staging real panic in front of a
live camera isn't something you can (or should) do, so a recorded clip is the
honest way to show the detector working.

The server runs the full pipeline over every frame and returns:

- an **annotated H.264 clip** — person boxes, live metrics, and the NORMAL/PANIC
  label burned in — playable and seekable right in the page
- a **panic timeline**: red bands mark every PANIC stretch over an `avg_speed`
  curve. Click anywhere on it to jump the player there.
- a **list of panic episodes** with start/end timestamps; click one to seek to it
- **summary stats** — duration, frame count, number of episodes, share of the
  clip spent in panic, peak speed, and which classifier backend decided it

Progress (frames done, processing fps, ETA) streams while it works. Analysis runs
at roughly 10–16 fps on CPU, so a 30-second clip takes about a minute. One clip
is analysed at a time.

Accepts `.mp4 .avi .mov .mkv .webm .m4v .mpg .mpeg`, up to 512 MB.

### Stream this device's camera

Open <http://localhost:8000/camera>, pick a room name and hit **Start camera**. The
page captures your camera, pushes JPEG frames to the server over a WebSocket, and the
full pipeline runs on every frame. You get your raw preview plus (toggleable) the
annotated stream the model sees — person boxes, live metrics and the NORMAL/PANIC
label burned in — and the feed also shows up as a tile on the main dashboard
automatically, no reload needed.

Because detection is scene-relative, the camera first spends ~90 frames learning what
*normal* looks like where it's pointed; the page shows that as a calibration bar. Hold
still through it, then move fast and erratically to trip the detector. If warm-up
happened while the scene was already busy, hit **Recalibrate** to re-learn.

> **This only works on `localhost`.** Browsers only allow camera access in a secure
> context. `http://localhost` counts, so your own machine is fine with no setup. A
> phone pointed at `http://<your-lan-ip>:8000` does **not**, and the camera fails
> silently rather than prompting — the page detects that and tells you. Serving over
> HTTPS is what unlocks other devices.

Frame rate and resolution are selectable (default 10 fps at 640x480). If the pipeline
falls behind the sender, stale frames are dropped rather than queued, so latency stays
bounded and only the effective frame rate drops.

### Monitor live feeds

Live rooms are opt-in — set `PANIC_SOURCES` (comma-separated `name=source`,
where a source is a webcam index or a file path the server can open):

```bash
# Windows (PowerShell)
$env:PANIC_SOURCES="lobby=demo_crowd_30s.avi,webcam=0"; venv\Scripts\python -m uvicorn server:app --port 8000
# macOS / Linux
PANIC_SOURCES="lobby=demo_crowd_30s.avi,webcam=0" venv/bin/python -m uvicorn server:app --port 8000
```

Each room gets a tile in the grid streaming its annotated feed as MJPEG, with
labels and metrics arriving over a WebSocket. A tile turns red and sounds an
alarm while its feed is in PANIC.

These rooms are **server-pulled** sources fixed at startup, in contrast to the
browser-pushed rooms above.

### HTTP API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/upload` | Upload a clip; returns a `job_id` |
| `WS` | `/ingest/{room}` | Push camera frames in (binary JPEG); pushes state back |
| `POST` | `/api/rooms/{room}/recalibrate` | Forget a feed's learned baseline |
| `GET` | `/api/jobs/{id}` | Job status, progress, and the full result |
| `GET` | `/api/jobs/{id}/video` | The annotated MP4 (supports Range requests) |
| `DELETE` | `/api/jobs/{id}` | Drop a job and its files |
| `GET` | `/rooms` | Configured live rooms |
| `GET` | `/video/{room}` | MJPEG stream for a live room |
| `WS` | `/ws/{room}` | Live labels + metrics |

## CLI viewer

```bash
PY detect.py --source 0                     # webcam
PY detect.py --source demo_crowd_30s.avi    # file; Q quits, --debug-flow overlays arrows
```

### Render an annotated clip offline

```bash
PY visualize.py --clip your_video.mp4 --out viz/out.mp4
```

Overlays a motion heatmap, person boxes, live metrics, and the NORMAL/PANIC
label — good for offline review or a presentation.

## Training the classifier

```bash
PY train_classifier.py                                  # synthetic sweep + real calm footage
PY train_classifier.py --extra-panic path/to/panic.avi  # fold in real footage (e.g. UMN)
```

Writes `panic_model.pkl` (StandardScaler + LogisticRegression over `avg_speed`,
`direction_variance`, `flow_entropy`).

If `panic_model.pkl` is absent the detector falls back to a transparent threshold
vote (panic when ≥2 of 4 metrics exceed their thresholds), so it works with zero
setup. If the file is *present but fails to load*, that's a broken install — the
detector says so loudly and the dashboard flags the result as degraded rather
than quietly downgrading.

## Evaluation

```bash
PY evaluate.py --dataset datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test
```

Reports precision / recall / F1 / accuracy against frame-level ground truth.

## Project layout

| Area | Files |
|---|---|
| Core pipeline | `metrics.py`, `classifier.py`, `baseline.py` |
| CLI viewer | `detect.py` (owns `extract_boxes` + `draw_hud`, reused everywhere) |
| Web app | `server.py` (FastAPI), `dashboard.html`, `camera.html` |
| Clip analysis | `analyze.py`, `video_writer.py` (ffmpeg/libx264 encoding) |
| Training | `train_classifier.py`, `synth.py` (synthetic clip generator) |
| Evaluation | `evaluate.py`, `visualize.py`, `umn_labels.py` (UMN ground truth) |
| Test assets | `make_test_video.py`, demo videos, `yolov8n.pt` |

## Limitations

- **Validated on real panic footage.** Measured against the UMN Unusual Crowd
  Activity dataset (11 real crowd escapes; ground truth recovered by
  `umn_labels.py` from the video's own burned-in banner), scoring each scene
  independently:

  | scene | old `trained-model` F1 | `scene-baseline` F1 | Recall |
  |---|---|---|---|
  | 1 | 0.000 | 0.845 | 0.925 |
  | 2 | 0.000 | 0.591 | 0.545 |
  | 3 *(held out from all tuning)* | 0.000 | 0.610 | **1.000** |

  On the held-out scene it raises exactly 3 alarms for the 3 real escapes.
  Parameters were tuned on scenes 1 and 2 only.
- **Precision is modest.** It catches panic reliably but over-reports: one false
  alarm in 399 frames of real calm footage, and roughly 2.5 spurious alarms per
  real one on UMN scene 2. Raise `k` in `baseline.py` to trade recall for fewer
  false alarms — appropriate if you'd rather miss an event than cry wolf.
- **It needs ~90 frames of calm to warm up** and assumes the feed starts calm. A
  camera already showing panic at startup will learn panic as its normal.
- **The `trained-model` backend does not work on real footage** (recall 0.000 on
  UMN) and is kept only for comparison. Retraining it on UMN inverts the failure
  rather than fixing it — see CLAUDE.md.
- **UCSD is a mismatched benchmark** — its anomalies are non-pedestrian objects
  (bikes, carts), not crowd panic; `evaluate.py` uses it for a reproducible
  baseline only.
- **Multi-camera throughput.** One stream runs ~15 fps on CPU and rooms share one
  YOLO model with no frame-skipping, so N rooms split that budget.

## Not built yet

- **Other devices joining a room by ID.** The ingest endpoint and capture page
  exist and work, but only over `localhost` — HTTPS is required before a phone can
  reach them, and there is no QR/join flow yet.
- **Alert delivery.** The alarm is a beep in whichever browser has the dashboard
  open. Nothing is pushed to the camera device or anywhere else.

## License

For educational / demonstration use.
