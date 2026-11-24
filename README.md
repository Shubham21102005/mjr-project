# Crowd Panic Detection System

Detect when a crowd shifts from normal movement into **panic** from any video
feed (an uploaded clip, a webcam, a video file, or **phone cameras**) and raise
an alert in real time. Ships as a standalone CLI viewer and a web dashboard that
analyses uploaded clips and monitors live **multi-camera rooms**.

Highlights:

- **Upload a clip** and get an annotated video plus a timeline of every panic episode.
- **Turn any phone into a camera** by scanning a QR code: no app to install.
- **Group cameras into rooms.** The room alarms when any camera in it detects panic, on the dashboard and on every phone in the room.
- **Scene-relative detection.** Each camera learns what calm looks like for its own view, so one setting works across cameras at different distances.

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

**For phone cameras, you also need a tunnel** that gives the server an HTTPS
address, because phone browsers only allow camera access over HTTPS. Either
[ngrok](https://ngrok.com/download) or
[cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
works. Neither is needed for uploads, the CLI, or server-opened sources.

Throughout the rest of this README, `PY` means `venv\Scripts\python` on Windows
and `venv/bin/python` on macOS/Linux.

## Web dashboard

```bash
# Windows
venv\Scripts\python -m uvicorn server:app --host 127.0.0.1 --port 8000
# macOS / Linux
venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Then open [http://localhost:8000](http://localhost:8000).
The page has two sections: **Analyse a video clip** (upload) and **Live cameras** (rooms, phones and server-opened sources).

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

### Monitor live cameras (multi-camera rooms, phones)

A **room** is a named group of **cameras**.
The room raises an alarm when any of its cameras is in PANIC, and every camera keeps its own scene baseline.
Cameras come from two places:

- **Phones (pushed).** A phone opens the `/camera` page, picks a room and a camera name, and streams its camera into the server.
  Rooms are created the first time a phone joins one.
- **Sources the server opens (pulled).** Webcams, files or stream URLs listed in `PANIC_SOURCES` at startup.

For how it works inside (architecture, protocols, security, tuning and troubleshooting), see **[docs/multi-camera.md](docs/multi-camera.md)**.

#### Connect a phone

Phone browsers only allow camera access over **HTTPS**, so the server needs an HTTPS address.
A tunnel provides one.

1. Start the server:
   ```bash
   # Windows (PowerShell)
   $env:PANIC_TOKEN="pick-a-secret"; venv\Scripts\python -m uvicorn server:app --host 127.0.0.1 --port 8000
   # macOS / Linux
   PANIC_TOKEN="pick-a-secret" venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8000
   ```
2. In a second terminal, start a tunnel to port 8000:
   ```bash
   ngrok http 8000
   # or, with no account and no data cap:
   cloudflared tunnel --url http://localhost:8000
   ```
3. Open http://localhost:8000 on the computer and expand **+ Add a phone camera**.
   With ngrok, the https address fills in automatically.
   With cloudflared, paste the `https://….trycloudflare.com` address it printed.
4. Enter a room and a camera name, then click **Show QR code**.
5. Scan the QR code with the phone, open the link, tap **Start** and allow camera access.
   If ngrok shows a "You are about to visit" warning page first, tap **Visit Site**.
6. Prop the phone up so it stays still.
   It appears on the dashboard within a second, learns what calm looks like for about 9 seconds, then starts monitoring.

Repeat steps 4 and 5 for each extra phone: same room name to group cameras, different camera names.
When any camera in a room goes into PANIC, its tile and the room flash red on the dashboard, the dashboard beeps, and every phone in that room flashes, vibrates and beeps.

**Access token.**
Anything that reaches the server through the tunnel, or from another device, needs the access token.
The QR link carries it and the phone keeps it in a cookie after the first visit.
Set `PANIC_TOKEN` to keep the same token across restarts.
Without it, a random token is printed at startup and old QR links stop working after a restart.
The dashboard opened directly on the computer (`localhost`) needs no token.

**Bandwidth.**
A phone sends about 10 frames per second at 640 px, roughly 200-400 KB/s, which is around 0.7-1.4 GB per hour.
The ngrok free plan has a monthly data allowance, so for long sessions prefer cloudflared.

If something doesn't work, the [troubleshooting table](docs/multi-camera.md#troubleshooting) covers the common cases.

#### Server-opened sources

`PANIC_SOURCES` is a comma-separated list of `room=source` or `room/camera=source` entries.
A source is a webcam index, a file path (looped and played at its own frame rate) or a stream URL OpenCV can open.

```bash
# Windows (PowerShell)
$env:PANIC_SOURCES="lobby=demo_crowd_30s.avi,lobby/webcam=0"; venv\Scripts\python -m uvicorn server:app --port 8000
# macOS / Linux
PANIC_SOURCES="lobby=demo_crowd_30s.avi,lobby/webcam=0" venv/bin/python -m uvicorn server:app --port 8000
```

A bare `room=source` becomes the camera `main` in that room.

#### Simulate phones without a phone

`fake_phone.py` streams a video file into a room through exactly the same WebSocket and protocol the phone page uses.
Use it to try multi-camera rooms, to load-test, or to demo an alarm on the live path:

```bash
PY fake_phone.py --source panic_test.avi --room lobby --cam sim1 --every-frame
PY fake_phone.py --source demo_crowd_30s.avi --room lobby --cam sim2 --loop
# through a tunnel:
PY fake_phone.py --url wss://xxxx.ngrok-free.app --token pick-a-secret --source panic_test.avi --every-frame
```

#### Live tuning

| Env var            | Default | Meaning                                                    |
| ------------------ | ------- | ---------------------------------------------------------- |
| `PANIC_TOKEN`      | random  | Access token for tunnel / other-device access              |
| `PANIC_PUBLIC_URL` | (auto)  | The https address phones should use (else asks ngrok)      |
| `PANIC_LIVE_FPS`   | `10`    | Per-camera analysis rate (optical flow + classifier)       |
| `PANIC_DETECT_FPS` | `3`     | Per-camera person-detection rate (YOLO, batched)           |

### HTTP API

| Method     | Path                     | Purpose                                     |
| ---------- | ------------------------ | ------------------------------------------- |
| `POST`   | `/api/upload`          | Upload a clip; returns a `job_id`         |
| `GET`    | `/api/jobs/{id}`       | Job status, progress, and the full result   |
| `GET`    | `/api/jobs/{id}/video` | The annotated MP4 (supports Range requests) |
| `DELETE` | `/api/jobs/{id}`       | Drop a job and its files                    |
| `GET`    | `/api/rooms`           | All rooms and cameras with live state       |
| `GET`    | `/api/join?room=&cam=` | Join link + QR code (SVG) for a phone       |
| `GET`    | `/camera`              | The phone camera page                       |
| `WS`     | `/ws/ingest/{room}/{cam}` | A device pushes JPEG frames in; JSON acks |
| `WS`     | `/ws/live`             | Dashboard feed: state + annotated frames    |
| `GET`    | `/video/{room}/{cam}`  | MJPEG stream of one camera                  |

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

## Tests

```bash
PY -m pip install -r requirements-dev.txt
PY -m pytest
```

`test_live.py` covers the multi-camera core with a stub detector and runs in seconds.
`test_server.py` runs the real app in-process with real YOLO and pushes real frames through the phone ingest path.
To try the live path by hand without a phone, use `fake_phone.py` (see above) or Chrome's fake camera, described in [docs/multi-camera.md](docs/multi-camera.md#testing).

## Project layout

| Area          | Files                                                                    |
| ------------- | ------------------------------------------------------------------------ |
| Core pipeline | `metrics.py`, `classifier.py`, `baseline.py`                       |
| CLI viewer    | `detect.py` (owns `extract_boxes` + `draw_hud`, reused everywhere) |
| Web app       | `server.py` (FastAPI), `dashboard.html`, `auth.py` (access token)    |
| Live cameras  | `live.py` (rooms, sources, analysis + detection threads), `camera.html` (phone page), `fake_phone.py` |
| Docs          | `docs/multi-camera.md` (live cameras in depth)                        |
| Tests         | `test_live.py`, `test_server.py` (`PY -m pytest`, after `pip install -r requirements-dev.txt`) |
| Clip analysis | `analyze.py`, `video_writer.py` (ffmpeg/libx264 encoding)            |
| Training      | `train_classifier.py`, `synth.py` (synthetic clip generator)         |
| Evaluation    | `evaluate.py`, `visualize.py`, `umn_labels.py` (UMN ground truth)  |
| Test assets   | `make_test_video.py`, demo videos, `yolov8n.pt`                      |

## Limitations

- **Validated on real panic footage.** Measured against the UMN Unusual Crowd
  Activity dataset (11 real crowd escapes; ground truth recovered by
  `umn_labels.py` from the video's own burned-in banner), scoring each scene
  independently:

  | scene                            | old `trained-model` F1 | `scene-baseline` F1 | Recall          |
  | -------------------------------- | ------------------------ | --------------------- | --------------- |
  | 1                                | 0.000                    | 0.845                 | 0.925           |
  | 2                                | 0.000                    | 0.591                 | 0.545           |
  | 3 *(held out from all tuning)* | 0.000                    | 0.610                 | **1.000** |

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
- **Multi-camera throughput.** Analysis (optical flow + classifier) and person
  detection (YOLO) run on separate threads. Analysis holds each camera at a
  steady `PANIC_LIVE_FPS`; detection is batched across cameras at
  `PANIC_DETECT_FPS`. On a 2-core laptop, three cameras held ~10 fps each. Past
  the CPU budget, every camera slows down evenly.
- **Phones must stay still.** A handheld phone's own shake produces motion
  everywhere in the frame, which reads exactly like panic. Mount or prop it.
  Switching lenses (Flip) or rotating the phone restarts that camera's warm-up.

## Not built yet

- **Alert delivery beyond open pages.** The alarm reaches the dashboard and every
  phone in the room, but only while those pages are open. Nothing is sent by
  push notification, SMS or email.
- **Camera-shake compensation.** Subtracting the global (median) flow before
  computing metrics would tolerate handheld phones; not done yet.

## License

For educational / demonstration use.
