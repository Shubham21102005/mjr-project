"""
FastAPI server + dashboard for the Crowd Panic Detection System.

Two modes, both served from the same page:

  UPLOAD (the demo path) — POST a video file, the server runs the full pipeline
  over it in a background thread, and the browser gets back an annotated H.264
  clip plus a timeline of exactly where the crowd went into PANIC. This is how
  the detector gets demonstrated: you cannot stage real panic in front of a live
  camera, so a recorded clip is the honest way to show it working.

  LIVE (multi-camera) - a ROOM is a named group of CAMERAS. Cameras are either
  pulled by the server (PANIC_SOURCES: webcam index, file, stream URL) or
  pushed in by devices: a phone opens /camera, and its browser streams JPEG
  frames over WS /ws/ingest/{room}/{cam}. Two threads serve every camera: one
  runs the analysis pipeline at a steady rate, the other batches person
  detection across cameras (see live.py). The dashboard gets every camera's
  state and annotated frames over a single WebSocket (/ws/live), and a room
  alarms when any of its cameras is in PANIC.

Pulled sources are opt-in via PANIC_SOURCES (comma-separated `room=source` or
`room/cam=source`):
    PANIC_SOURCES="lobby=demo_crowd_30s.avi,lobby/door=0"

Phones need HTTPS for camera access, so put the server behind a tunnel:
    ngrok http 8000            (or: cloudflared tunnel --url http://localhost:8000)
Requests that come through a tunnel or from another device must carry the
access token (PANIC_TOKEN, or a random one printed at startup); see auth.py.

Run:
    venv/Scripts/python -m uvicorn server:app --host 127.0.0.1 --port 8000   # Windows
    venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8000       # macOS/Linux
Then open http://localhost:8000
"""
import asyncio
import io
import json
import os
import threading
import time
import urllib.parse
import urllib.request
import uuid
from contextlib import asynccontextmanager

import numpy as np
import cv2
import qrcode
import qrcode.image.svg
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from ultralytics import YOLO

import live
from analyze import analyze_video
from auth import TokenGate, load_token
from video_writer import ffmpeg_available

MODEL_WEIGHTS = os.environ.get("PANIC_MODEL", "yolov8n.pt")

HERE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(HERE, "uploads")
ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
MAX_UPLOAD_BYTES = 512 * 1024 * 1024      # 512 MB

# One shared YOLO model across worker threads (ultralytics inference is stateless
# per call, so sharing avoids loading the weights N times).
_yolo = YOLO(MODEL_WEIGHTS)

# Warm up the model ONCE here in the main thread. YOLO fuses its layers lazily on
# the first inference call, and fusing mutates the model in place (it deletes
# batchnorm layers). If two worker threads make their first call concurrently they
# race on that fuse — one deletes `.bn`, the other crashes with
# "'Conv' object has no attribute 'bn'". Fusing up front, single-threaded, makes
# every later per-thread call a no-op setup, so the workers never race.
_yolo(np.zeros((32, 32, 3), dtype=np.uint8), classes=[0], verbose=False)


# --------------------------------------------------------------------------- #
# Uploaded-clip analysis jobs
# --------------------------------------------------------------------------- #

JOBS = {}
JOBS_LOCK = threading.Lock()

# Only one analysis at a time. Each job already saturates the CPU with YOLO +
# dense flow; running several concurrently would make every one of them crawl
# and make the progress bar meaningless.
ANALYSIS_SEM = threading.Semaphore(1)


def _set_job(job_id, **fields):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(fields)


def _run_analysis(job_id, src_path, out_path):
    _set_job(job_id, status="queued", message="waiting for the analyzer")
    with ANALYSIS_SEM:
        started = time.time()
        _set_job(job_id, status="running", message="analyzing", started=started)

        def progress(done, total):
            elapsed = time.time() - started
            pct = (done / total * 100.0) if total else 0.0
            eta = (elapsed / done * (total - done)) if (done and total and total > done) else 0.0
            _set_job(job_id, frames_done=done, frames_total=total,
                     progress=round(pct, 1), eta=round(eta, 1),
                     fps_processing=round(done / elapsed, 2) if elapsed > 0 else 0.0)

        try:
            result = analyze_video(src_path, out_path, _yolo, progress_cb=progress)
        except Exception as e:
            _set_job(job_id, status="error", message=f"{type(e).__name__}: {e}",
                     progress=0.0)
            return

        result["elapsed"] = round(time.time() - started, 1)
        _set_job(job_id, status="done", progress=100.0, message="complete",
                 result=result)


# --------------------------------------------------------------------------- #
# Live cameras: registry + scheduler (see live.py), access token (see auth.py)
# --------------------------------------------------------------------------- #

REGISTRY = live.Registry()
SCHEDULER = DETECTOR = None
ACCESS_TOKEN, TOKEN_GENERATED = load_token()

MAX_FRAME_BYTES = 2 * 1024 * 1024        # one pushed JPEG; 640px at q0.6 is ~40 KB


@asynccontextmanager
async def lifespan(app: FastAPI):
    global SCHEDULER, DETECTOR
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Empty by default: there is no sensible default feed, and a source that
    # isn't there just shows up as a permanently dead tile.
    for room, cam, src in live.parse_sources(os.environ.get("PANIC_SOURCES", "")):
        try:
            REGISTRY.add_pull(room, cam, src)
        except (ValueError, live.CameraInUse) as e:
            print(f"[server] skipping PANIC_SOURCES entry {room}/{cam}: {e}")
    SCHEDULER = live.Scheduler(REGISTRY)
    DETECTOR = live.Detector(REGISTRY, _yolo)
    SCHEDULER.start()
    DETECTOR.start()

    pulled = [c.id for c in REGISTRY.cameras()]
    print(f"[server] pulled cameras: {pulled or '(none; phones can still join via /camera)'}")
    print(f"[server] access token for phones / tunnel: {ACCESS_TOKEN}"
          + ("  (random per run; set PANIC_TOKEN to keep it fixed)" if TOKEN_GENERATED else ""))
    if not ffmpeg_available():
        print("[server] WARNING: ffmpeg not on PATH; annotated clips may not play in the browser")
    yield
    SCHEDULER.running = False
    DETECTOR.running = False
    REGISTRY.stop()


app = FastAPI(title="Crowd Panic Detection", lifespan=lifespan)
app.add_middleware(TokenGate, token=ACCESS_TOKEN)


# --------------------------------------------------------------------------- #
# Upload API
# --------------------------------------------------------------------------- #

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        return JSONResponse(
            {"error": f"unsupported file type '{ext or '?'}'. "
                      f"Allowed: {', '.join(sorted(ALLOWED_EXT))}"},
            status_code=400)

    job_id = uuid.uuid4().hex[:12]
    src_path = os.path.join(UPLOAD_DIR, f"{job_id}_src{ext}")
    out_path = os.path.join(UPLOAD_DIR, f"{job_id}_annotated.mp4")

    # Stream to disk in chunks and enforce the cap as we go, so an oversized
    # upload can't balloon memory before we notice it.
    size = 0
    try:
        with open(src_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    out.close()
                    os.remove(src_path)
                    return JSONResponse(
                        {"error": f"file exceeds the {MAX_UPLOAD_BYTES // (1024*1024)} MB limit"},
                        status_code=413)
                out.write(chunk)
    finally:
        await file.close()

    if size == 0:
        os.remove(src_path)
        return JSONResponse({"error": "the uploaded file was empty"}, status_code=400)

    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "filename": file.filename,
            "size": size,
            "status": "queued",
            "progress": 0.0,
            "message": "queued",
            "frames_done": 0,
            "frames_total": 0,
            "result": None,
        }

    threading.Thread(target=_run_analysis, args=(job_id, src_path, out_path),
                     daemon=True).start()
    return {"job_id": job_id, "filename": file.filename, "size": size}


@app.get("/api/jobs")
def list_jobs():
    with JOBS_LOCK:
        return {"jobs": [
            {k: v for k, v in j.items() if k != "result"} for j in JOBS.values()
        ]}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return JSONResponse({"error": "unknown job"}, status_code=404)
        return dict(job)


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None or job.get("status") != "done":
        return JSONResponse({"error": "result not ready"}, status_code=404)
    path = os.path.join(UPLOAD_DIR, f"{job_id}_annotated.mp4")
    if not os.path.exists(path):
        return JSONResponse({"error": "annotated video missing"}, status_code=404)
    # FileResponse honours Range requests, which is what lets the player seek
    # straight to a panic segment instead of downloading the whole clip first.
    # Serve it INLINE: passing filename= would set Content-Disposition:attachment,
    # which is a download hint, not what a <video> source should send. The
    # "download" link in the page uses the anchor's download attribute instead.
    return FileResponse(path, media_type="video/mp4", headers={
        "Content-Disposition": f'inline; filename="panic_analysis_{job_id}.mp4"',
    })


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.pop(job_id, None)
    if job is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    for f in os.listdir(UPLOAD_DIR):
        if f.startswith(job_id):
            try:
                os.remove(os.path.join(UPLOAD_DIR, f))
            except OSError:
                pass
    return {"deleted": job_id}


# --------------------------------------------------------------------------- #
# Live API
# --------------------------------------------------------------------------- #

_ngrok_cache = {"t": 0.0, "url": None}


def detect_public_url():
    """The HTTPS address phones should use: PANIC_PUBLIC_URL if set, otherwise
    whatever a locally running ngrok agent reports on its inspection API."""
    env = os.environ.get("PANIC_PUBLIC_URL", "").strip().rstrip("/")
    if env:
        return env
    now = time.monotonic()
    if now - _ngrok_cache["t"] < 5.0:
        return _ngrok_cache["url"]
    url = None
    try:
        with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=0.4) as r:
            tunnels = json.load(r).get("tunnels", [])
        https = [t["public_url"] for t in tunnels
                 if str(t.get("public_url", "")).startswith("https://")]
        url = https[0] if https else None
    except Exception:
        url = None
    _ngrok_cache.update(t=now, url=url)
    return url


@app.get("/api/rooms")
def api_rooms():
    return {"rooms": REGISTRY.snapshot(), "ffmpeg": ffmpeg_available(),
            "public_url": detect_public_url(), "live_fps": live.LIVE_FPS}


@app.get("/api/join")
def api_join(room: str, cam: str, base: str = ""):
    """The link (and its QR code) a phone opens to join room/cam."""
    if not (live.valid_name(room) and live.valid_name(cam)):
        return JSONResponse({"error": "room and camera names may use letters, digits, "
                                      "'-' and '_' (max 32)"}, status_code=400)
    base = (base or detect_public_url() or "").strip().rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return JSONResponse({"error": "no public URL: start a tunnel (ngrok http 8000) "
                                      "or enter the address phones should use"},
                            status_code=400)
    query = urllib.parse.urlencode({"room": room, "cam": cam, "token": ACCESS_TOKEN})
    url = f"{parsed.scheme}://{parsed.netloc}/camera?{query}"
    img = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, box_size=8, border=2)
    buf = io.BytesIO()
    img.save(buf)
    return {"url": url, "svg": buf.getvalue().decode("utf-8"),
            "secure": parsed.scheme == "https"
                      or parsed.hostname in ("localhost", "127.0.0.1")}


@app.websocket("/ws/ingest/{room}/{cam}")
async def ingest(websocket: WebSocket, room: str, cam: str):
    """A device pushes its camera into room/cam.

    Protocol: the device sends one binary JPEG frame, the server replies with a
    JSON ack carrying this camera's label and the room's alarm, then the device
    sends the next frame. Waiting for the ack is the device's backpressure: on a
    slow uplink it sends less often instead of piling up seconds of lag.

    Text messages are control messages: {"type": "reset"} asks the server to
    relearn this camera's baseline (the phone sends it after switching between
    front and back camera, since that is a different scene).
    """
    await websocket.accept()
    try:
        camera, gen = REGISTRY.attach_push(room, cam)
    except ValueError as e:
        await websocket.close(code=4400, reason=str(e))
        return
    except live.CameraInUse as e:
        await websocket.close(code=4409, reason=str(e))
        return

    print(f"[server] camera joined: {camera.id}")
    try:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if camera.conn_gen != gen:
                # A newer connection from this device took over (see attach_push).
                await websocket.close(code=4409, reason="replaced by a newer connection")
                break
            data = msg.get("bytes")
            if data is None:
                try:
                    if json.loads(msg.get("text") or "{}").get("type") == "reset":
                        camera.reset()
                except (ValueError, AttributeError):
                    pass
                continue
            if len(data) > MAX_FRAME_BYTES:
                await websocket.close(code=4413, reason="frame too large")
                break
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                camera.slot.put(frame)
            s = camera.snapshot()
            panic_cams = REGISTRY.room_alarm(room)
            await websocket.send_json({
                "label": s["label"], "score": s["score"], "warming_up": s["warming_up"],
                "fps": s["fps"], "room_alarm": bool(panic_cams), "panic_cams": panic_cams,
                "decoded": frame is not None,
            })
    except (WebSocketDisconnect, OSError, RuntimeError):
        pass
    finally:
        REGISTRY.detach_push(camera, gen)
        print(f"[server] camera left: {camera.id}")


@app.websocket("/ws/live")
async def live_feed(websocket: WebSocket):
    """Everything the dashboard needs on ONE socket: JSON state for all rooms
    (~5x/s) and each camera's annotated JPEG whenever it changes.

    Frames go over this WebSocket instead of one MJPEG <img> per camera because
    browsers allow only ~6 concurrent HTTP/1.1 connections per host: with
    MJPEG, a 7th camera would never load and would also block every fetch().

    Binary message layout: [1 byte id length][camera id, utf-8][jpeg bytes].
    """
    await websocket.accept()
    sent = {}
    last_state = 0.0
    try:
        while True:
            now = time.monotonic()
            if now - last_state >= 0.2:
                await websocket.send_json({"type": "state", "rooms": REGISTRY.snapshot()})
                last_state = now
            for camera in REGISTRY.cameras():
                seq, jpeg = camera.latest_jpeg()
                if jpeg is not None and sent.get(camera.id) != seq:
                    sent[camera.id] = seq
                    cid = camera.id.encode("utf-8")
                    await websocket.send_bytes(bytes([len(cid)]) + cid + jpeg)
            await asyncio.sleep(0.03)
    except (WebSocketDisconnect, OSError, RuntimeError):
        pass


@app.get("/video/{room}/{cam}")
async def video(room: str, cam: str):
    """MJPEG for one camera, for viewing a single feed directly (or in VLC)."""
    camera = REGISTRY.get(room, cam)
    if camera is None:
        return HTMLResponse(f"unknown camera {room}/{cam}", status_code=404)

    async def gen():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        last = None
        while REGISTRY.get(room, cam) is camera:
            seq, jpeg = camera.latest_jpeg()
            if jpeg is not None and seq != last:
                last = seq
                yield boundary + jpeg + b"\r\n"
            await asyncio.sleep(0.03)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


def _page(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return f.read()


@app.get("/camera", response_class=HTMLResponse)
def camera_page():
    return _page("camera.html")


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return _page("dashboard.html")
