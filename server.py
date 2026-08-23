"""
FastAPI server + dashboard for the Crowd Panic Detection System.

Two modes, both served from the same page:

  UPLOAD (the demo path) — POST a video file, the server runs the full pipeline
  over it in a background thread, and the browser gets back an annotated H.264
  clip plus a timeline of exactly where the crowd went into PANIC. This is how
  the detector gets demonstrated: you cannot stage real panic in front of a live
  camera, so a recorded clip is the honest way to show it working.

  LIVE — each "room" is one camera feed (a webcam index or a file path the
  server can open). A background worker thread per room runs the same pipeline
  and publishes the latest annotated JPEG + state; the page streams each as
  MJPEG and receives labels/metrics over a WebSocket, flashing red and sounding
  an alarm on PANIC.

Live rooms are opt-in via the PANIC_SOURCES env var (comma-separated
name=source), because there is no sensible default camera to point at:
    PANIC_SOURCES="lobby=demo_crowd_30s.avi,webcam=0"

Run:
    venv/Scripts/uvicorn server:app --host 127.0.0.1 --port 8000   # Windows
    venv/bin/uvicorn server:app --host 127.0.0.1 --port 8000       # macOS/Linux
Then open http://localhost:8000
"""
import asyncio
import os
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from ultralytics import YOLO

from analyze import analyze_video
from classifier import PanicClassifier
from detect import extract_boxes, draw_hud
from metrics import compute_metrics
from video_writer import ffmpeg_available

FLOW_SCALE = 0.5
CONF = 0.4
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    for room, src in parse_sources().items():
        w = CameraWorker(room, src)
        w.start()
        WORKERS[room] = w
    print(f"[server] live rooms: {list(WORKERS) or '(none — set PANIC_SOURCES to add some)'}")
    if not ffmpeg_available():
        print("[server] WARNING: ffmpeg not on PATH; annotated clips may not play in the browser")
    yield
    for w in WORKERS.values():
        w.running = False


# --------------------------------------------------------------------------- #
# Live camera rooms
# --------------------------------------------------------------------------- #

class CameraWorker(threading.Thread):
    """Reads a source, runs the pipeline, and publishes latest frame + state."""

    def __init__(self, room, source):
        super().__init__(daemon=True)
        self.room = room
        self.source = int(source) if str(source).isdigit() else source
        self.is_file = not str(source).isdigit()
        self.classifier = PanicClassifier()
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.state = {"room": room, "label": "NORMAL", "panic_ratio": 0.0,
                      "backend": self.classifier.backend,
                      "degraded": self.classifier.degraded,
                      "metrics": {}, "online": False}
        self.running = True

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def run(self):
        cap = cv2.VideoCapture(self.source)
        prev = None
        metrics, label, flow = {}, "NORMAL", None
        while self.running:
            ret, frame = cap.read()
            if not ret:
                if self.is_file:                       # loop video files for a live demo
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    prev = None
                    continue
                time.sleep(0.2)
                continue

            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            H, W = frame.shape[:2]

            boxes = extract_boxes(_yolo(frame, classes=[0], verbose=False), CONF)
            small = cv2.resize(frame, (0, 0), fx=FLOW_SCALE, fy=FLOW_SCALE)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                metrics = compute_metrics(flow, boxes, (H, W))
                label = self.classifier.update(metrics)
            prev = gray

            disp = draw_hud(frame.copy(), boxes, flow, metrics, label,
                            self.classifier.panic_ratio(), FLOW_SCALE, False)
            ok, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                with self.lock:
                    self.latest_jpeg = buf.tobytes()
                    self.state = {
                        "room": self.room, "label": label,
                        "panic_ratio": round(self.classifier.panic_ratio(), 2),
                        "backend": self.classifier.backend,
                        "degraded": self.classifier.degraded,
                        "score": round(self.classifier.anomaly_score, 2),
                        "warming_up": self.classifier.warming_up,
                        "metrics": {k: round(float(v), 3) for k, v in metrics.items()},
                        "online": True,
                    }
        cap.release()


def parse_sources():
    """Live rooms from PANIC_SOURCES. Empty by default — there is no sensible
    default feed, and a room pointing at a file that isn't there just shows up
    as a permanently dead tile."""
    raw = os.environ.get("PANIC_SOURCES", "").strip()
    rooms = {}
    for part in raw.split(","):
        if "=" in part:
            name, src = part.split("=", 1)
            if name.strip() and src.strip():
                rooms[name.strip()] = src.strip()
    return rooms


app = FastAPI(title="Crowd Panic Detection", lifespan=lifespan)
WORKERS = {}


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

@app.get("/rooms")
def rooms():
    return {"rooms": list(WORKERS), "ffmpeg": ffmpeg_available()}


@app.get("/video/{room}")
def video(room: str):
    worker = WORKERS.get(room)
    if worker is None:
        return HTMLResponse(f"unknown room {room}", status_code=404)

    def gen():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            frame = worker.latest_jpeg
            if frame is not None:
                yield boundary + frame + b"\r\n"
            time.sleep(0.04)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws/{room}")
async def ws(websocket: WebSocket, room: str):
    await websocket.accept()
    worker = WORKERS.get(room)
    try:
        while True:
            if worker is not None:
                await websocket.send_json(worker.snapshot())
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass


@app.get("/", response_class=HTMLResponse)
def dashboard():
    with open(os.path.join(HERE, "dashboard.html"), encoding="utf-8") as f:
        return f.read()
