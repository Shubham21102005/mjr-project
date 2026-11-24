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
        w = PullRoom(room, src)
        w.start()
        WORKERS[room] = w
    print(f"[server] pull rooms: {list(WORKERS) or '(none)'}  "
          f"(push rooms are created on demand from /camera)")
    if not ffmpeg_available():
        print("[server] WARNING: ffmpeg not on PATH; annotated clips may not play in the browser")
    yield
    for w in WORKERS.values():
        w.running = False


# --------------------------------------------------------------------------- #
# Live rooms
# --------------------------------------------------------------------------- #

# How long a browser-pushed room survives with no clients attached. This grace
# period is not politeness: tearing the room down the instant a tab reloads would
# throw away the scene baseline and force another 90-frame warm-up before the
# detector could fire again.
ROOM_IDLE_GRACE = 60.0


class RoomBase(threading.Thread):
    """One live feed: owns a classifier and publishes annotated frames + state.

    Subclasses differ ONLY in where frames come from. process() is the single
    implementation of the pipeline, so a frame pushed from a browser and a frame
    pulled from a webcam are scored by identical code -- the same invariant that
    makes detect.py, analyze.py and evaluate.py share compute_metrics.
    """

    kind = "base"

    def __init__(self, room):
        super().__init__(daemon=True)
        self.room = room
        self.classifier = PanicClassifier()
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.running = True
        self.prev_gray = None
        self.clients = 0
        self.idle_since = None
        self.received = 0
        self.dropped = 0
        self.fps_in = 0.0
        self._t_last = None
        self.state = self._state("NORMAL", {}, online=False)

    # -- state -------------------------------------------------------------- #

    def _state(self, label, metrics, online=True):
        bl = self.classifier.baseline
        return {
            "room": self.room,
            "kind": self.kind,
            "label": label,
            "panic_ratio": round(self.classifier.panic_ratio(), 2),
            "backend": self.classifier.backend,
            "degraded": self.classifier.degraded,
            "score": round(self.classifier.anomaly_score, 2),
            "threshold": bl.k if bl is not None else None,
            "warming_up": self.classifier.warming_up,
            "warmup_have": len(bl.history) if bl is not None else 0,
            "warmup_total": bl.warmup if bl is not None else 0,
            "metrics": {k: round(float(v), 3) for k, v in metrics.items()},
            "clients": self.clients,
            "received": self.received,
            "dropped": self.dropped,
            "fps_in": round(self.fps_in, 1),
            "online": online,
        }

    def snapshot(self):
        # Connection counters are facts about right now, not about the last frame
        # that happened to be processed. Reading them off the frozen state dict
        # left `clients` stale after a device disconnected, so overlay them live.
        with self.lock:
            s = dict(self.state)
        s["clients"] = self.clients
        s["received"] = self.received
        s["dropped"] = self.dropped
        s["fps_in"] = round(self.fps_in, 1)
        if self.kind == "push":
            s["online"] = self.clients > 0
        return s

    def reset_baseline(self):
        """Re-learn what normal looks like for this feed. Needed when the camera
        was pointed at something busy during warm-up and learned chaos as calm."""
        with self.lock:
            self.classifier = PanicClassifier()
            self.prev_gray = None

    def stop(self):
        self.running = False

    # -- the pipeline, defined once ----------------------------------------- #

    def process(self, frame):
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        H, W = frame.shape[:2]

        boxes = extract_boxes(_yolo(frame, classes=[0], verbose=False), CONF)
        small = cv2.resize(frame, (0, 0), fx=FLOW_SCALE, fy=FLOW_SCALE)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        metrics, flow = {}, None
        label = self.classifier.state
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            metrics = compute_metrics(flow, boxes, (H, W))
            label = self.classifier.update(metrics)
        self.prev_gray = gray

        disp = draw_hud(frame.copy(), boxes, flow, metrics, label,
                        self.classifier.panic_ratio(), FLOW_SCALE, False)
        ok, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 75])

        now = time.time()
        if self._t_last is not None:
            dt = now - self._t_last
            if dt > 0:
                inst = 1.0 / dt
                self.fps_in = inst if not self.fps_in else 0.8 * self.fps_in + 0.2 * inst
        self._t_last = now

        if ok:
            with self.lock:
                self.latest_jpeg = buf.tobytes()
                self.state = self._state(label, metrics)


class PullRoom(RoomBase):
    """Server-pulled feed: a file path or a webcam index the server opens itself."""

    kind = "pull"

    def __init__(self, room, source):
        super().__init__(room)
        self.source = int(source) if str(source).isdigit() else source
        self.is_file = not str(source).isdigit()

    def run(self):
        cap = cv2.VideoCapture(self.source)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                if self.is_file:                   # loop video files for a live demo
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.prev_gray = None
                    continue
                time.sleep(0.2)
                continue
            self.received += 1
            self.process(frame)
        cap.release()


class PushRoom(RoomBase):
    """Browser-pushed feed. JPEG frames arrive over WS /ingest/{room}."""

    kind = "push"

    def __init__(self, room):
        super().__init__(room)
        self.cond = threading.Condition()
        self.pending = None

    def submit(self, data):
        """Hand the newest frame to the worker.

        Overwriting `pending` IS the backpressure. The pipeline runs ~15 fps on
        CPU; if the browser sends faster, queueing would push the annotated view
        further behind real time every second. Dropping the stale frame instead
        keeps latency bounded and simply lowers the effective rate.
        """
        with self.cond:
            if self.pending is not None:
                self.dropped += 1
            self.pending = data
            self.received += 1
            self.cond.notify()

    def stop(self):
        self.running = False
        with self.cond:
            self.cond.notify_all()

    def run(self):
        while self.running:
            with self.cond:
                while self.pending is None and self.running:
                    self.cond.wait(0.5)
                data, self.pending = self.pending, None
            if not data:
                continue
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                self.process(frame)


def parse_sources():
    """Pull rooms from PANIC_SOURCES. Empty by default -- there is no sensible
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
WORKERS_LOCK = threading.Lock()


def get_or_create_push_room(room):
    with WORKERS_LOCK:
        r = WORKERS.get(room)
        if r is None or not r.is_alive():
            r = PushRoom(room)
            r.start()
            WORKERS[room] = r
        return r


def reap_idle_rooms():
    """Stop push rooms whose last client left more than ROOM_IDLE_GRACE ago."""
    now = time.time()
    with WORKERS_LOCK:
        for name, r in list(WORKERS.items()):
            if r.kind != "push" or r.clients > 0 or r.idle_since is None:
                continue
            if now - r.idle_since > ROOM_IDLE_GRACE:
                r.stop()
                WORKERS.pop(name, None)


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
    reap_idle_rooms()
    with WORKERS_LOCK:
        items = []
        for name, w in WORKERS.items():
            snap = w.snapshot()
            items.append({
                "name": name,
                "kind": w.kind,
                "clients": w.clients,
                "online": snap.get("online", False),
            })
    return {"rooms": items, "ffmpeg": ffmpeg_available()}


@app.get("/video/{room}")
def video(room: str):
    """Annotated MJPEG for a room.

    The room is re-resolved on every iteration rather than bound once: a push
    room usually does not exist yet when the page that displays it first asks
    for the stream, and binding once would 404 forever. Instead we hold the
    connection open and start emitting as soon as the room appears.

    This generator is ASYNC on purpose. As a sync generator it was iterated via
    Starlette's threadpool, costing a threadpool round-trip ~25x a second per
    viewer; with a browser watching a feed while a device pushed to it, that
    starved the event loop badly enough that the ingest socket missed its
    keepalive pongs and the server killed the upload mid-stream. Awaiting here
    keeps everything on the loop.

    It also only emits when the pipeline has actually produced a NEW frame.
    Re-sending the same JPEG 25x/sec when the pipeline only manages ~15 fps was
    pure waste on both bandwidth and loop time.
    """
    async def gen():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        waited = 0.0
        last = None
        while True:
            worker = WORKERS.get(room)
            frame = worker.latest_jpeg if worker is not None else None
            if frame is not None:
                if frame is not last:
                    yield boundary + frame + b"\r\n"
                    last = frame
                waited = 0.0
            else:
                waited += 0.03
                if waited > 30.0:        # room never showed up; let the client go
                    break
            await asyncio.sleep(0.03)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws/{room}")
async def ws(websocket: WebSocket, room: str):
    """Live labels + metrics for a dashboard tile.

    Same re-resolution point as /video: the previous version captured the worker
    once at connect time, so a room created later stayed None forever and the
    socket silently sent nothing for the life of the connection.
    """
    await websocket.accept()
    try:
        while True:
            worker = WORKERS.get(room)
            if worker is not None:
                await websocket.send_json(worker.snapshot())
            else:
                await websocket.send_json({"room": room, "label": "NORMAL",
                                           "online": False, "metrics": {}})
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass


@app.websocket("/ingest/{room}")
async def ingest(websocket: WebSocket, room: str):
    """Receive camera frames pushed from a browser.

    Two independent tasks run for the life of the connection: one draining
    incoming JPEG frames into the room, one pushing state back. They are
    deliberately decoupled -- the sender must keep reporting label and score
    even if the camera stalls, and that same downstream channel is what device
    alerts will use once multiple devices can share a room.
    """
    await websocket.accept()
    r = get_or_create_push_room(room)
    r.clients += 1
    r.idle_since = None
    print(f"[server] room '{room}': device connected ({r.clients} total)")

    async def pump_in():
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                return
            data = msg.get("bytes")
            if data:
                r.submit(data)

    async def pump_out():
        while True:
            await websocket.send_json(r.snapshot())
            await asyncio.sleep(0.2)

    tasks = [asyncio.create_task(pump_in()), asyncio.create_task(pump_out())]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        for t in tasks:
            t.cancel()
        r.clients = max(0, r.clients - 1)
        if r.clients == 0:
            r.idle_since = time.time()
            with r.lock:
                r.state["online"] = False
        print(f"[server] room '{room}': device disconnected ({r.clients} left)")


@app.post("/api/rooms/{room}/recalibrate")
def recalibrate(room: str):
    """Throw away this feed's learned baseline and start learning again.

    scene-baseline assumes the feed starts calm. If warm-up happened while the
    scene was busy, the detector has learned chaos as normal and will not fire;
    this is the escape hatch.
    """
    worker = WORKERS.get(room)
    if worker is None:
        return JSONResponse({"error": f"unknown room {room}"}, status_code=404)
    worker.reset_baseline()
    return {"ok": True, "room": room}


@app.get("/camera", response_class=HTMLResponse)
def camera_page():
    with open(os.path.join(HERE, "camera.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/", response_class=HTMLResponse)
def dashboard():
    with open(os.path.join(HERE, "dashboard.html"), encoding="utf-8") as f:
        return f.read()
