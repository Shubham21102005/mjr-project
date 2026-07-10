"""
FastAPI server + live dashboard for the Crowd Panic Detection System.

Each "room" is one camera feed (a webcam index or a video file). A background
worker thread per room runs the full pipeline (YOLO + optical flow + trained
classifier) and keeps the latest annotated JPEG and state. The browser dashboard
shows every room in a grid, streams each annotated feed as MJPEG, and receives
live metrics + the NORMAL/PANIC label over a WebSocket. On PANIC it flashes the
tile red and sounds an alarm.

Configure rooms with the PANIC_SOURCES env var (comma-separated name=source):
    PANIC_SOURCES="cam1=panic_test.avi,lobby=demo_crowd.mp4,webcam=0"
Default is a single room 'cam1' pointing at panic_test.avi.

Run:
    venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
Then open http://localhost:8000
"""
import asyncio
import os
import threading
import time

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO

from metrics import compute_metrics
from classifier import PanicClassifier
from detect import extract_boxes, draw_hud

FLOW_SCALE = 0.5
CONF = 0.4
MODEL_WEIGHTS = os.environ.get("PANIC_MODEL", "yolov8n.pt")

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
                      "backend": self.classifier.backend, "metrics": {}, "online": False}
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
                        "metrics": {k: round(float(v), 3) for k, v in metrics.items()},
                        "online": True,
                    }
        cap.release()


def parse_sources():
    raw = os.environ.get("PANIC_SOURCES", "cam1=panic_test.avi")
    rooms = {}
    for part in raw.split(","):
        if "=" in part:
            name, src = part.split("=", 1)
            rooms[name.strip()] = src.strip()
    return rooms


app = FastAPI(title="Crowd Panic Detection")
WORKERS = {}


@app.on_event("startup")
def _start():
    for room, src in parse_sources().items():
        w = CameraWorker(room, src)
        w.start()
        WORKERS[room] = w
    print(f"[server] started rooms: {list(WORKERS)}")


@app.get("/rooms")
def rooms():
    return {"rooms": list(WORKERS)}


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
    with open(os.path.join(os.path.dirname(__file__), "dashboard.html")) as f:
        return f.read()
