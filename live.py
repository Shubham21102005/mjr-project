"""
Live multi-camera monitoring: frame sources, per-camera pipelines, the room
registry, and the single inference scheduler that drives them all.

MODEL
-----
A ROOM is a named group of cameras (e.g. "lobby"). A CAMERA is one feed inside
a room. Cameras come from two kinds of source:

  pull  - the server opens the source itself (webcam index, file, RTSP/HTTP URL),
          configured at startup with PANIC_SOURCES.
  push  - a device (a phone browser, or fake_phone.py) connects to
          WS /ws/ingest/{room}/{cam} and pushes JPEG frames in. Rooms and
          cameras are created on the fly the first time a device joins.

Every camera owns its OWN PanicClassifier. The default scene-baseline backend
learns what THIS feed looks like when calm, so a baseline must never be shared
between cameras, even cameras in the same room. The room's alarm is simply
"any camera in the room is in PANIC".

LATEST-FRAME SEMANTICS
----------------------
Each source writes into a single-slot FrameSlot: a new frame overwrites the
previous one, nothing is ever queued. If the analyser falls behind, it skips
stale frames instead of building up seconds of lag.

TWO THREADS, DECOUPLED
----------------------
Instead of one thread per camera all fighting over the CPU and the shared YOLO
model, there are exactly two:

  Scheduler - the ANALYSIS path. Every camera that has a new frame and is due
      (capped at LIVE_FPS) gets optical flow, metrics, classification and HUD
      rendering, in a small thread pool (OpenCV releases the GIL).
  Detector  - person detection. Collects the latest frame of each camera
      (capped at DETECT_FPS per camera), runs ONE batched YOLO call over all of
      them, and hands each camera its newest boxes.

They are separate because YOLO costs ~10x what flow does on CPU (measured
250-400 ms vs ~30 ms per frame on a 2-core laptop), and because the analysis
cadence must stay STEADY: avg_speed is motion per processed frame, so if the
interval between processed frames swung with YOLO's batch cost, the scene
baseline would be learning that jitter. The default scene-baseline decision
uses only flow features, so boxes that are a few hundred ms old change nothing
in the decision; they feed the HUD and crowd_density/person_count.
"""
import os
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import cv2

from classifier import PanicClassifier
from detect import extract_boxes, draw_hud
from metrics import compute_metrics

FLOW_SCALE = 0.5
CONF = 0.4

# Per-camera analysis rate cap. Phones send at this rate too (camera.html default).
LIVE_FPS = float(os.environ.get("PANIC_LIVE_FPS", "10"))
# Per-camera person-detection rate cap. Detection is the expensive part and
# does not drive the default decision, so it runs less often than analysis.
DETECT_FPS = float(os.environ.get("PANIC_DETECT_FPS", "3"))

# A camera with no new frame for this long is shown as offline.
OFFLINE_AFTER = 3.0
# A push camera that has been disconnected for this long is removed entirely,
# so the dashboard doesn't fill up with dead tiles from phones that left.
PRUNE_AFTER = 30.0

JPEG_QUALITY = 75

NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def valid_name(name: str) -> bool:
    """Room and camera names go into URLs, DOM ids and log lines; keep them tame."""
    return bool(NAME_RE.match(name or ""))


class CameraInUse(Exception):
    """A device tried to join as a camera id that is already live."""


# --------------------------------------------------------------------------- #
# Frame sources
# --------------------------------------------------------------------------- #

class FrameSlot:
    """Single-slot mailbox: holds only the most recent frame.

    `epoch` increments on a discontinuity (a looped file jumping back to frame 0,
    a webcam that dropped out). The pipeline must not compute optical flow
    across a discontinuity, because the jump-cut would read as a huge burst of
    motion.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.epoch = 0
        self.last_seen = 0.0          # time.monotonic() of the last put()

    def put(self, frame, cut: bool = False):
        with self._lock:
            self.frame = frame
            self.seq += 1
            if cut:
                self.epoch += 1
            self.last_seen = time.monotonic()

    def get(self):
        with self._lock:
            return self.seq, self.epoch, self.frame


def parse_source(src: str):
    """'0' -> webcam index 0; anything else is a path or URL for VideoCapture."""
    src = str(src).strip()
    return int(src) if src.isdigit() else src


class CaptureSource(threading.Thread):
    """Pull source: reads a webcam / file / URL into a FrameSlot.

    Files are paced at their own frame rate and looped, so a file behaves like a
    real camera instead of being read as fast as the CPU allows.
    """

    def __init__(self, source, slot: FrameSlot):
        super().__init__(daemon=True)
        self.source = parse_source(source)
        self.is_file = isinstance(self.source, str) and os.path.exists(self.source)
        self.slot = slot
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        period = 1.0 / fps if (self.is_file and 1 <= fps <= 120) else 0.0
        cut = True
        next_t = time.monotonic()
        while self.running:
            ret, frame = cap.read()
            if not ret:
                if self.is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    # Webcam or stream dropped out: back off, then reopen.
                    time.sleep(0.5)
                    cap.release()
                    cap = cv2.VideoCapture(self.source)
                cut = True
                continue
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            self.slot.put(frame, cut=cut)
            cut = False
            if period:
                next_t += period
                delay = next_t - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.monotonic()
        cap.release()


# --------------------------------------------------------------------------- #
# One camera = one source + its own pipeline state
# --------------------------------------------------------------------------- #

class Camera:
    def __init__(self, room: str, cam: str, kind: str):
        self.room = room
        self.cam = cam
        self.id = f"{room}/{cam}"
        self.kind = kind                      # "pull" or "push"
        self.slot = FrameSlot()
        self.source_thread = None             # CaptureSource for pull cameras

        # push-camera connection bookkeeping (guarded by the registry lock)
        self.connected = kind == "pull"
        self.disconnected_at = None
        self.conn_gen = 0                     # bumps on every device (re)attach

        # Pipeline state, touched only under pipe_lock.
        self.pipe_lock = threading.Lock()
        self.classifier = PanicClassifier()
        self._prev_gray = None
        self._prev_epoch = None
        self._last_seq = 0
        self._last_processed = 0.0
        self._fps_ema = 0.0

        # Latest person boxes from the Detector, with the frame shape they were
        # found in: boxes from before a phone rotated must not be drawn after it.
        self.boxes_lock = threading.Lock()
        self._boxes = []
        self._boxes_shape = None
        self.det_seq = 0                      # slot seq of the last detected frame
        self.det_at = 0.0                     # monotonic time of the last detection

        # Published output, read by the dashboard / ingest acks.
        self.out_lock = threading.Lock()
        self.jpeg = None
        self.jpeg_seq = 0
        self.state = self._base_state()

    # ---- published state ------------------------------------------------- #

    def _base_state(self):
        return {"room": self.room, "cam": self.cam, "id": self.id, "kind": self.kind,
                "label": "NORMAL", "panic_ratio": 0.0, "score": 0.0,
                "warming_up": self.classifier.warming_up,
                "backend": self.classifier.backend,
                "degraded": self.classifier.degraded,
                "metrics": {}, "fps": 0.0}

    @property
    def online(self) -> bool:
        return (self.connected and self.slot.last_seen > 0
                and time.monotonic() - self.slot.last_seen < OFFLINE_AFTER)

    def snapshot(self) -> dict:
        with self.out_lock:
            s = dict(self.state)
        s["online"] = self.online
        return s

    def latest_jpeg(self):
        with self.out_lock:
            return self.jpeg_seq, self.jpeg

    # ---- pipeline --------------------------------------------------------- #

    def reset(self):
        """Forget this feed's baseline. Used when a phone reconnects: it has
        probably been moved, so its old idea of 'calm' no longer applies."""
        with self.pipe_lock:
            self.classifier = PanicClassifier()
            self._prev_gray = None
            self._prev_epoch = None
            self._fps_ema = 0.0
        with self.out_lock:
            self.state = self._base_state()

    def take_due(self, now: float):
        """Return (seq, epoch, frame) if a new frame is ready and this camera's
        fps cap allows processing it now; else None."""
        if now - self._last_processed < (1.0 / LIVE_FPS) * 0.95:
            return None
        seq, epoch, frame = self.slot.get()
        if frame is None or seq == self._last_seq:
            return None
        return seq, epoch, frame

    def take_for_detection(self, now: float):
        """Return (seq, frame) if this camera should get person detection now."""
        if now - self.det_at < (1.0 / DETECT_FPS) * 0.95:
            return None
        seq, _, frame = self.slot.get()
        if frame is None or seq == self.det_seq:
            return None
        return seq, frame

    def set_boxes(self, seq, shape, boxes, now: float):
        with self.boxes_lock:
            self._boxes = boxes
            self._boxes_shape = shape
            self.det_seq = seq
            self.det_at = now

    def current_boxes(self, shape):
        with self.boxes_lock:
            return list(self._boxes) if self._boxes_shape == shape else []

    def process(self, seq, epoch, frame, now: float):
        with self.pipe_lock:
            self._last_seq = seq
            if self._last_processed:
                inst = 1.0 / max(now - self._last_processed, 1e-6)
                self._fps_ema = inst if not self._fps_ema else 0.8 * self._fps_ema + 0.2 * inst
            self._last_processed = now

            H, W = frame.shape[:2]
            boxes = self.current_boxes(frame.shape)
            small = cv2.resize(frame, (0, 0), fx=FLOW_SCALE, fy=FLOW_SCALE)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            prev = self._prev_gray
            if prev is not None and prev.shape != gray.shape:
                # The frame geometry changed (a phone rotated, a stream renegotiated).
                # It's effectively a different camera now: relearn its baseline.
                self.classifier = PanicClassifier()
                prev = None
            if epoch != self._prev_epoch:
                prev = None                    # never compute flow across a cut

            flow, metrics = None, {}
            label = self.classifier.state
            if prev is not None:
                flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                metrics = compute_metrics(flow, boxes, (H, W))
                label = self.classifier.update(metrics)
            self._prev_gray = gray
            self._prev_epoch = epoch

            disp = draw_hud(frame.copy(), boxes, flow, metrics, label,
                            self.classifier.panic_ratio(), FLOW_SCALE, False,
                            panel=False, quit_hint=False)
            ok, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            state = {
                "room": self.room, "cam": self.cam, "id": self.id, "kind": self.kind,
                "label": label,
                "panic_ratio": round(self.classifier.panic_ratio(), 2),
                "score": round(self.classifier.anomaly_score, 2),
                "warming_up": self.classifier.warming_up,
                "backend": self.classifier.backend,
                "degraded": self.classifier.degraded,
                "metrics": {k: round(float(v), 3) for k, v in metrics.items()},
                "fps": round(self._fps_ema, 1),
            }
        with self.out_lock:
            if ok:
                self.jpeg = buf.tobytes()
                self.jpeg_seq += 1
            self.state = state


# --------------------------------------------------------------------------- #
# Registry: room -> cam -> Camera
# --------------------------------------------------------------------------- #

class Registry:
    def __init__(self):
        self._lock = threading.Lock()
        self._rooms = {}

    def cameras(self):
        with self._lock:
            return [c for cams in self._rooms.values() for c in cams.values()]

    def get(self, room, cam):
        with self._lock:
            return self._rooms.get(room, {}).get(cam)

    def add_pull(self, room: str, cam: str, source: str) -> Camera:
        if not (valid_name(room) and valid_name(cam)):
            raise ValueError(f"invalid room/camera name {room!r}/{cam!r} "
                             "(use letters, digits, '-' or '_', max 32)")
        with self._lock:
            if cam in self._rooms.get(room, {}):
                raise CameraInUse(f"{room}/{cam} is already configured")
            c = Camera(room, cam, "pull")
            self._rooms.setdefault(room, {})[cam] = c
        c.source_thread = CaptureSource(source, c.slot)
        c.source_thread.start()
        return c

    def attach_push(self, room: str, cam: str):
        """A device joined as room/cam. Returns (camera, generation).

        Creates the camera, or takes over an existing push camera that is either
        disconnected or has gone silent. Taking over a silent one matters: when a
        phone drops off Wi-Fi the server may not notice its dead socket for tens
        of seconds, and the phone's reconnect must not be locked out meanwhile.
        A live camera that is still sending frames is refused (CameraInUse).

        Any re-attach starts a fresh baseline: the device has probably moved.
        """
        if not (valid_name(room) and valid_name(cam)):
            raise ValueError("invalid room/camera name")
        with self._lock:
            c = self._rooms.get(room, {}).get(cam)
            if c is not None and (c.kind == "pull" or (c.connected and c.online)):
                raise CameraInUse(f"{room}/{cam} is already live")
            fresh = c is None
            if fresh:
                c = Camera(room, cam, "push")
                self._rooms.setdefault(room, {})[cam] = c
            c.conn_gen += 1
            gen = c.conn_gen
            c.connected = True
            c.disconnected_at = None
        if not fresh:
            c.reset()
        return c, gen

    def detach_push(self, c: Camera, gen: int):
        """Mark disconnected, unless a newer connection has already taken over."""
        with self._lock:
            if c.conn_gen != gen:
                return
            c.connected = False
            c.disconnected_at = time.monotonic()

    def prune(self, now: float = None):
        now = time.monotonic() if now is None else now
        with self._lock:
            for room in list(self._rooms):
                cams = self._rooms[room]
                for cam in list(cams):
                    c = cams[cam]
                    if (c.kind == "push" and not c.connected and c.disconnected_at
                            and now - c.disconnected_at > PRUNE_AFTER):
                        del cams[cam]
                if not cams:
                    del self._rooms[room]

    def room_alarm(self, room: str):
        """Camera ids in `room` that are currently in PANIC (empty = all clear)."""
        with self._lock:
            cams = list(self._rooms.get(room, {}).values())
        return [c.cam for c in cams if c.online and c.snapshot()["label"] == "PANIC"]

    def snapshot(self):
        with self._lock:
            rooms = {r: list(cams.values()) for r, cams in self._rooms.items()}
        out = []
        for room in sorted(rooms):
            cams = [c.snapshot() for c in sorted(rooms[room], key=lambda c: c.cam)]
            panic = [s["cam"] for s in cams if s["online"] and s["label"] == "PANIC"]
            out.append({"room": room, "alarm": bool(panic), "panic_cams": panic,
                        "cameras": cams})
        return out

    def stop(self):
        for c in self.cameras():
            if c.source_thread is not None:
                c.source_thread.running = False


def parse_sources(raw: str):
    """PANIC_SOURCES -> [(room, cam, source)].

    Entries are comma-separated `room=source` or `room/cam=source`. A bare
    `room=source` becomes camera "main" in that room.
    """
    out = []
    for part in (raw or "").split(","):
        if "=" not in part:
            continue
        name, src = part.split("=", 1)
        name, src = name.strip(), src.strip()
        if not name or not src:
            continue
        room, _, cam = name.partition("/")
        out.append((room.strip(), (cam.strip() or "main"), src))
    return out


# --------------------------------------------------------------------------- #
# Worker threads
# --------------------------------------------------------------------------- #

class _Loop(threading.Thread):
    """Calls tick() until stopped; sleeps briefly when there was nothing to do."""

    def __init__(self, name):
        super().__init__(daemon=True, name=name)
        self.running = True

    def tick(self) -> int:
        raise NotImplementedError

    def run(self):
        while self.running:
            try:
                if not self.tick():
                    time.sleep(0.005)
            except Exception:
                # One bad frame must not take every camera down with it.
                traceback.print_exc()
                time.sleep(0.5)


class Scheduler(_Loop):
    """The analysis path: flow -> metrics -> classifier -> HUD, per camera."""

    def __init__(self, registry: Registry, workers: int = None):
        super().__init__("live-analysis")
        self.registry = registry
        self.pool = ThreadPoolExecutor(max_workers=workers or min(4, os.cpu_count() or 2),
                                       thread_name_prefix="live-analysis")
        self._last_prune = 0.0

    def tick(self) -> int:
        """Process every due camera once. Returns how many were processed."""
        now = time.monotonic()
        if now - self._last_prune > 1.0:
            self.registry.prune(now)
            self._last_prune = now

        due = []
        for c in self.registry.cameras():
            item = c.take_due(now)
            if item is not None:
                due.append((c, *item))
        futures = [self.pool.submit(c.process, seq, epoch, frame, now)
                   for (c, seq, epoch, frame) in due]
        for f in futures:
            f.result()
        return len(due)

    def run(self):
        super().run()
        self.pool.shutdown(wait=False)


class Detector(_Loop):
    """Person detection: one batched YOLO call over every camera that is due."""

    def __init__(self, registry: Registry, model):
        super().__init__("live-detector")
        self.registry = registry
        self.model = model

    def tick(self) -> int:
        now = time.monotonic()
        todo = []
        for c in self.registry.cameras():
            item = c.take_for_detection(now)
            if item is not None:
                todo.append((c, *item))
        if not todo:
            return 0
        results = self.model([f for (_, _, f) in todo], classes=[0], verbose=False)
        done = time.monotonic()
        for (c, seq, frame), res in zip(todo, results):
            c.set_boxes(seq, frame.shape, extract_boxes([res], CONF), done)
        return len(todo)
