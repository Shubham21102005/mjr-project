"""
Offline analysis of an uploaded clip.

Runs the same four-stage pipeline the CLI and the live dashboard run — YOLO
person detection, Farneback dense flow, compute_metrics, PanicClassifier — over
every frame of a video file, and produces:

  * an annotated H.264 MP4 the browser can play back
  * the per-frame label/metric series, for the timeline strip
  * the contiguous PANIC stretches as (start, end) timestamps, so the UI can
    list them and seek straight to them

This is the demo path: staging real panic in front of a live camera is not
something you can (or should) do, so an uploaded clip is how the detector gets
shown working.

Nothing here re-implements pipeline logic — extract_boxes and draw_hud come from
detect.py and the metrics from metrics.py, exactly as server.py and evaluate.py
use them.
"""
import cv2
import numpy as np

from classifier import PanicClassifier
from detect import extract_boxes, draw_hud
from metrics import compute_metrics
from video_writer import BrowserVideoWriter

FLOW_SCALE = 0.5
CONF = 0.4
OUT_WIDTH = 960          # annotated output width; small sources get upscaled so
                         # the HUD text stays legible
MAX_SERIES_POINTS = 900  # cap what we ship to the browser for the timeline


def _segments(labels, fps):
    """Contiguous runs of PANIC -> [{start,end,start_frame,end_frame,duration}]."""
    out = []
    start = None
    for i, lab in enumerate(labels):
        if lab == "PANIC" and start is None:
            start = i
        elif lab != "PANIC" and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(labels) - 1))

    return [
        {
            "start_frame": a,
            "end_frame": b,
            "start": round(a / fps, 2),
            "end": round((b + 1) / fps, 2),
            "duration": round((b - a + 1) / fps, 2),
        }
        for a, b in out
    ]


def _downsample(series, limit=MAX_SERIES_POINTS):
    """Evenly thin a per-frame series so long clips don't ship 50k points."""
    n = len(series)
    if n <= limit:
        return series
    idx = np.linspace(0, n - 1, limit).astype(int)
    return [series[i] for i in idx]


def analyze_video(src_path, out_path, model, progress_cb=None,
                  flow_scale=FLOW_SCALE, conf=CONF, window=10):
    """Run the pipeline over src_path, write annotated video to out_path.

    progress_cb(frames_done, total_frames) is called as work proceeds.
    Returns a result dict. Raises on unreadable input.
    """
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError("could not open the uploaded video (unsupported or corrupt file)")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:      # 0, or NaN on some containers
        fps = 20.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    clf = PanicClassifier(window_size=window)
    writer = None
    prev = None
    flow = None
    metrics = {}
    label = "NORMAL"

    labels = []
    series = []
    n = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            n += 1

            if len(frame.shape) == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            H, W = frame.shape[:2]

            boxes = extract_boxes(model(frame, classes=[0], verbose=False), conf)

            small = cv2.resize(frame, (0, 0), fx=flow_scale, fy=flow_scale)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                metrics = compute_metrics(flow, boxes, (H, W))
                label = clf.update(metrics)
            prev = gray

            labels.append(label)
            series.append({
                "t": round((n - 1) / fps, 3),
                "panic": label == "PANIC",
                "ratio": round(clf.panic_ratio(), 3),
                # Robust z of this frame against the clip's own calm baseline.
                # This, not avg_speed, is what the scene-baseline backend decides
                # on, so it is what the timeline should plot.
                "score": round(clf.anomaly_score, 2),
                "warming_up": clf.warming_up,
                "avg_speed": round(float(metrics.get("avg_speed", 0.0)), 3),
                "flow_entropy": round(float(metrics.get("flow_entropy", 0.0)), 3),
                "direction_variance": round(float(metrics.get("direction_variance", 0.0)), 3),
                "crowd_density": round(float(metrics.get("crowd_density", 0.0)), 4),
                "person_count": int(metrics.get("person_count", 0)),
            })

            # Scale up first, then draw, so the HUD is crisp on small sources.
            # Boxes are in full-res coords, so they scale with the frame.
            s = OUT_WIDTH / W
            disp = cv2.resize(frame, (OUT_WIDTH, max(2, int(H * s))),
                              interpolation=cv2.INTER_LINEAR)
            sboxes = [[x1 * s, y1 * s, x2 * s, y2 * s] for (x1, y1, x2, y2) in boxes]
            disp = draw_hud(disp, sboxes, flow, metrics, label,
                            clf.panic_ratio(), flow_scale, False, quit_hint=False)

            if writer is None:
                writer = BrowserVideoWriter(out_path, disp.shape[1], disp.shape[0], fps)
            writer.write(disp)

            if progress_cb and (n % 5 == 0 or n == total):
                progress_cb(n, total)
    finally:
        cap.release()

    if writer is None:
        raise RuntimeError("the uploaded video contained no readable frames")

    warning = writer.close()

    if progress_cb:
        progress_cb(n, total or n)

    segs = _segments(labels, fps)
    panic_frames = sum(l == "PANIC" for l in labels)
    peak = max((p["avg_speed"] for p in series), default=0.0)
    peak_score = max((p["score"] for p in series), default=0.0)
    warmup_frames = sum(1 for p in series if p["warming_up"])

    return {
        "fps": round(fps, 3),
        "frames": n,
        "duration": round(n / fps, 2),
        "panic_frames": panic_frames,
        "panic_fraction": round(panic_frames / n, 3) if n else 0.0,
        "segments": segs,
        "series": _downsample(series),
        "backend": clf.backend,
        "degraded": clf.degraded,
        "load_error": clf.load_error,
        "peak_avg_speed": peak,
        "peak_score": peak_score,
        "score_threshold": clf.baseline.k if clf.baseline is not None else None,
        "warmup_frames": warmup_frames,
        "warning": warning,
        "verdict": "PANIC DETECTED" if segs else "NO PANIC DETECTED",
    }
