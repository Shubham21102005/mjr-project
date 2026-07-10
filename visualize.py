"""
Render an ANNOTATED video of the detector running on a clip, so you can visually
judge whether it behaves sensibly (instead of only reading F1 numbers).

Works on:
  - a UCSD clip directory of frames:  --clip datasets/.../UCSDped1/Test/Test018
    (if a matching <clip>_gt folder exists, the ground-truth anomaly is overlaid)
  - any normal video file:            --clip panic_test.avi

For each frame it overlays:
  - a red motion heatmap of the optical flow  (what the detector 'sees' moving)
  - person boxes from YOLO
  - the live metrics + the NORMAL/PANIC label (green/red)
  - the ground-truth anomaly region in yellow (UCSD only), so you can compare
    what the detector flags vs. what is actually anomalous

Usage:
    python visualize.py --clip datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test018
    python visualize.py --clip panic_test.avi --out viz/panic.mp4
"""
import argparse
import glob
import os
import cv2
import numpy as np
from ultralytics import YOLO

from metrics import compute_metrics
from classifier import PanicClassifier


def frame_source(clip):
    """Yield (frame_number, bgr_frame). Directory of images or a video file."""
    if os.path.isdir(clip):
        files = sorted(glob.glob(os.path.join(clip, "*.tif")) +
                       glob.glob(os.path.join(clip, "*.jpg")) +
                       glob.glob(os.path.join(clip, "*.png")))
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            num = int("".join(ch for ch in stem if ch.isdigit()) or 0)
            img = cv2.imread(f)
            if img is not None:
                yield num, img
    else:
        cap = cv2.VideoCapture(clip)
        n = 0
        while True:
            ret, img = cap.read()
            if not ret:
                break
            n += 1
            yield n, img
        cap.release()


def gt_boxes_for(gt_dir, frame_num, shape):
    """Return list of (x1,y1,x2,y2) anomaly boxes from the frame's GT mask."""
    if not gt_dir:
        return []
    path = os.path.join(gt_dir, f"{frame_num:03d}.bmp")
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.max() == 0:
        return []
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]))
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= 12:                       # ignore specks
            out.append((x, y, x + w, y + h))
    return out


def render(frame, s, boxes, gt, flow, flow_scale, metrics, label, ratio, title):
    H, W = frame.shape[:2]
    disp = cv2.resize(frame, (int(W * s), int(H * s)), interpolation=cv2.INTER_LINEAR)
    Hd, Wd = disp.shape[:2]

    # Motion heatmap: upscale flow magnitude, colourise, blend where there is motion.
    if flow is not None:
        mag = cv2.magnitude(flow[..., 0], flow[..., 1])
        mag = cv2.resize(mag, (Wd, Hd))
        norm = np.clip(mag / 6.0, 0, 1)                     # 6 px ~ strong motion
        heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        a = (norm * 0.55)[..., None]
        disp = (disp * (1 - a) + heat * a).astype(np.uint8)

    is_panic = label == "PANIC"
    col = (0, 0, 255) if is_panic else (0, 200, 0)

    for (x1, y1, x2, y2) in boxes:                          # YOLO persons
        cv2.rectangle(disp, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), col, 2)

    for (x1, y1, x2, y2) in gt:                             # ground-truth anomaly
        cv2.rectangle(disp, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), (0, 220, 255), 2)
        cv2.putText(disp, "GT anomaly", (int(x1 * s), int(y1 * s) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

    # HUD panel
    ov = disp.copy()
    cv2.rectangle(ov, (8, 8), (300, 168), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, disp, 0.45, 0, disp)
    cv2.putText(disp, label, (16, 48), cv2.FONT_HERSHEY_DUPLEX, 1.3, col, 2, cv2.LINE_AA)
    lines = [
        f"Avg Speed:    {metrics.get('avg_speed', 0):.2f} %diag/f  (thr 1.8)",
        f"Flow Entropy: {metrics.get('flow_entropy', 0):.2f} / 3.58 (thr 3.2)",
        f"Dir Variance: {metrics.get('direction_variance', 0):.2f}      (thr 0.75)",
        f"Occupancy:    {metrics.get('crowd_density', 0):.3f}      (thr 0.18)",
        f"Panic Ratio:  {ratio:.2f}    Persons: {metrics.get('person_count', 0)}",
    ]
    for i, ln in enumerate(lines):
        cv2.putText(disp, ln, (16, 76 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (215, 215, 215), 1, cv2.LINE_AA)

    cv2.putText(disp, title, (16, Hd - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return disp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help="clip directory or video file")
    ap.add_argument("--out", default=None, help="output .mp4 (default viz/<name>.mp4)")
    ap.add_argument("--gt", default=None, help="ground-truth dir (default: <clip>_gt if it exists)")
    ap.add_argument("--width", type=int, default=760)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--flow-scale", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--model", default="yolov8n.pt")
    args = ap.parse_args()

    name = os.path.basename(os.path.normpath(args.clip)).split(".")[0]
    out = args.out or os.path.join("viz", f"{name}.mp4")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    gt_dir = args.gt
    if gt_dir is None and os.path.isdir(args.clip) and os.path.isdir(args.clip + "_gt"):
        gt_dir = args.clip + "_gt"

    model = YOLO(args.model)
    clf = PanicClassifier()
    writer = None
    prev = None
    metrics, label, flow = {}, "NORMAL", None
    n_panic = n_gt = n = 0

    for fnum, frame in frame_source(args.clip):
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        H, W = frame.shape[:2]
        s = args.width / W

        res = model(frame, classes=[0], verbose=False)
        boxes = []
        if res[0].boxes is not None and len(res[0].boxes):
            xy = res[0].boxes.xyxy.cpu().numpy(); cf = res[0].boxes.conf.cpu().numpy()
            boxes = [xy[i].tolist() for i in range(len(xy)) if cf[i] >= args.conf]

        small = cv2.resize(frame, (0, 0), fx=args.flow_scale, fy=args.flow_scale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            metrics = compute_metrics(flow, boxes, (H, W))
            label = clf.update(metrics)
        prev = gray

        gt = gt_boxes_for(gt_dir, fnum, (H, W))
        n += 1
        n_panic += label == "PANIC"
        n_gt += len(gt) > 0
        title = f"{name}  frame {fnum}   GT: {'ANOMALY' if gt else 'normal'}"
        disp = render(frame, s, boxes, gt, flow, args.flow_scale, metrics, label, clf.panic_ratio(), title)

        if writer is None:
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (disp.shape[1], disp.shape[0]))
        writer.write(disp)

    if writer:
        writer.release()
    print(f"wrote {out}")
    print(f"  frames={n}  detector-PANIC={n_panic}  frames-with-GT-anomaly={n_gt}")


if __name__ == "__main__":
    main()
