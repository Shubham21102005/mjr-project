import argparse
import cv2
import numpy as np
from ultralytics import YOLO

from metrics import compute_metrics
from classifier import PanicClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Crowd Panic Detection")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source: '0' for webcam, or path to video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLOv8 model weights (default: yolov8n)")
    parser.add_argument("--flow-scale", type=float, default=0.5,
                        help="Scale factor for optical flow frame (default: 0.5)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="YOLO confidence threshold (default: 0.4)")
    parser.add_argument("--window", type=int, default=10,
                        help="Sliding window size in frames (default: 10)")
    parser.add_argument("--debug-flow", action="store_true",
                        help="Overlay optical flow arrows on frame")
    parser.add_argument("--display-width", type=int, default=960,
                        help="Display window width in pixels (default: 960)")
    return parser.parse_args()


def extract_boxes(results, conf_thresh):
    boxes = []
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        xyxy = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        for i, box in enumerate(xyxy):
            if confs[i] >= conf_thresh:
                boxes.append(box.tolist())
    return boxes


def draw_hud(frame, boxes, flow, metrics, label, panic_ratio, flow_scale, debug_flow):
    H, W = frame.shape[:2]
    is_panic = label == "PANIC"
    status_color = (0, 0, 255) if is_panic else (0, 200, 0)
    box_color = (0, 0, 255) if is_panic else (0, 200, 0)

    # Bounding boxes
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
        cv2.putText(frame, "person", (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

    # Optional flow arrows
    if debug_flow and flow is not None:
        step = 20
        for y in range(0, H, step):
            for x in range(0, W, step):
                fy = int(y * flow_scale)
                fx = int(x * flow_scale)
                if fy < flow.shape[0] and fx < flow.shape[1]:
                    dx = flow[fy, fx, 0] / flow_scale
                    dy = flow[fy, fx, 1] / flow_scale
                    end = (int(x + dx * 2), int(y + dy * 2))
                    cv2.arrowedLine(frame, (x, y), end, (255, 140, 0), 1, tipLength=0.3)

    # Semi-transparent metrics panel
    panel_h, panel_w = 175, 310
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Status label
    cv2.putText(frame, label, (18, 52),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, status_color, 2, cv2.LINE_AA)

    # Metrics text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (210, 210, 210)
    y0, dy = 82, 19
    lines = [
        f"Avg Speed:    {metrics.get('avg_speed', 0):.2f} px/f",
        f"Dir Variance: {metrics.get('direction_variance', 0):.2f} rad",
        f"Density:      {metrics.get('crowd_density', 0):.6f}",
        f"Flow Entropy: {metrics.get('flow_entropy', 0):.2f} / 3.58",
        f"Panic Ratio:  {panic_ratio:.2f}",
        f"Persons:      {metrics.get('person_count', 0)}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (18, y0 + i * dy),
                    font, 0.48, text_color, 1, cv2.LINE_AA)

    # Quit hint
    cv2.putText(frame, "Q to quit", (W - 105, H - 12),
                font, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

    return frame


def main():
    args = parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    model = YOLO(args.model)
    classifier = PanicClassifier(window_size=args.window)

    prev_gray_small = None
    latest_metrics = {}
    latest_label = "NORMAL"
    latest_flow = None

    print(f"Running crowd panic detection. Source: {args.source}. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        H, W = frame.shape[:2]

        # YOLOv8 person detection (class 0 = person)
        results = model(frame, classes=[0], verbose=False)
        boxes = extract_boxes(results, args.conf)

        # Optical flow on downscaled grayscale
        small = cv2.resize(frame, (0, 0), fx=args.flow_scale, fy=args.flow_scale)
        curr_gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if prev_gray_small is not None:
            latest_flow = cv2.calcOpticalFlowFarneback(
                prev_gray_small, curr_gray_small, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            latest_metrics = compute_metrics(latest_flow, boxes, (H, W))
            latest_label = classifier.update(latest_metrics)

        prev_gray_small = curr_gray_small

        display = draw_hud(
            frame.copy(), boxes, latest_flow, latest_metrics,
            latest_label, classifier.panic_ratio(),
            args.flow_scale, args.debug_flow
        )

        # Upscale for display if frame is smaller than target width
        dh, dw = display.shape[:2]
        if dw != args.display_width:
            scale = args.display_width / dw
            display = cv2.resize(display, (args.display_width, int(dh * scale)),
                                 interpolation=cv2.INTER_LINEAR)

        cv2.imshow("Crowd Panic Detection", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
