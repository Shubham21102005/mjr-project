"""
Evaluate panic detection against UCSD Ped1 ground truth.

Usage:
    python evaluate.py --dataset "C:/Users/shubh/Downloads/UCSD_Anomaly_Dataset.tar/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test"

Prints per-clip and overall: precision, recall, F1, accuracy.
A frame is ground-truth anomalous if any pixel in its _gt mask is non-zero.
"""

import argparse
import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

from metrics import compute_metrics
from classifier import PanicClassifier


def load_gt_flags(gt_dir: str) -> dict:
    """Return {frame_number: bool} — True if frame has any anomaly pixels."""
    flags = {}
    for f in glob.glob(os.path.join(gt_dir, "*.bmp")):
        n = int(os.path.splitext(os.path.basename(f))[0])
        mask = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        flags[n] = mask is not None and mask.max() > 0
    return flags


def run_clip(clip_dir: str, model, flow_scale=0.5, conf=0.4, window=10):
    """Run the full pipeline on a clip directory of .tif frames.
    Returns list of (frame_number, predicted_panic: bool)."""
    frames = sorted(glob.glob(os.path.join(clip_dir, "*.tif")))
    classifier = PanicClassifier(window_size=window)
    prev_gray = None
    results_out = []

    for f in frames:
        frame_num = int(os.path.splitext(os.path.basename(f))[0])
        frame = cv2.imread(f)
        if frame is None:
            continue
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        H, W = frame.shape[:2]

        yolo_results = model(frame, classes=[0], verbose=False)
        boxes = []
        if yolo_results[0].boxes is not None and len(yolo_results[0].boxes) > 0:
            xyxy = yolo_results[0].boxes.xyxy.cpu().numpy()
            confs = yolo_results[0].boxes.conf.cpu().numpy()
            boxes = [xyxy[i].tolist() for i in range(len(xyxy)) if confs[i] >= conf]

        small = cv2.resize(frame, (0, 0), fx=flow_scale, fy=flow_scale)
        curr_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            m = compute_metrics(flow, boxes, (H, W))
            label = classifier.update(m)
            results_out.append((frame_num, label == "PANIC"))

        prev_gray = curr_gray

    return results_out


def evaluate_clip(predicted: list, gt_flags: dict):
    tp = fp = tn = fn = 0
    for frame_num, pred_panic in predicted:
        gt_panic = gt_flags.get(frame_num, False)
        if pred_panic and gt_panic:
            tp += 1
        elif pred_panic and not gt_panic:
            fp += 1
        elif not pred_panic and gt_panic:
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def print_metrics(label, tp, fp, tn, fn):
    total = tp + fp + tn + fn
    accuracy  = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    print(f"  {label:<12}  Acc={accuracy:.2f}  Prec={precision:.2f}  Rec={recall:.2f}  F1={f1:.2f}"
          f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    return tp, fp, tn, fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
        default=r"C:\Users\shubh\Downloads\UCSD_Anomaly_Dataset.tar\UCSD_Anomaly_Dataset\UCSD_Anomaly_Dataset.v1p2\UCSDped1\Test")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--flow-scale", type=float, default=0.5)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()

    model = YOLO(args.model)

    clips = sorted(
        d for d in os.listdir(args.dataset)
        if not d.endswith("_gt") and os.path.isdir(os.path.join(args.dataset, d))
    )
    gt_clips = [c for c in clips if os.path.isdir(os.path.join(args.dataset, c + "_gt"))]

    print(f"\nEvaluating {len(gt_clips)} clips with ground truth...\n")

    total_tp = total_fp = total_tn = total_fn = 0

    for clip in gt_clips:
        clip_dir = os.path.join(args.dataset, clip)
        gt_dir   = os.path.join(args.dataset, clip + "_gt")
        gt_flags = load_gt_flags(gt_dir)
        predicted = run_clip(clip_dir, model, args.flow_scale, args.conf, args.window)
        tp, fp, tn, fn = evaluate_clip(predicted, gt_flags)
        print_metrics(clip, tp, fp, tn, fn)
        total_tp += tp; total_fp += fp; total_tn += tn; total_fn += fn

    print()
    print_metrics("OVERALL", total_tp, total_fp, total_tn, total_fn)


if __name__ == "__main__":
    main()
