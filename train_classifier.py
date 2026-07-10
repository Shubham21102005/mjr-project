"""
Train the panic classifier on crowd-motion features.

Instead of hand-picking thresholds, we learn a decision boundary over three
resolution-independent motion features:

    avg_speed, direction_variance, flow_entropy

Training data:
  * SYNTHETIC  — synth.gen_clip() sweeps speed x coherence across the whole
    spectrum. Slow + coherent = NORMAL (label 0); fast + incoherent = PANIC
    (label 1). Every frame is labelled by construction.
  * REAL-NORMAL — frames sampled from a real calm crowd video (demo_crowd.mp4),
    all labelled NORMAL, so the model learns that real walking crowds are safe.
  * (optional) --extra-normal / --extra-panic video files, e.g. UMN clips, to
    fold in genuine footage when you have it.

Output: panic_model.pkl  (a scikit-learn Pipeline: StandardScaler + LogisticRegression)

Usage:
    python train_classifier.py
    python train_classifier.py --extra-panic path/to/umn_panic.avi
"""
import argparse
import pickle

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from metrics import compute_metrics
from synth import gen_clip

FEATURES = ["avg_speed", "direction_variance", "flow_entropy"]
FLOW_SCALE = 0.5


def feats_from_flow(prev_gray, gray, shape):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    m = compute_metrics(flow, [], shape)
    return [m[k] for k in FEATURES]


def features_from_frames(frames, shape):
    """Turn an iterable of BGR frames into a list of per-frame feature vectors."""
    rows, prev = [], None
    for frame in frames:
        small = cv2.resize(frame, (0, 0), fx=FLOW_SCALE, fy=FLOW_SCALE)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            rows.append(feats_from_flow(prev, gray, shape))
        prev = gray
    return rows


def video_frames(path, max_frames=600, stride=1):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  ! could not open {path}, skipping")
        return
    i = kept = 0
    while kept < max_frames:
        ret, f = cap.read()
        if not ret:
            break
        if i % stride == 0:
            if len(f.shape) == 2:
                f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            yield f
            kept += 1
        i += 1
    cap.release()


def build_dataset(args):
    X, y = [], []

    # --- Synthetic sweep: the backbone of the training set ------------------
    # NORMAL: slow, coherent.  PANIC: fast, incoherent.  Plus a graded middle so
    # the boundary is learned, not memorised as a single point.
    normal_cfgs = [(2.0, 0.95), (2.5, 0.9), (3.0, 0.85), (3.5, 0.8), (1.5, 0.7)]
    panic_cfgs  = [(20, 0.1), (24, 0.05), (18, 0.2), (22, 0.15), (26, 0.0)]

    print("Generating synthetic NORMAL clips...")
    for k, (spd, coh) in enumerate(normal_cfgs):
        rows = features_from_frames(gen_clip(spd, coh, 60, seed=100 + k), (480, 480))
        X += rows; y += [0] * len(rows)

    print("Generating synthetic PANIC clips...")
    for k, (spd, coh) in enumerate(panic_cfgs):
        rows = features_from_frames(gen_clip(spd, coh, 60, seed=200 + k), (480, 480))
        X += rows; y += [1] * len(rows)

    # --- Real calm crowd: labelled NORMAL ----------------------------------
    print(f"Sampling real-normal frames from {args.real_normal}...")
    frames = list(video_frames(args.real_normal, max_frames=400, stride=3))
    if frames:
        rows = features_from_frames(frames, frames[0].shape[:2])
        X += rows; y += [0] * len(rows)

    # --- Optional real footage ---------------------------------------------
    for path, label in [(args.extra_normal, 0), (args.extra_panic, 1)]:
        if path:
            print(f"Sampling {'PANIC' if label else 'NORMAL'} frames from {path}...")
            frames = list(video_frames(path, max_frames=600))
            if frames:
                rows = features_from_frames(frames, frames[0].shape[:2])
                X += rows; y += [label] * len(rows)

    return np.array(X, dtype=float), np.array(y, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-normal", default="demo_crowd.mp4",
                    help="A real calm-crowd video, labelled NORMAL (default demo_crowd.mp4)")
    ap.add_argument("--extra-normal", default=None, help="Optional extra NORMAL video")
    ap.add_argument("--extra-panic", default=None, help="Optional real PANIC video (e.g. UMN)")
    ap.add_argument("--out", default="panic_model.pkl")
    args = ap.parse_args()

    X, y = build_dataset(args)
    print(f"\nDataset: {len(y)} frames  ({int((y == 0).sum())} normal, {int((y == 1).sum())} panic)")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(class_weight="balanced", max_iter=1000))
    clf.fit(Xtr, ytr)

    print("\nHeld-out test performance:")
    print(classification_report(yte, clf.predict(Xte), target_names=["NORMAL", "PANIC"]))

    coef = clf.named_steps["logisticregression"].coef_[0]
    print("Learned feature weights (higher => pushes toward PANIC):")
    for name, w in zip(FEATURES, coef):
        print(f"  {name:20s} {w:+.3f}")

    with open(args.out, "wb") as f:
        pickle.dump({"model": clf, "features": FEATURES}, f)
    print(f"\nSaved trained model -> {args.out}")


if __name__ == "__main__":
    main()
