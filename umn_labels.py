"""
Frame-level ground truth for the UMN Unusual Crowd Activity dataset.

`Crowd-Activity-All.avi` concatenates 11 escape sequences across 3 scenes, and
each sequence runs NORMAL for a while before the crowd scatters. So the file is
NOT "a panic video" — roughly two thirds of it is calm. Training on it with
train_classifier.py's --extra-panic (which labels a whole file class 1) would
teach the model that people milling about on a lawn is panic.

Luckily UMN burns its own label into the frame: a red "Abnormal Crowd Activity"
banner appears in the upper-left for exactly the abnormal stretches. Detecting
that banner recovers per-frame ground truth directly from the pixels, with no
hand-typed frame ranges to get wrong.

Detection is deliberately narrow: count strongly-red pixels in the banner strip.
The scenes are a green lawn, a grey indoor corridor, and a grey plaza; none of
them put saturated red in that corner, and scene 2 is effectively greyscale.

Scene boundaries are the published frame counts (1453 / 4144 / 2142), used to
hold out a whole scene for testing — splitting on random frames would leak,
since neighbouring frames are nearly identical.
"""
import cv2
import numpy as np

# Banner strip in the 320x240 frame. The banner is a fixed white box spanning
# roughly x 0-170, y 5-28, identical in all three scenes.
BANNER = (slice(0, 32), slice(0, 180))

# A banner pixel is strongly red: high R, and clearly above both G and B.
RED_MIN = 120
DOMINANCE = 60

# Measured separation is wide: the banner yields 555-563 red pixels in this
# strip, while banner-free frames yield 0-91 (people in red or pink clothing
# walking along the top edge). 300 sits in the middle of that gap. An earlier
# threshold of 40 caught the clothing and invented 33 "escape events" out of 11.
MIN_RED_PIXELS = 300

SCENE_BOUNDS = [(0, 1452), (1453, 5596), (5597, 7738)]   # inclusive, 0-indexed


def frame_is_abnormal(frame) -> bool:
    """True when the red 'Abnormal Crowd Activity' banner is present."""
    strip = frame[BANNER[0], BANNER[1]]
    if strip.ndim != 3:
        return False
    b, g, r = strip[..., 0].astype(np.int16), strip[..., 1].astype(np.int16), strip[..., 2].astype(np.int16)
    red = (r > RED_MIN) & (r - g > DOMINANCE) & (r - b > DOMINANCE)
    return int(red.sum()) >= MIN_RED_PIXELS


def scene_of(frame_idx: int) -> int:
    for i, (a, b) in enumerate(SCENE_BOUNDS):
        if a <= frame_idx <= b:
            return i
    return len(SCENE_BOUNDS) - 1


def label_video(path):
    """Return (labels, scenes) as lists, one entry per frame."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")
    labels, scenes, i = [], [], 0
    while True:
        ret, f = cap.read()
        if not ret:
            break
        labels.append(1 if frame_is_abnormal(f) else 0)
        scenes.append(scene_of(i))
        i += 1
    cap.release()
    return labels, scenes


def segments(labels):
    """Contiguous abnormal runs as (start, end) inclusive frame indices."""
    out, start = [], None
    for i, v in enumerate(labels):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(labels) - 1))
    return out


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "datasets/umn/Crowd-Activity-All.avi"
    labels, scenes = label_video(path)
    segs = segments(labels)
    n = len(labels)
    print(f"frames: {n}   abnormal: {sum(labels)} ({sum(labels)/n*100:.1f}%)")
    print(f"escape events found: {len(segs)}  (UMN ships 11)")
    for i, (a, b) in enumerate(segs, 1):
        print(f"  #{i:2d}  frames {a:5d}-{b:5d}  ({b-a+1:4d} frames)  scene {scene_of(a)+1}")
    for s in range(3):
        idx = [i for i, sc in enumerate(scenes) if sc == s]
        pos = sum(labels[i] for i in idx)
        print(f"  scene {s+1}: {len(idx)} frames, {pos} abnormal ({pos/len(idx)*100:.1f}%)")
