import cv2
import numpy as np
from scipy.stats import entropy as scipy_entropy


def compute_metrics(flow: np.ndarray, boxes: list, frame_shape: tuple) -> dict:
    """
    Compute crowd behaviour metrics from dense optical flow and person detections.

    Parameters
    ----------
    flow        : np.ndarray shape (H_small, W_small, 2), dtype float32
                  Output of cv2.calcOpticalFlowFarneback.
                  flow[..., 0] = dx, flow[..., 1] = dy
    boxes       : list of [x1, y1, x2, y2] in full-resolution pixel coords
    frame_shape : (H, W) of the full-resolution frame

    Returns
    -------
    dict with keys: avg_speed, direction_variance, crowd_density, flow_entropy,
    person_count. All metrics are resolution-independent so one threshold set
    works across webcam / phone / file sources of any size.
    """
    magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])  # (H_small, W_small)
    angle = cv2.phase(flow[..., 0], flow[..., 1])           # radians, 0 to 2π

    # Only consider pixels with meaningful motion (ignore static background)
    mask = magnitude > 0.5

    if mask.sum() > 0:
        # Average speed of *moving* pixels only, expressed as a percentage of the
        # frame diagonal per frame. Dividing by the flow-frame diagonal makes this
        # independent of both the source resolution and flow_scale (magnitude and
        # diagonal scale together), so the threshold transfers across sources.
        Hs, Ws = magnitude.shape
        diag_small = float(np.hypot(Hs, Ws))
        avg_speed = float(np.mean(magnitude[mask]) / diag_small * 100.0)
        # Circular variance of motion angles: 1 - R, where R is the length of the
        # mean resultant vector. Angles are circular (wrap at 0/2pi), so a plain
        # np.std is wrong — it reports motion clustered near 0 as maximally varied.
        # This is bounded [0, 1]: 0 = perfectly coherent motion (a crowd all moving
        # one way), ~1 = directions spread uniformly (chaotic panic).
        a = angle[mask]
        R = float(np.hypot(np.mean(np.cos(a)), np.mean(np.sin(a))))
        direction_variance = 1.0 - R
        angle_deg = np.degrees(angle[mask]) % 360
        hist, _ = np.histogram(angle_deg, bins=36, range=(0, 360))
        hist = hist.astype(float) + 1e-9  # Laplace smooth
        flow_entropy = float(scipy_entropy(hist))
    else:
        avg_speed = 0.0
        direction_variance = 0.0
        flow_entropy = 0.0

    # Crowd occupancy: fraction of the frame area covered by person boxes. This is
    # resolution-independent (unlike people-per-pixel, which shrinks as resolution
    # grows) and is a better proxy for how packed the scene is. Clamped to 1.0
    # since overlapping boxes can otherwise sum past the full frame area.
    H, W = frame_shape
    frame_area = float(H * W)
    box_area = sum(max(0.0, x2 - x1) * max(0.0, y2 - y1) for (x1, y1, x2, y2) in boxes)
    crowd_density = min(1.0, box_area / frame_area) if frame_area > 0 else 0.0

    return {
        "avg_speed": avg_speed,
        "direction_variance": direction_variance,
        "crowd_density": crowd_density,
        "flow_entropy": flow_entropy,
        "person_count": len(boxes),
    }
