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
    dict with keys: avg_speed, direction_variance, crowd_density, flow_entropy
    """
    magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])  # (H_small, W_small)
    angle = cv2.phase(flow[..., 0], flow[..., 1])           # radians, 0 to 2π

    # Only consider pixels with meaningful motion
    mask = magnitude > 0.5

    avg_speed = float(np.mean(magnitude))

    if mask.sum() > 0:
        direction_variance = float(np.std(angle[mask]))
        angle_deg = np.degrees(angle[mask]) % 360
        hist, _ = np.histogram(angle_deg, bins=36, range=(0, 360))
        hist = hist.astype(float) + 1e-9  # Laplace smooth
        flow_entropy = float(scipy_entropy(hist))
    else:
        direction_variance = 0.0
        flow_entropy = 0.0

    H, W = frame_shape
    crowd_density = len(boxes) / (H * W) if (H * W) > 0 else 0.0

    return {
        "avg_speed": avg_speed,
        "direction_variance": direction_variance,
        "crowd_density": crowd_density,
        "flow_entropy": flow_entropy,
        "person_count": len(boxes),
    }
