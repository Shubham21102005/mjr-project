import os
import pickle
from collections import deque

DEFAULT_MODEL_PATH = "panic_model.pkl"

# Thresholds calibrated to sit just above a NORMAL walking crowd (measured on the
# 238x158 demo clip), so normal footage stays NORMAL and genuine panic — faster and
# more chaotic — trips >= 2 of the 4 votes. Validate against labelled panic footage
# before production use.
DEFAULT_THRESHOLDS = {
    "avg_speed": 1.8,          # % of frame diagonal / frame (resolution-independent)
    "direction_variance": 0.75,# circular variance, bounded [0,1] (0=coherent, 1=chaotic).
                               # Calibrated on the synthetic clip: calm frames mean ~0.46,
                               # panic frames mean ~0.70; 0.75 sits above the calm spread.
    "crowd_density": 0.18,     # occupancy: fraction of frame area covered by persons
    "flow_entropy": 3.2,       # nats, out of max ln(36)=3.58
}


class PanicClassifier:
    """
    Maintains a sliding window of per-frame panic votes and emits PANIC/NORMAL.

    Per-frame decision uses one of two backends:
      * TRAINED MODEL (preferred) — a scikit-learn model (train_classifier.py)
        loaded from panic_model.pkl predicts panic from the motion features.
      * THRESHOLD VOTE (fallback) — if no model file is present, a frame is
        "panic" when >= 2 of 4 metrics exceed their thresholds. This keeps the
        detector working with zero setup.

    Either way the window label uses HYSTERESIS to avoid flickering: it switches
    to PANIC only when >= enter_fraction of recent frames voted panic, and drops
    back to NORMAL only when the ratio falls to <= exit_fraction. Between those
    two bounds it holds its current state, so a ratio wobbling around the
    boundary can no longer toggle the label every frame.
    """

    def __init__(self, thresholds=None, window_size=10,
                 enter_fraction=0.6, exit_fraction=0.3,
                 model_path=DEFAULT_MODEL_PATH, use_model=True):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.window = deque(maxlen=window_size)
        self.enter_fraction = enter_fraction
        self.exit_fraction = exit_fraction
        self.state = "NORMAL"

        self.model = None
        self.features = None
        if use_model and model_path and os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    bundle = pickle.load(f)
                self.model = bundle["model"]
                self.features = bundle["features"]
            except Exception as e:
                print(f"[classifier] could not load {model_path} ({e}); using thresholds")

    @property
    def backend(self) -> str:
        return "trained-model" if self.model is not None else "threshold-vote"

    def _is_frame_panic(self, metrics: dict) -> bool:
        if self.model is not None:
            x = [[metrics.get(k, 0.0) for k in self.features]]
            return bool(self.model.predict(x)[0])
        triggered = sum([
            metrics.get("avg_speed", 0)          > self.thresholds["avg_speed"],
            metrics.get("direction_variance", 0) > self.thresholds["direction_variance"],
            metrics.get("crowd_density", 0)      > self.thresholds["crowd_density"],
            metrics.get("flow_entropy", 0)        > self.thresholds["flow_entropy"],
        ])
        return triggered >= 2

    def update(self, metrics: dict) -> str:
        self.window.append(self._is_frame_panic(metrics))
        ratio = self.panic_ratio()
        if self.state == "NORMAL" and ratio >= self.enter_fraction:
            self.state = "PANIC"
        elif self.state == "PANIC" and ratio <= self.exit_fraction:
            self.state = "NORMAL"
        return self.state

    def panic_ratio(self) -> float:
        if not self.window:
            return 0.0
        return sum(self.window) / len(self.window)
