import os
import pickle
from collections import deque

from baseline import SceneBaseline

DEFAULT_MODEL_PATH = "panic_model.pkl"

# Which per-frame decision rule to use. Overridable with the PANIC_BACKEND env
# var so the CLI, server and evaluation harness can all be switched at once.
#
#   scene-baseline  (default) — flag frames that depart from what THIS feed
#       looks like when calm. The only backend that detects real panic; see
#       baseline.py for the measurements behind that claim.
#   trained-model   — the pickled scikit-learn model. Works on footage that
#       resembles its synthetic calibration, and scores recall 0.000 on real
#       UMN panic footage. Kept for comparison and reproducibility.
#   threshold-vote  — transparent fixed thresholds, no model file needed.
DEFAULT_BACKEND = os.environ.get("PANIC_BACKEND", "scene-baseline")

# Thresholds calibrated to sit just above a NORMAL walking crowd (measured on the
# 238x158 demo clip), so normal footage stays NORMAL and genuine panic — faster and
# more chaotic — trips >= 2 of the 4 votes. These are ABSOLUTE and therefore only
# valid for footage framed like the calibration clip; that limitation is exactly
# what the scene-baseline backend exists to remove.
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

    Per-frame decision uses one of three backends (see DEFAULT_BACKEND above):

      * SCENE BASELINE (default) — learns this feed's own calm statistics and
        flags robust-z departures from them. Scene-relative, so it transfers
        across cameras; the other two do not.
      * TRAINED MODEL — a scikit-learn model loaded from panic_model.pkl.
      * THRESHOLD VOTE — a frame is "panic" when >= 2 of 4 metrics exceed their
        fixed thresholds. Keeps the detector working with zero setup.

    Either way the window label uses HYSTERESIS to avoid flickering: it switches
    to PANIC only when >= enter_fraction of recent frames voted panic, and drops
    back to NORMAL only when the ratio falls to <= exit_fraction. Between those
    two bounds it holds its current state, so a ratio wobbling around the
    boundary can no longer toggle the label every frame.
    """

    def __init__(self, thresholds=None, window_size=10,
                 enter_fraction=0.6, exit_fraction=0.3,
                 model_path=DEFAULT_MODEL_PATH, use_model=True,
                 backend=None, baseline_kwargs=None):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.window = deque(maxlen=window_size)
        self.enter_fraction = enter_fraction
        self.exit_fraction = exit_fraction
        self.state = "NORMAL"

        self.model = None
        self.features = None
        self.load_error = None
        self.baseline = None

        requested = backend or DEFAULT_BACKEND
        if not use_model and requested == "trained-model":
            requested = "threshold-vote"

        if requested == "scene-baseline":
            self.baseline = SceneBaseline(**(baseline_kwargs or {}))
        elif requested == "trained-model" and model_path and os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    bundle = pickle.load(f)
                self.model = bundle["model"]
                self.features = bundle["features"]
            except Exception as e:
                # A model file that is PRESENT but unloadable is a broken install
                # (usually a missing or too-old scikit-learn), not a deliberate
                # choice to run on thresholds. Failing quietly made those two cases
                # indistinguishable, so say so loudly and keep the reason on the
                # instance for the dashboard to surface.
                self.load_error = f"{type(e).__name__}: {e}"
                print(f"[classifier] WARNING: {model_path} exists but could not "
                      f"be loaded ({self.load_error}).")
                print("[classifier] Falling back to threshold-vote -- the TRAINED "
                      "MODEL IS NOT IN USE. Fix: pip install -r requirements.txt")

    @property
    def backend(self) -> str:
        if self.baseline is not None:
            return "scene-baseline"
        return "trained-model" if self.model is not None else "threshold-vote"

    @property
    def degraded(self) -> bool:
        """True when a model file was found but could not be loaded, so the
        detector is silently running on the fallback instead of the model."""
        return self.load_error is not None

    @property
    def warming_up(self) -> bool:
        """scene-baseline only: still learning this feed's normal, not yet voting."""
        return self.baseline is not None and self.baseline.warming_up

    @property
    def anomaly_score(self) -> float:
        """scene-baseline only: robust z of the last frame against this feed."""
        return self.baseline.last_score if self.baseline is not None else 0.0

    def _is_frame_panic(self, metrics: dict) -> bool:
        if self.baseline is not None:
            # Freeze the baseline while already in PANIC, so a long escape is
            # never absorbed into this scene's idea of normal.
            return self.baseline.is_panic(metrics, frozen=(self.state == "PANIC"))
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
