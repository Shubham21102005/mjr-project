from collections import deque

DEFAULT_THRESHOLDS = {
    "avg_speed": 2.5,         # px/frame at flow resolution
    "direction_variance": 1.0, # radians (std of motion angles)
    "crowd_density": 0.0002,   # people / pixel (kept low — part of 2-of-4 vote)
    "flow_entropy": 2.8,       # out of max ~3.58 (ln 36)
}


class PanicClassifier:
    """
    Maintains a sliding window of per-frame panic votes and emits PANIC/NORMAL.

    A frame is "panic" if >= 2 of 4 metrics exceed their thresholds.
    The window label is PANIC if >= panic_fraction of recent frames voted panic.
    """

    def __init__(self, thresholds=None, window_size=10, panic_fraction=0.5):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.window = deque(maxlen=window_size)
        self.panic_fraction = panic_fraction

    def _is_frame_panic(self, metrics: dict) -> bool:
        triggered = sum([
            metrics.get("avg_speed", 0)          > self.thresholds["avg_speed"],
            metrics.get("direction_variance", 0) > self.thresholds["direction_variance"],
            metrics.get("crowd_density", 0)      > self.thresholds["crowd_density"],
            metrics.get("flow_entropy", 0)        > self.thresholds["flow_entropy"],
        ])
        return triggered >= 2

    def update(self, metrics: dict) -> str:
        self.window.append(self._is_frame_panic(metrics))
        return "PANIC" if self.panic_ratio() >= self.panic_fraction else "NORMAL"

    def panic_ratio(self) -> float:
        if not self.window:
            return 0.0
        return sum(self.window) / len(self.window)
