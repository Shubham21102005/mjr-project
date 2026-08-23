"""
Scene-relative panic detection.

WHY THIS EXISTS
---------------
The absolute-threshold approach does not work across cameras, and that is
measured, not suspected. Scored against UMN Crowd-Activity-All.avi (11 real
crowd escapes, ground truth from umn_labels.py), the globally-trained model
detects 0 of 1136 panic frames — recall 0.000. The reason:

    UMN real PANIC       avg_speed 0.397   flow_entropy 1.408
    demo_crowd real CALM avg_speed 0.520   flow_entropy 1.659

Real panic is SLOWER, frame-relative, than real calm — because UMN is a wide
overhead shot and demo_crowd is a closer one. compute_metrics normalises by
frame diagonal, which cancels resolution but not camera distance or field of
view. Every feature overlaps between those two groups, so no global threshold
or model over this feature set can separate them.

Retraining on the real footage does not fix it either; it inverts the failure
(recall 1.000 on a held-out UMN scene, but then it flags 159/199 real calm
frames as panic, because the scaler ends up centred on UMN's scene statistics).

THE FIX
-------
Stop asking "is this fast in absolute terms?" and ask "is this much faster than
normal FOR THIS CAMERA?". This is also what the literature does: Bird et al.,
"Holistic Features For Real-Time Crowd Behaviour Anomaly Detection" (arXiv
1606.05310), train on each scene's own initial normal period and flag outliers
with a threshold derived from that scene's distribution — and they report their
cross-scene variant as clearly inferior. Same conclusion, arrived at from the
other direction.

So: keep a rolling picture of what this feed looks like when calm, and flag a
frame when it departs from that by more than k robust standard deviations.

    z_i = (ema_i - median_i) / scale_i          per feature
    panic vote = max_i(z_i) > k

Details that matter, each of which was measured:

* MEDIAN + MAD, not mean + std. Panic frames are outliers by construction; a
  mean-based baseline gets dragged toward them.
* EMA SMOOTHING (alpha=0.15) on the raw features before scoring. Frame-to-frame
  flow is noisy; smoothing lifted tuning F1 from 0.548 to 0.718.
* THE BASELINE FREEZES while the label is PANIC. Otherwise a long escape is
  absorbed into "normal" and the detector talks itself out of the alarm.
* SCALE FLOORS. Scene 2's calm direction_variance has a median of 0.007 and a
  near-zero MAD; without a floor the z-score explodes on meaningless jitter.
* WARM-UP. The first `warmup` frames only build the baseline and never vote,
  so the detector cannot fire before it knows what calm looks like here.

k=4.0, alpha=0.15 and max-combination were tuned on UMN scenes 1 and 2 ONLY;
scene 3 was held out and never looked at during tuning.
"""
from collections import deque

import numpy as np

# The three flow-derived metrics. crowd_density and person_count are excluded
# on purpose: they depend on YOLO finding people, which is unreliable in the
# wide, low-resolution shots this is meant to work on.
FEATURES = ["avg_speed", "direction_variance", "flow_entropy"]

# Lower bounds on the robust scale, one per feature, in the metric's own units.
# These stop a very quiet baseline from turning noise into a huge z-score.
SCALE_FLOORS = np.array([0.03, 0.03, 0.08])

DEFAULTS = dict(warmup=90, window=600, k=4.0, alpha=0.15, rel_floor=0.15)


class SceneBaseline:
    """Learns one camera's idea of 'normal' and scores departures from it."""

    def __init__(self, features=None, warmup=None, window=None, k=None,
                 alpha=None, rel_floor=None):
        self.features = list(features or FEATURES)
        self.warmup = DEFAULTS["warmup"] if warmup is None else warmup
        self.k = DEFAULTS["k"] if k is None else k
        self.alpha = DEFAULTS["alpha"] if alpha is None else alpha
        self.rel_floor = DEFAULTS["rel_floor"] if rel_floor is None else rel_floor
        self.history = deque(maxlen=DEFAULTS["window"] if window is None else window)
        self.ema = None
        self.last_score = 0.0
        self.floors = SCALE_FLOORS[:len(self.features)]

    @property
    def warming_up(self) -> bool:
        """True while still learning this feed's baseline; no votes are cast."""
        return len(self.history) < self.warmup

    def _vector(self, metrics: dict) -> np.ndarray:
        return np.array([float(metrics.get(f, 0.0)) for f in self.features])

    def score(self, metrics: dict) -> float:
        """Robust z of the current frame against this feed's calm baseline.

        Returns 0.0 while warming up. Does NOT mutate the baseline — call
        observe() for that, so the caller controls freezing.
        """
        x = self._vector(metrics)
        self.ema = x if self.ema is None else self.alpha * x + (1 - self.alpha) * self.ema
        if self.warming_up:
            self.last_score = 0.0
            return 0.0
        H = np.array(self.history)
        med = np.median(H, axis=0)
        mad = np.median(np.abs(H - med), axis=0) * 1.4826       # -> ~sigma
        scale = np.maximum(np.maximum(mad, self.rel_floor * np.abs(med)), self.floors)
        self.last_score = float(((self.ema - med) / scale).max())
        return self.last_score

    def observe(self, frozen: bool = False) -> None:
        """Fold the current smoothed frame into the baseline.

        `frozen` should be True while the label is PANIC, so an ongoing escape
        is never learned as this scene's normal.
        """
        if self.ema is None:
            return
        if self.warming_up or not frozen:
            self.history.append(self.ema.copy())

    def is_panic(self, metrics: dict, frozen: bool = False) -> bool:
        s = self.score(metrics)
        self.observe(frozen=frozen)
        return (not self.warming_up) and s > self.k
