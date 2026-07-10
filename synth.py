"""
Synthetic crowd-clip generator (shared by make_test_video.py and train_classifier.py).

Produces short clips of textured noise sprites moving over a FLAT background, so
dense optical flow is well-defined and the background contributes exactly zero
motion. Two knobs control behaviour:

  speed     : sprite speed in px/frame (low = calm stroll, high = sprint)
  coherence : 0..1. 1.0 = every sprite moves the same direction (orderly crowd),
              0.0 = every sprite a random direction (chaotic scatter).

This lets us sweep a whole spectrum from clearly-normal to clearly-panic and label
each frame by construction, which is exactly what a trainable classifier needs.
"""
import numpy as np

W = H = 480
R = 14           # sprite half-size
N = 45           # number of agents


def make_sprites(rng):
    # 3-channel rich-texture sprites => strong, unambiguous optical flow
    return [np.repeat(rng.integers(0, 255, (2 * R, 2 * R, 1), dtype=np.uint8), 3, axis=2)
            for _ in range(N)]


def _bg():
    return np.full((H, W, 3), 90, dtype=np.uint8)


def _step(pos, vel):
    pos = pos + vel
    for d, lim in ((0, W), (1, H)):                  # reflect off walls
        vel[pos[:, d] < R, d] *= -1
        vel[pos[:, d] > lim - R, d] *= -1
        pos[:, d] = np.clip(pos[:, d], R, lim - R)
    return pos, vel


def gen_clip(speed, coherence, n_frames, seed):
    """Yield BGR frames of a crowd with the given speed and directional coherence."""
    rng = np.random.default_rng(seed)
    sprites = make_sprites(rng)
    bg = _bg()
    pos = rng.uniform(R, W - R, (N, 2)).astype(np.float64)

    base_ang = rng.uniform(0, 2 * np.pi)             # the crowd's dominant heading
    for f in range(n_frames):
        # coherence blends a shared heading with per-agent random headings.
        # Re-roll the random component periodically so panic keeps churning.
        if f % 8 == 0:
            jitter = rng.uniform(0, 2 * np.pi, N)
            ang = coherence * base_ang + (1 - coherence) * jitter
            spd = speed * rng.uniform(0.8, 1.2, N)
            vel = np.stack([spd * np.cos(ang), spd * np.sin(ang)], axis=1)
        frame = bg.copy()
        for i, (x, y) in enumerate(pos):
            xi, yi = int(x), int(y)
            frame[yi - R:yi + R, xi - R:xi + R] = sprites[i]
        yield frame
        pos, vel = _step(pos, vel)
