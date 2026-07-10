"""
Generate a synthetic crowd clip that is calm, then panics.

  Phase 1 (~6s): agents drift slowly in one direction   -> NORMAL
  Phase 2 (~6s): agents sprint in random directions      -> PANIC

Each 'agent' is a textured noise sprite (so dense optical flow is well-defined)
moving over a FLAT background (which stays exactly 0 flow and is ignored). The
clip is written LOSSLESS so compression noise can't dilute the speed metric.
Feed the result to the UNMODIFIED detector:

    python detect.py --source panic_test.avi

You should see the label switch from green NORMAL to red PANIC ~halfway through.
"""
import cv2
import numpy as np

W = H = 480
FPS = 20
N = 45            # number of agents
R = 14            # sprite half-size
NORMAL_FRAMES = 120
PANIC_FRAMES = 120
rng = np.random.default_rng(42)

# Flat mid-gray background -> zero optical flow, so only the sprites count as motion
# and avg_speed reflects the real sprite speed instead of compression noise.
bg = np.full((H, W, 3), 90, dtype=np.uint8)

# One fixed noise sprite per agent (rich texture => strong, unambiguous flow).
sprites = [cv2.cvtColor(rng.integers(0, 255, (2 * R, 2 * R), dtype=np.uint8),
                        cv2.COLOR_GRAY2BGR) for _ in range(N)]
pos = rng.uniform(R, W - R, (N, 2)).astype(np.float64)


def draw(pos):
    frame = bg.copy()
    for i, (x, y) in enumerate(pos):
        xi, yi = int(x), int(y)
        frame[yi - R:yi + R, xi - R:xi + R] = sprites[i]
    return frame


def step(pos, vel):
    pos = pos + vel
    for d, lim in ((0, W), (1, H)):                 # reflect off walls, stay in-frame
        vel[pos[:, d] < R, d] *= -1
        vel[pos[:, d] > lim - R, d] *= -1
        pos[:, d] = np.clip(pos[:, d], R, lim - R)
    return pos, vel


# Prefer a lossless codec so texture (and thus flow magnitude) survives encoding.
OUT = "panic_test.avi"
writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"FFV1"), FPS, (W, H))
if not writer.isOpened():
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"HFYU"), FPS, (W, H))
assert writer.isOpened(), "no lossless VideoWriter codec available"

# Phase 1 — calm: slow coherent drift
vel = np.tile([2.5, 0.4], (N, 1)).astype(np.float64)
for _ in range(NORMAL_FRAMES):
    writer.write(draw(pos))
    pos, vel = step(pos, vel)

# Phase 2 — panic: fast, direction re-randomised every 8 frames
for f in range(PANIC_FRAMES):
    if f % 8 == 0:
        ang = rng.uniform(0, 2 * np.pi, N)
        spd = rng.uniform(20, 26, N)
        vel = np.stack([spd * np.cos(ang), spd * np.sin(ang)], axis=1)
    writer.write(draw(pos))
    pos, vel = step(pos, vel)

writer.release()
print(f"wrote {OUT}  ({NORMAL_FRAMES} calm + {PANIC_FRAMES} panic frames @ {FPS}fps)")
