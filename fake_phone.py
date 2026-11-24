"""
Simulate a phone camera: stream a video file into a room over the same ingest
WebSocket and protocol the phone page (camera.html) uses.

Useful for testing multi-camera rooms without a drawer full of phones, for
load-testing (run several at once), and for demoing a PANIC alarm through the
live path, since real panic can't be staged in front of a phone.

    PY fake_phone.py --source panic_test.avi --room lobby --cam sim1 --every-frame
    PY fake_phone.py --source demo_crowd_30s.avi --cam sim2 --loop
    # through a tunnel (token required there):
    PY fake_phone.py --url wss://xxxx.ngrok-free.app --token <token> --source panic_test.avi

Like the phone page, it resizes to a 640px long side, sends JPEG at quality 60,
and waits for the server's ack before sending the next frame.
"""
import argparse
import json
import sys
import time
import urllib.parse

import cv2
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

LONG_SIDE = 640
QUALITY = 60


def parse_args():
    p = argparse.ArgumentParser(description="Push a video file into a room like a phone would")
    p.add_argument("--source", required=True, help="video file to stream")
    p.add_argument("--url", default="ws://127.0.0.1:8000", help="server base (ws:// or wss://)")
    p.add_argument("--room", default="lobby")
    p.add_argument("--cam", default="sim1")
    p.add_argument("--token", default="", help="access token (not needed on localhost)")
    p.add_argument("--fps", type=float, default=10.0, help="send rate (phone default: 10)")
    p.add_argument("--loop", action="store_true", help="restart the file when it ends")
    p.add_argument("--every-frame", action="store_true",
                   help="send every frame of the file instead of resampling it to --fps "
                        "(plays slower than real time when the file is faster than --fps, "
                        "but keeps its per-frame motion exactly; use it for short clips "
                        "such as panic_test.avi, whose 6s calm lead-in is otherwise shorter "
                        "than the baseline's 90-frame warm-up)")
    return p.parse_args()


def frames(path, fps, loop, every_frame=False):
    """Yield frames from `path`, resampled from the file's own rate to `fps`, so
    the motion between consecutive sent frames matches what a phone sending at
    `fps` would see. With every_frame, yield all of them."""
    while True:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            sys.exit(f"cannot open {path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        step = 1.0 if every_frame else max(src_fps / fps, 1.0)
        i, next_pick = 0, 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i >= next_pick:
                next_pick += step
                yield frame
            i += 1
        cap.release()
        if not loop:
            return


def encode(frame):
    h, w = frame.shape[:2]
    s = min(1.0, LONG_SIDE / max(h, w))
    if s < 1.0:
        frame = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    return buf.tobytes() if ok else None


def main():
    a = parse_args()
    q = urllib.parse.urlencode({"token": a.token}) if a.token else ""
    url = f"{a.url.rstrip('/')}/ws/ingest/{a.room}/{a.cam}" + (f"?{q}" if q else "")
    period = 1.0 / a.fps
    sent, last_label, t0 = 0, None, time.monotonic()

    try:
        with connect(url, max_size=2 ** 20, open_timeout=10) as ws:
            print(f"[fake_phone] streaming {a.source} -> {a.room}/{a.cam} at {a.fps:g} fps")
            next_t = time.monotonic()
            for frame in frames(a.source, a.fps, a.loop, a.every_frame):
                data = encode(frame)
                if data is None:
                    continue
                ws.send(data)
                ack = json.loads(ws.recv(timeout=10))
                sent += 1
                label = "LEARNING" if ack["warming_up"] else ack["label"]
                if label != last_label:
                    room = f"  room alarm: {ack['panic_cams']}" if ack["room_alarm"] else ""
                    print(f"[fake_phone] t={time.monotonic() - t0:6.1f}s frame {sent:5d}  {label}{room}")
                    last_label = label
                next_t += period
                delay = next_t - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.monotonic()
    except ConnectionClosed as e:
        reason = {4401: "access token missing or wrong",
                  4409: "that camera name is already live in this room",
                  4400: "invalid room/camera name"}.get(e.rcvd.code if e.rcvd else None, str(e))
        sys.exit(f"[fake_phone] server closed the connection: {reason}")
    except KeyboardInterrupt:
        pass
    print(f"[fake_phone] done, sent {sent} frames")


if __name__ == "__main__":
    main()
