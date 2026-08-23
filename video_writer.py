"""
Browser-playable video writing.

OpenCV's H.264 writer is unreliable across machines: its bundled openh264 DLL is
frequently missing or version-mismatched (it is on this one), and its `mp4v`
fallback produces MPEG-4 Part 2, which browsers refuse to play in <video>. Since
the whole point of an uploaded-clip result is that it plays back in the browser,
guessing at fourccs is not good enough.

So: pipe raw BGR frames into system ffmpeg and let libx264 do the encoding. That
yields H.264 / yuv420p / +faststart — the combination every browser plays and
can start streaming before the whole file arrives. If ffmpeg is absent we fall
back to cv2.VideoWriter and flag the result as possibly-unplayable rather than
pretending it worked.
"""
import shutil
import subprocess

import cv2
import numpy as np

FFMPEG = shutil.which("ffmpeg")


def ffmpeg_available() -> bool:
    return FFMPEG is not None


class BrowserVideoWriter:
    """Writes BGR frames to an MP4 that a browser can play.

    Dimensions are forced even — libx264 with yuv420p requires it and silently
    fails on odd sizes.
    """

    def __init__(self, path: str, width: int, height: int, fps: float):
        self.width = width - (width % 2)
        self.height = height - (height % 2)
        self.fps = max(1.0, float(fps))
        self.path = path
        self.proc = None
        self.cv = None
        self.using_ffmpeg = False

        if FFMPEG:
            cmd = [
                FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}", "-r", f"{self.fps}",
                "-i", "-",
                "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                path,
            ]
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            self.using_ffmpeg = True
        else:
            self.cv = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"avc1"), self.fps,
                (self.width, self.height),
            )

    def write(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if (w, h) != (self.width, self.height):
            frame = cv2.resize(frame, (self.width, self.height),
                               interpolation=cv2.INTER_AREA)
        if self.using_ffmpeg:
            # ffmpeg reads a flat bgr24 byte stream; a non-contiguous array (a
            # crop or a slice) would write garbage strides.
            self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        else:
            self.cv.write(frame)

    def close(self) -> str:
        """Finish the file. Returns '' on success or an error string."""
        if self.using_ffmpeg:
            try:
                self.proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            err = self.proc.stderr.read().decode("utf-8", "replace").strip()
            code = self.proc.wait()
            self.proc.stderr.close()
            if code != 0:
                return f"ffmpeg exited {code}: {err[:500]}"
            return ""
        else:
            self.cv.release()
            if not FFMPEG:
                return ("ffmpeg not found; wrote with OpenCV instead. If the clip "
                        "will not play in the browser, install ffmpeg and re-run.")
            return ""
