"""Integration tests for the live multi-camera API in server.py.

These run the real app (real YOLO, real scheduler thread) in-process via
Starlette's TestClient and push real frames from demo_crowd_30s.avi through
the same ingest WebSocket a phone uses.
"""
import os
import time

os.environ["PANIC_TOKEN"] = "test-token"      # must be set before server is imported
os.environ.pop("PANIC_SOURCES", None)
os.environ.pop("PANIC_PUBLIC_URL", None)

import cv2
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import server

HERE = os.path.dirname(os.path.abspath(__file__))
CLIP = os.path.join(HERE, "demo_crowd_30s.avi")
TOKEN = "test-token"


def clip_jpegs(n):
    cap = cv2.VideoCapture(CLIP)
    out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 60])[1].tobytes())
    cap.release()
    assert out, f"could not read {CLIP}"
    return out


@pytest.fixture(scope="module")
def app_client():
    # The default TestClient peer is "testclient": a REMOTE client, so the token
    # gate applies, just as it does for anything arriving through a tunnel.
    with TestClient(server.app) as c:
        yield c


@pytest.fixture(scope="module")
def local_client():
    with TestClient(server.app, client=("127.0.0.1", 50000)) as c:
        yield c


def wait_for(pred, timeout=15.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.05)
    raise AssertionError("condition not met in time")


# -------------------------------------------------------------- token gate

def test_remote_request_without_token_is_refused(app_client):
    app_client.cookies.clear()
    assert app_client.get("/").status_code == 401
    assert app_client.get("/api/rooms").status_code == 401


def test_token_in_query_sets_a_cookie_that_keeps_working(app_client):
    app_client.cookies.clear()
    r = app_client.get(f"/camera?token={TOKEN}")
    assert r.status_code == 200 and "Panic Camera" in r.text
    assert app_client.cookies.get("panic_token") == TOKEN
    assert app_client.get("/api/rooms").status_code == 200       # cookie alone
    app_client.cookies.clear()


def test_wrong_token_is_refused(app_client):
    app_client.cookies.clear()
    assert app_client.get("/?token=nope").status_code == 401


def test_direct_local_browser_needs_no_token(local_client):
    assert local_client.get("/").status_code == 200


def test_local_request_through_a_tunnel_still_needs_the_token(local_client):
    # A tunnel agent connects from 127.0.0.1 but always adds forwarding headers.
    r = local_client.get("/api/rooms", headers={"X-Forwarded-For": "203.0.113.9"})
    assert r.status_code == 401


def test_websocket_without_token_is_closed_with_4401(app_client):
    app_client.cookies.clear()
    with pytest.raises(WebSocketDisconnect) as e:
        with app_client.websocket_connect("/ws/ingest/lobby/x") as ws:
            ws.receive_text()
    assert e.value.code == 4401


# ------------------------------------------------------------------ ingest

def test_phone_push_end_to_end(app_client):
    frames = clip_jpegs(12)
    with app_client.websocket_connect(f"/ws/ingest/e2e/phone1?token={TOKEN}") as ws:
        acks = []
        for jpeg in frames:
            ws.send_bytes(jpeg)
            acks.append(ws.receive_json())
            time.sleep(0.11)                 # ~9 fps, under the per-camera cap
        assert all(a["decoded"] for a in acks)
        assert acks[-1]["warming_up"] is True              # 12 frames < 90 warm-up
        assert acks[-1]["label"] == "NORMAL" and acks[-1]["room_alarm"] is False

        rooms = app_client.get(f"/api/rooms?token={TOKEN}").json()["rooms"]
        room = next(r for r in rooms if r["room"] == "e2e")
        cam = room["cameras"][0]
        assert cam["cam"] == "phone1" and cam["kind"] == "push" and cam["online"]
        assert cam["metrics"], "the pipeline should have produced metrics"

        # The dashboard socket delivers state and this camera's annotated JPEG.
        with app_client.websocket_connect(f"/ws/live?token={TOKEN}") as live_ws:
            got_state = got_frame = False
            for _ in range(50):
                msg = live_ws.receive()
                if msg.get("text"):
                    got_state = got_state or '"e2e"' in msg["text"]
                elif msg.get("bytes"):
                    b = msg["bytes"]
                    cid = b[1:1 + b[0]].decode()
                    if cid == "e2e/phone1":
                        assert b[1 + b[0]:][:2] == b"\xff\xd8"
                        got_frame = True
                if got_state and got_frame:
                    break
            assert got_state and got_frame

        # A second device can't stream under the same live name...
        with pytest.raises(WebSocketDisconnect) as e:
            with app_client.websocket_connect(f"/ws/ingest/e2e/phone1?token={TOKEN}") as dup:
                dup.send_bytes(frames[0])
                dup.receive_json()
        assert e.value.code == 4409

        # ...and a reset control message is accepted without dropping the stream.
        ws.send_text('{"type": "reset"}')
        ws.send_bytes(frames[0])
        assert ws.receive_json()["decoded"]

    # After disconnecting, the camera shows offline (and is pruned later).
    cam = server.REGISTRY.get("e2e", "phone1")
    wait_for(lambda: not cam.connected)
    assert not cam.snapshot()["online"]


def test_bad_names_are_rejected(app_client):
    with pytest.raises(WebSocketDisconnect) as e:
        with app_client.websocket_connect(f"/ws/ingest/bad%20room/x?token={TOKEN}") as ws:
            ws.receive_json()
    assert e.value.code == 4400


def test_undecodable_frame_is_acked_but_not_fed(app_client):
    with app_client.websocket_connect(f"/ws/ingest/junk/cam?token={TOKEN}") as ws:
        ws.send_bytes(b"not a jpeg")
        assert ws.receive_json()["decoded"] is False


# -------------------------------------------------------------------- join

def test_join_link_carries_room_cam_and_token(app_client):
    r = app_client.get("/api/join", params={"room": "lobby", "cam": "phone1",
                                            "base": "https://abc.ngrok-free.app/",
                                            "token": TOKEN})
    d = r.json()
    assert r.status_code == 200
    assert d["url"] == f"https://abc.ngrok-free.app/camera?room=lobby&cam=phone1&token={TOKEN}"
    assert d["secure"] is True
    assert "<svg" in d["svg"]


def test_public_url_is_read_from_a_running_ngrok_agent(monkeypatch):
    import io
    import json

    body = json.dumps({"tunnels": [
        {"public_url": "http://abc.ngrok-free.app"},
        {"public_url": "https://abc.ngrok-free.app"}]}).encode()

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(server.urllib.request, "urlopen", lambda *a, **k: Resp(body))
    monkeypatch.setitem(server._ngrok_cache, "t", 0.0)
    assert server.detect_public_url() == "https://abc.ngrok-free.app"

    monkeypatch.setenv("PANIC_PUBLIC_URL", "https://fixed.example.com/")
    assert server.detect_public_url() == "https://fixed.example.com"


def test_no_ngrok_agent_means_no_public_url(monkeypatch):
    def refuse(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr(server.urllib.request, "urlopen", refuse)
    monkeypatch.setitem(server._ngrok_cache, "t", 0.0)
    assert server.detect_public_url() is None


def test_join_needs_a_public_address(app_client, monkeypatch):
    monkeypatch.setattr(server, "detect_public_url", lambda: None)
    r = app_client.get("/api/join", params={"room": "lobby", "cam": "p", "token": TOKEN})
    assert r.status_code == 400


def test_join_rejects_bad_names_and_schemes(app_client):
    assert app_client.get("/api/join", params={"room": "a b", "cam": "p", "base": "https://x.io",
                                               "token": TOKEN}).status_code == 400
    assert app_client.get("/api/join", params={"room": "a", "cam": "p", "base": "javascript:alert(1)",
                                               "token": TOKEN}).status_code == 400
