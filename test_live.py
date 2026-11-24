"""Unit tests for live.py: sources, registry, per-camera pipelines, scheduler.

The scheduler is driven with a stub detector (no YOLO weights needed), since
what's under test is the scheduling and per-camera state, not person detection.
"""
import time

import numpy as np
import pytest

import live


class _NoBoxes:
    boxes = None


class StubModel:
    """Stands in for YOLO: records batch sizes, finds no people."""

    def __init__(self):
        self.batches = []

    def __call__(self, frames, **kw):
        self.batches.append(len(frames))
        return [_NoBoxes() for _ in frames]


def frame(h=120, w=160, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------- parsing

def test_parse_sources_room_and_room_slash_cam():
    assert live.parse_sources("lobby=a.avi, lobby/door=0 ,gate/east=rtsp://x/y") == [
        ("lobby", "main", "a.avi"), ("lobby", "door", "0"), ("gate", "east", "rtsp://x/y")]


def test_parse_sources_ignores_junk():
    assert live.parse_sources("") == []
    assert live.parse_sources("noequals, =x, a=") == []


@pytest.mark.parametrize("name,ok", [("lobby", True), ("phone-1_a", True), ("", False),
                                     ("a b", False), ("../x", False), ("x" * 33, False)])
def test_valid_name(name, ok):
    assert live.valid_name(name) is ok


# -------------------------------------------------------------- frame slot

def test_slot_keeps_only_latest_and_counts_cuts():
    s = live.FrameSlot()
    a, b = frame(seed=1), frame(seed=2)
    s.put(a)
    s.put(b, cut=True)
    seq, epoch, f = s.get()
    assert seq == 2 and epoch == 1 and f is b


# ---------------------------------------------------------------- registry

def test_each_camera_has_its_own_classifier():
    reg = live.Registry()
    a, _ = reg.attach_push("lobby", "a")
    b, _ = reg.attach_push("lobby", "b")
    assert a.classifier is not b.classifier
    assert a.classifier.baseline is not b.classifier.baseline


def test_live_camera_name_is_refused():
    reg = live.Registry()
    c, _ = reg.attach_push("lobby", "a")
    c.slot.put(frame())                     # it is actively streaming
    with pytest.raises(live.CameraInUse):
        reg.attach_push("lobby", "a")


def test_silent_camera_can_be_taken_over_and_old_detach_is_ignored():
    reg = live.Registry()
    c, gen1 = reg.attach_push("lobby", "a")
    c.slot.put(frame())
    c.slot.last_seen -= live.OFFLINE_AFTER + 1          # the phone went quiet
    old_classifier = c.classifier

    c2, gen2 = reg.attach_push("lobby", "a")
    assert c2 is c and gen2 == gen1 + 1
    assert c.classifier is not old_classifier           # fresh baseline on re-attach

    reg.detach_push(c, gen1)                            # stale handler finally exits
    assert c.connected                                  # ...and must not kick the new one
    reg.detach_push(c, gen2)
    assert not c.connected


def test_pull_camera_cannot_be_hijacked_by_a_device():
    reg = live.Registry()
    reg._rooms["lobby"] = {"main": live.Camera("lobby", "main", "pull")}
    with pytest.raises(live.CameraInUse):
        reg.attach_push("lobby", "main")


def test_disconnected_push_cameras_are_pruned_with_their_room():
    reg = live.Registry()
    c, gen = reg.attach_push("lobby", "a")
    reg.detach_push(c, gen)
    reg.prune(time.monotonic() + live.PRUNE_AFTER - 1)
    assert reg.get("lobby", "a") is c
    reg.prune(time.monotonic() + live.PRUNE_AFTER + 1)
    assert reg.get("lobby", "a") is None
    assert reg.snapshot() == []


def test_room_alarm_is_any_online_camera_in_panic():
    reg = live.Registry()
    a, _ = reg.attach_push("lobby", "a")
    b, _ = reg.attach_push("lobby", "b")
    for c in (a, b):
        c.slot.put(frame())
    with b.out_lock:
        b.state["label"] = "PANIC"
    assert reg.room_alarm("lobby") == ["b"]
    room = reg.snapshot()[0]
    assert room["alarm"] and room["panic_cams"] == ["b"]


# ------------------------------------------------------------- scheduler

def test_scheduler_processes_due_cameras_and_publishes_jpegs():
    reg = live.Registry()
    sched = live.Scheduler(reg, workers=2)
    cams = [reg.attach_push("lobby", n)[0] for n in ("a", "b", "c")]
    for i, c in enumerate(cams):
        c.slot.put(frame(seed=i))

    assert sched.tick() == 3
    for c in cams:
        seq, jpeg = c.latest_jpeg()
        assert seq == 1 and jpeg[:2] == b"\xff\xd8"       # a JPEG was published

    assert sched.tick() == 0                              # nothing new: nothing processed


def test_fps_cap_holds_back_a_camera_that_was_just_processed():
    reg = live.Registry()
    sched = live.Scheduler(reg, workers=1)
    c, _ = reg.attach_push("lobby", "a")
    c.slot.put(frame(seed=1))
    assert sched.tick() == 1
    c.slot.put(frame(seed=2))
    assert sched.tick() == 0                              # too soon for LIVE_FPS
    c._last_processed -= 1.0
    assert sched.tick() == 1


def test_detector_batches_all_due_cameras_in_one_call():
    reg, model = live.Registry(), StubModel()
    det = live.Detector(reg, model)
    cams = [reg.attach_push("lobby", n)[0] for n in ("a", "b", "c")]
    for i, c in enumerate(cams):
        c.slot.put(frame(seed=i))
    assert det.tick() == 3
    assert model.batches == [3]                           # ONE batched YOLO call
    assert det.tick() == 0                                # same frames: not redone
    for c in cams:
        c.slot.put(frame(seed=9))
    assert det.tick() == 0                                # new frames, but DETECT_FPS cap
    for c in cams:
        c.det_at -= 1.0
    assert det.tick() == 3 and model.batches == [3, 3]


def test_analysis_does_not_wait_for_detection():
    reg = live.Registry()
    sched = live.Scheduler(reg, workers=1)
    c, _ = reg.attach_push("lobby", "a")
    c.slot.put(frame(seed=1))
    assert sched.tick() == 1                              # no Detector ran at all
    assert c.latest_jpeg()[1] is not None


def test_boxes_from_a_different_frame_shape_are_dropped():
    c = live.Camera("lobby", "a", "push")
    c.set_boxes(1, (120, 160, 3), [[1, 2, 3, 4]], time.monotonic())
    assert c.current_boxes((120, 160, 3)) == [[1, 2, 3, 4]]
    assert c.current_boxes((160, 120, 3)) == []           # rotated since detection


def _process(c, f, epoch=0):
    c._last_processed = 0.0
    seq = c._last_seq + 1
    c.process(seq, epoch, f, time.monotonic())


def test_flow_is_never_computed_across_a_cut():
    c = live.Camera("lobby", "a", "pull")
    _process(c, frame(seed=1), epoch=0)
    _process(c, frame(seed=2), epoch=0)
    assert c.snapshot()["metrics"]                        # consecutive frames: metrics
    _process(c, frame(seed=3), epoch=1)
    assert c.snapshot()["metrics"] == {}                  # across the cut: none


def test_frame_geometry_change_relearns_the_baseline():
    c = live.Camera("lobby", "a", "push")
    _process(c, frame(120, 160, seed=1))
    _process(c, frame(120, 160, seed=2))
    before = c.classifier
    _process(c, frame(160, 120, seed=3))                  # the phone was rotated
    assert c.classifier is not before
    assert c.snapshot()["metrics"] == {}
