import threading

import numpy as np
import pytest

from app import config
from app.engine import CameraWorker


class FakeCapture:
    """Yields `frames` good frames, then fails every read."""

    def __init__(self, frames=1000):
        self.frames = frames
        self.released = False

    def read(self):
        if self.frames <= 0:
            return False, None
        self.frames -= 1
        return True, np.zeros((8, 8, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class FakeEngine:
    def __init__(self, fail_first=0, always_fail=False):
        self.fail_first = fail_first
        self.always_fail = always_fail
        self.calls = 0
        self.tracker = type("T", (), {"reset": lambda self: None})()

    def process(self, frame, source="stream", annotate=False):
        self.calls += 1
        if self.always_fail or self.calls <= self.fail_first:
            raise RuntimeError("boom")
        return frame, []


def make_worker(engine, capture):
    worker = CameraWorker(engine, 0)
    worker.open = lambda: capture
    return worker


def run_to_completion(worker, timeout=5.0):
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "worker loop did not terminate"


@pytest.fixture(autouse=True)
def fast_limits(monkeypatch):
    monkeypatch.setattr(config, "CAMERA_MAX_ERRORS", 3)
    monkeypatch.setattr(config, "CAMERA_READ_TIMEOUT", 0.2)


def test_repeated_processing_errors_stop_the_worker():
    capture = FakeCapture()
    worker = make_worker(FakeEngine(always_fail=True), capture)
    run_to_completion(worker)
    assert worker.running is False
    assert "frame processing failed" in worker.error
    assert capture.released


def test_unexpected_exception_does_not_leave_worker_marked_running():
    class Exploding(FakeCapture):
        def read(self):
            raise OSError("device vanished")

    capture = Exploding()
    worker = make_worker(FakeEngine(), capture)
    with pytest.raises(OSError):
        worker.run()
    assert worker.running is False
    assert capture.released


def test_occasional_errors_are_survived_and_cleared():
    capture = FakeCapture(frames=20)
    engine = FakeEngine(fail_first=2)
    worker = make_worker(engine, capture)
    run_to_completion(worker)  # ends via read timeout once frames run out
    assert engine.calls >= 3
    assert worker.frame is not None
    assert "stopped delivering frames" in worker.error


def test_camera_that_stops_delivering_frames_ends_the_worker():
    capture = FakeCapture(frames=0)
    worker = make_worker(FakeEngine(), capture)
    run_to_completion(worker)
    assert worker.running is False
    assert "stopped delivering frames" in worker.error
    assert capture.released


def test_failed_open_leaves_worker_not_running():
    worker = make_worker(FakeEngine(), None)
    worker.run()
    assert worker.running is False
    assert worker.started.is_set()
