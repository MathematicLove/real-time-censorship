import threading

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config
from app.api import api
from app.engine import Runtime


def jpeg(width=64, height=48):
    ok, buf = cv2.imencode(".jpg", np.zeros((height, width, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture(scope="module")
def client():
    with TestClient(api) as c:
        yield c


def post(client, path, data):
    return client.post(path, files={"file": ("a.jpg", data, "image/jpeg")})


class TestUploads:
    def test_detect_blank_image(self, client):
        r = post(client, "/detect?store=false", jpeg())
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 0 and body["detections"] == []

    def test_censor_returns_jpeg(self, client):
        r = post(client, "/censor", jpeg())
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"
        assert r.headers["x-detections"] == "0"

    @pytest.mark.parametrize("path", ["/detect", "/censor"])
    def test_empty_file_is_rejected(self, client, path):
        assert post(client, path, b"").status_code == 400

    @pytest.mark.parametrize("path", ["/detect", "/censor"])
    def test_garbage_is_rejected(self, client, path):
        assert post(client, path, b"not an image").status_code == 400

    @pytest.mark.parametrize("path", ["/detect", "/censor"])
    def test_too_large_upload_is_rejected(self, client, monkeypatch, path):
        monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0.001)  # ~1 KB
        r = post(client, path, b"x" * 5000)
        assert r.status_code == 413

    def test_upload_at_the_limit_is_accepted(self, client, monkeypatch):
        data = jpeg()
        monkeypatch.setattr(config, "MAX_UPLOAD_MB", len(data) / (1024 * 1024))
        assert post(client, "/detect?store=false", data).status_code == 200


class TestSettings:
    def test_unknown_class_is_rejected(self, client):
        r = client.post("/settings", json={"censor_classes": ["NOPE"]})
        assert r.status_code == 400

    def test_update_and_read_back(self, client):
        r = client.post("/settings", json={"padding": 40, "show_boxes": False})
        assert r.status_code == 200
        assert r.json()["padding"] == 40 and r.json()["show_boxes"] is False
        client.post("/settings", json={"padding": config.BOX_PADDING, "show_boxes": True})


class TestEngineLocking:
    def test_setters_wait_for_a_frame_in_progress(self, client):
        engine = Runtime.get_engine()
        done = threading.Event()

        def change():
            engine.set_padding(10)
            done.set()

        engine.lock.acquire()
        try:
            t = threading.Thread(target=change, daemon=True)
            t.start()
            assert not done.wait(0.3), "setter ran while a frame was being processed"
        finally:
            engine.lock.release()
        assert done.wait(2.0)
        engine.set_padding(config.BOX_PADDING)

    def test_concurrent_requests_all_succeed(self, client):
        data = jpeg()
        codes = []

        def hit():
            codes.append(post(client, "/detect?store=false", data).status_code)

        threads = [threading.Thread(target=hit) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert codes == [200] * 6
