import sys
import time
import threading
import cv2
import numpy as np
from app import config
from app.logger import log
from app.device import pick_device, onnx_providers, describe, available_devices
from app.detector import NsfwDetector
from app.censor import BlurEngine, expand_box
from app.tracker import PersonTracker
from app.db import Database

BOX_COLOR = (60, 200, 255)
TEXT_COLOR = (20, 20, 20)

class Engine:
    def __init__(self):
        self.device = pick_device()
        self.providers = onnx_providers(self.device)
        self.detector = NsfwDetector(self.providers)
        self.blur = BlurEngine(self.device)
        self.database = Database()
        self.database.connect(device=self.device, providers=self.providers)
        self.censor_classes = set(config.DEFAULT_CENSOR_CLASSES)
        self.record_classes = set(config.RECORD_CLASSES)
        self.padding = config.BOX_PADDING
        self.score_threshold = config.SCORE_THRESHOLD
        self.censor_enabled = True
        self.show_boxes = True
        self.tracker = PersonTracker()
        self.lock = threading.Lock()
        self.marks = {}
        self.total_detections = 0
        self.total_blurred = 0
        log(describe(self.device, self.providers))
        log("available devices " + ",".join(available_devices()))

    def set_blur_strength(self, value):
        self.blur.set_strength(value)

    def set_padding(self, value):
        self.padding = max(0, int(value))

    def set_threshold(self, value):
        self.score_threshold = float(value)

    def set_classes(self, classes):
        self.censor_classes = set(classes)

    def toggle_censor(self, enabled):
        self.censor_enabled = bool(enabled)

    def _should_store(self, source, person_no, label, now):
        if source != "stream" or config.DB_LOG_INTERVAL <= 0:
            return True
        key = (source, person_no, label)
        last = self.marks.get(key, 0.0)
        if now - last < config.DB_LOG_INTERVAL:
            return False
        self.marks[key] = now
        if len(self.marks) > 4096:
            self.marks = {item: stamp for item, stamp in self.marks.items() if now - stamp < 60.0}
        return True

    def process(self, frame, source="stream", tracker=None, annotate=False, store=True):
        height, width = frame.shape[:2]
        detections = self.detector.detect(frame, score_threshold=self.score_threshold)
        active_tracker = tracker if tracker is not None else self.tracker
        persons = active_tracker.assign([item["box"] for item in detections])
        results = []
        regions = []
        now = time.time()
        for index, item in enumerate(detections):
            blurred = self.censor_enabled and item["label"] in self.censor_classes
            region = expand_box(item["box"], self.padding, width, height)
            if blurred:
                regions.append(region)
            results.append({
                "person_no": int(persons[index]),
                "label": item["label"],
                "score": item["score"],
                "box": item["box"],
                "region": [region[0], region[1], region[2] - region[0], region[3] - region[1]],
                "blurred": bool(blurred),
            })
        if regions:
            frame = self.blur.blur_regions(frame, regions)
        if annotate and self.show_boxes:
            frame = self._annotate(frame, results)
        if store:
            self._store(source, results, now)
        self.total_detections += len(results)
        self.total_blurred += len(regions)
        return frame, results

    def _store(self, source, results, now):
        for item in results:
            if item["label"] not in self.record_classes:
                continue
            if not self._should_store(source, item["person_no"], item["label"], now):
                continue
            state = "blurred" if item["blurred"] else "not blurred"
            log("detected " + item["label"] + " person " + str(item["person_no"]) + " score " + str(item["score"]) + " source " + source + " " + state)
            self.database.record(source, item["person_no"], item["label"], item["score"], item["box"], item["blurred"])

    def _annotate(self, frame, results):
        for item in results:
            x, y, w, h = item["region"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, 2)
            caption = "person " + str(item["person_no"]) + " " + item["label"].lower().replace("_", " ")
            size = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            top = max(0, y - size[1] - 8)
            cv2.rectangle(frame, (x, top), (x + size[0] + 10, top + size[1] + 8), BOX_COLOR, -1)
            cv2.putText(frame, caption, (x + 5, top + size[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)
        return frame

    def process_bytes(self, payload, source="api", store=True):
        buffer = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cannot decode image")
        return self.process(frame, source=source, tracker=PersonTracker(), annotate=False, store=store)

    def shutdown(self):
        self.database.close()

class CameraWorker(threading.Thread):
    def __init__(self, engine, index=None):
        super().__init__(daemon=True)
        self.engine = engine
        self.index = config.CAMERA_INDEX if index is None else int(index)
        self.capture = None
        self.running = False
        self.frame = None
        self.raw = None
        self.results = []
        self.fps = 0.0
        self.latency = 0.0
        self.error = ""
        self.lock = threading.Lock()
        self.started = threading.Event()

    def open(self):
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        capture = cv2.VideoCapture(self.index, backend)
        if not capture.isOpened():
            capture = cv2.VideoCapture(self.index)
        if not capture.isOpened():
            self.error = "camera " + str(self.index) + " not available"
            log(self.error)
            return None
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.error = ""
        log("camera " + str(self.index) + " opened")
        return capture

    def run(self):
        self.capture = self.open()
        if self.capture is None:
            self.running = False
            self.started.set()
            return
        self.running = True
        self.started.set()
        self.engine.tracker.reset()
        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            started = time.time()
            raw = frame.copy()
            processed, results = self.engine.process(frame, source="stream", annotate=True)
            elapsed = time.time() - started
            with self.lock:
                self.frame = processed
                self.raw = raw
                self.results = results
                self.latency = elapsed * 1000.0
                instant = 1.0 / max(elapsed, 0.001)
                self.fps = instant if self.fps == 0.0 else self.fps * 0.9 + instant * 0.1
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        log("camera " + str(self.index) + " closed")

    def stop(self):
        self.running = False
        self.join(timeout=3.0)

    def snapshot(self):
        with self.lock:
            if self.frame is None:
                return None, [], 0.0
            return self.frame.copy(), list(self.results), self.fps

    def jpeg(self, quality=80):
        frame, _, _ = self.snapshot()
        if frame is None:
            return None
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None
        return buffer.tobytes()

class Runtime:
    engine = None
    camera = None
    lock = threading.Lock()

    @classmethod
    def get_engine(cls):
        with cls.lock:
            if cls.engine is None:
                cls.engine = Engine()
            return cls.engine

    @classmethod
    def start_camera(cls, index=None):
        engine = cls.get_engine()
        with cls.lock:
            if cls.camera is not None and cls.camera.running:
                if index is None or int(index) == cls.camera.index:
                    return cls.camera
                cls.camera.stop()
                cls.camera = None
            worker = CameraWorker(engine, index)
            worker.start()
            cls.camera = worker
        worker.started.wait(timeout=20.0)
        for _ in range(100):
            if worker.frame is not None or not worker.running:
                break
            time.sleep(0.05)
        return worker

    @classmethod
    def stop_camera(cls):
        with cls.lock:
            if cls.camera is not None:
                cls.camera.stop()
                cls.camera = None

    @classmethod
    def get_camera(cls):
        return cls.camera

    @classmethod
    def shutdown(cls):
        cls.stop_camera()
        with cls.lock:
            if cls.engine is not None:
                cls.engine.shutdown()
                cls.engine = None