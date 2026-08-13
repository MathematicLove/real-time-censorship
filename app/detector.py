import os
import cv2
import numpy as np
import onnxruntime
import nudenet
from app import config
from app.logger import log

MODEL_PATH = os.path.join(os.path.dirname(nudenet.__file__), "320n.onnx")

class NsfwDetector:
    def __init__(self, providers, size=None, model_path=None):
        self.size = int(size or config.INFERENCE_SIZE)
        self.model_path = model_path or MODEL_PATH
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self.session = onnxruntime.InferenceSession(self.model_path, sess_options=options, providers=providers)
        self.providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name
        self.labels = list(config.ALL_CLASSES)
        self.warmup()

    def warmup(self):
        blank = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        for _ in range(2):
            self.detect(blank)
        log("model loaded from " + os.path.basename(self.model_path) + " active providers " + ",".join(self.providers))

    def _preprocess(self, frame):
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        height, width = frame.shape[:2]
        side = max(height, width)
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        canvas[:height, :width] = frame
        resized = cv2.resize(canvas, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
        return blob, float(side) / float(self.size), width, height

    def _postprocess(self, raw, scale, width, height, score_threshold, nms_threshold):
        matrix = raw[0][0]
        class_scores = matrix[4:, :]
        class_ids = np.argmax(class_scores, axis=0)
        scores = class_scores[class_ids, np.arange(class_scores.shape[1])]
        keep = scores >= score_threshold
        if not np.any(keep):
            return []
        scores = scores[keep]
        class_ids = class_ids[keep]
        cx = matrix[0, keep] * scale
        cy = matrix[1, keep] * scale
        bw = matrix[2, keep] * scale
        bh = matrix[3, keep] * scale
        x = np.clip(cx - bw / 2.0, 0, width)
        y = np.clip(cy - bh / 2.0, 0, height)
        bw = np.minimum(bw, width - x)
        bh = np.minimum(bh, height - y)
        valid = (bw > 1) & (bh > 1)
        if not np.any(valid):
            return []
        boxes = np.stack([x[valid], y[valid], bw[valid], bh[valid]], axis=1)
        scores = scores[valid]
        class_ids = class_ids[valid]
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold, nms_threshold)
        if len(indices) == 0:
            return []
        indices = np.array(indices).reshape(-1)
        detections = []
        for index in indices:
            box = boxes[index]
            detections.append({
                "label": self.labels[int(class_ids[index])],
                "score": round(float(scores[index]), 4),
                "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
            })
        detections.sort(key=lambda item: item["score"], reverse=True)
        return detections

    def detect(self, frame, score_threshold=None, nms_threshold=None):
        score_threshold = float(config.SCORE_THRESHOLD if score_threshold is None else score_threshold)
        nms_threshold = float(config.NMS_THRESHOLD if nms_threshold is None else nms_threshold)
        blob, scale, width, height = self._preprocess(frame)
        raw = self.session.run(None, {self.input_name: blob})
        return self._postprocess(raw, scale, width, height, score_threshold, nms_threshold)