import os
import time
import asyncio
import threading
import cv2
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel, Field
from app import config, APP_NAME, APP_VERSION
from app.engine import Runtime
from app.device import available_devices
from app.logger import log, tail

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")

class Settings(BaseModel):
    censor_enabled: Optional[bool] = None
    show_boxes: Optional[bool] = None
    blur_strength: Optional[int] = Field(default=None, ge=3)
    padding: Optional[int] = Field(default=None, ge=0, le=400)
    score_threshold: Optional[float] = Field(default=None, ge=0.05, le=0.95)
    censor_classes: Optional[List[str]] = None

class CameraRequest(BaseModel):
    index: Optional[int] = None

@asynccontextmanager
async def lifespan(application):
    Runtime.get_engine()
    if config.AUTO_CAMERA:
        threading.Thread(target=Runtime.start_camera, daemon=True).start()
    log("api ready on http://" + config.API_HOST + ":" + str(config.API_PORT))
    yield
    Runtime.shutdown()

api = FastAPI(title=APP_NAME, version=APP_VERSION, lifespan=lifespan)
api.add_middleware(
    CORSMiddleware,
    allow_origins=[item.strip() for item in config.API_CORS.split(",") if item.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

@api.get("/", include_in_schema=False)
def page():
    return FileResponse(PAGE)

@api.get("/info")
def info():
    return {"name": APP_NAME, "version": APP_VERSION, "session": config.SESSION_ID}

@api.get("/health")
def health():
    engine = Runtime.get_engine()
    camera = Runtime.get_camera()
    return {
        "status": "ok",
        "database": engine.database.ready,
        "backend": engine.database.backend,
        "camera": bool(camera is not None and camera.running),
        "session": config.SESSION_ID,
    }

@api.get("/device")
def device():
    engine = Runtime.get_engine()
    return {
        "device": engine.device,
        "providers": engine.providers,
        "available": available_devices(),
        "inference_size": engine.detector.size,
    }

@api.get("/settings")
def get_settings():
    engine = Runtime.get_engine()
    return {
        "censor_enabled": engine.censor_enabled,
        "show_boxes": engine.show_boxes,
        "blur_strength": engine.blur.strength,
        "blur_max": config.BLUR_MAX,
        "padding": engine.padding,
        "score_threshold": engine.score_threshold,
        "censor_classes": sorted(engine.censor_classes),
        "all_classes": config.ALL_CLASSES,
    }

@api.post("/settings")
def update_settings(payload: Settings):
    engine = Runtime.get_engine()
    if payload.censor_enabled is not None:
        engine.toggle_censor(payload.censor_enabled)
    if payload.show_boxes is not None:
        engine.show_boxes = bool(payload.show_boxes)
    if payload.blur_strength is not None:
        engine.set_blur_strength(payload.blur_strength)
    if payload.padding is not None:
        engine.set_padding(payload.padding)
    if payload.score_threshold is not None:
        engine.set_threshold(payload.score_threshold)
    if payload.censor_classes is not None:
        unknown = [name for name in payload.censor_classes if name not in config.ALL_CLASSES]
        if unknown:
            raise HTTPException(status_code=400, detail="unknown classes " + ",".join(unknown))
        engine.set_classes(payload.censor_classes)
    return get_settings()

@api.post("/detect")
async def detect(file: UploadFile = File(...), store: bool = Query(default=True)):
    engine = Runtime.get_engine()
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty file")
    started = time.time()
    try:
        _, results = engine.process_bytes(payload, source="api", store=store)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    return {
        "count": len(results),
        "persons": sorted({item["person_no"] for item in results}),
        "elapsed_ms": round((time.time() - started) * 1000.0, 2),
        "detections": results,
    }

@api.post("/censor")
async def censor(file: UploadFile = File(...), quality: int = Query(default=90, ge=30, le=100)):
    engine = Runtime.get_engine()
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty file")
    try:
        frame, results = engine.process_bytes(payload, source="api")
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise HTTPException(status_code=500, detail="encode failed")
    headers = {
        "X-Detections": str(len(results)),
        "X-Blurred": str(len([item for item in results if item["blurred"]])),
        "X-Persons": str(len({item["person_no"] for item in results})),
    }
    return Response(content=buffer.tobytes(), media_type="image/jpeg", headers=headers)

@api.get("/logs")
def logs(limit: int = Query(default=100, ge=1, le=1000), source: Optional[str] = None, person_no: Optional[int] = None, blurred: Optional[bool] = None, session_id: Optional[str] = None):
    engine = Runtime.get_engine()
    if not engine.database.ready:
        return JSONResponse(status_code=503, content={"detail": "database not available"})
    rows = engine.database.recent(limit=limit, source=source, person_no=person_no, blurred=blurred, session_id=session_id)
    return {"count": len(rows), "rows": rows}

@api.get("/log/tail", include_in_schema=False)
def log_tail(limit: int = Query(default=100, ge=1, le=500)):
    return {"lines": tail(limit)}

@api.get("/stats")
def stats():
    engine = Runtime.get_engine()
    camera = Runtime.get_camera()
    payload = engine.database.stats()
    payload["session_detections"] = engine.total_detections
    payload["session_blurred"] = engine.total_blurred
    payload["fps"] = round(camera.fps, 2) if camera is not None else 0.0
    payload["latency_ms"] = round(camera.latency, 2) if camera is not None else 0.0
    return payload

@api.post("/camera/start")
def camera_start(payload: CameraRequest = CameraRequest()):
    worker = Runtime.start_camera(payload.index)
    if not worker.running:
        return JSONResponse(status_code=503, content={"running": False, "detail": worker.error or "camera failed to start"})
    return {"running": True, "index": worker.index}

@api.post("/camera/stop")
def camera_stop():
    Runtime.stop_camera()
    return {"running": False}

@api.get("/camera/status")
def camera_status():
    camera = Runtime.get_camera()
    if camera is None:
        return {"running": False, "index": config.CAMERA_INDEX, "fps": 0.0, "latency_ms": 0.0, "detections": 0, "persons": 0, "blurred": 0, "error": ""}
    results = list(camera.results)
    return {
        "running": camera.running,
        "index": camera.index,
        "fps": round(camera.fps, 2),
        "latency_ms": round(camera.latency, 2),
        "detections": len(results),
        "persons": len({item["person_no"] for item in results}),
        "blurred": len([item for item in results if item["blurred"]]),
        "error": camera.error,
    }

@api.get("/camera/frame")
def camera_frame(quality: int = Query(default=80, ge=30, le=100)):
    camera = Runtime.get_camera()
    if camera is None or not camera.running:
        raise HTTPException(status_code=409, detail="camera is not running")
    payload = camera.jpeg(quality)
    if payload is None:
        raise HTTPException(status_code=503, detail="no frame yet")
    return Response(content=payload, media_type="image/jpeg")

@api.get("/camera/stream")
def camera_stream(quality: int = Query(default=75, ge=30, le=100), fps: int = Query(default=25, ge=1, le=60)):
    camera = Runtime.get_camera()
    if camera is None or not camera.running:
        raise HTTPException(status_code=409, detail="camera is not running")
    delay = 1.0 / float(fps)

    async def generator():
        boundary = b"--frame\r\n"
        while camera.running:
            payload = camera.jpeg(quality)
            if payload is not None:
                yield boundary + b"Content-Type: image/jpeg\r\nContent-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload + b"\r\n"
            await asyncio.sleep(delay)

    return StreamingResponse(generator(), media_type="multipart/x-mixed-replace; boundary=frame")