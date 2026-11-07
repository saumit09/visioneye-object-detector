"""
FastAPI Server for Real-Time Webcam Object Recognition.
Provides WebSocket stream for low-latency browser detection, REST API, and static UI.
Optimized for cross-platform access (laptops, iOS Safari, Android Chrome).
"""

import os
import sys
import time
import base64
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from detector import ObjectDetector
from tracker import DetectionTracker

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="VisionEye - Object Detector", version="2.0.0")

# Mount static frontend directory
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Global detector instance
detector: Optional[ObjectDetector] = None


@app.on_event("startup")
def startup_event():
    global detector
    print("Pre-loading ObjectDetector ONNX model...")
    detector = ObjectDetector()
    print("Detector ready for inference.")


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def read_root():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>VisionEye Object Detector Running</h1>")


@app.get("/style.css")
def get_root_style():
    style_file = FRONTEND_DIR / "style.css"
    if style_file.exists():
        return FileResponse(style_file, media_type="text/css")
    return JSONResponse({"error": "style.css not found"}, status_code=404)


@app.get("/app.js")
def get_root_js():
    js_file = FRONTEND_DIR / "app.js"
    if js_file.exists():
        return FileResponse(js_file, media_type="application/javascript")
    return JSONResponse({"error": "app.js not found"}, status_code=404)


@app.get("/api/info")
def get_info():
    """Returns server and hardware acceleration status."""
    global detector
    provider_name = "CPU"
    if detector is not None and hasattr(detector, "session") and detector.session:
        providers = detector.session.get_providers()
        if providers:
            provider_name = providers[0]

    tunnel = os.environ.get("TUNNEL_URL", "https://repurpose-usage-trodden.ngrok-free.dev")

    return {
        "status": "online",
        "app": "VisionEye",
        "version": "2.0.0",
        "model": "YOLO11n (ONNX)",
        "provider": provider_name,
        "tunnel_url": tunnel
    }


class DetectRequest(BaseModel):
    image: str  # Base64 encoded JPEG/PNG
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45


@app.post("/api/detect")
async def detect_api(payload: DetectRequest):
    """REST endpoint for single image object detection."""
    global detector
    if detector is None:
        return JSONResponse({"error": "Detector not initialized"}, status_code=500)

    try:
        raw_b64 = payload.image
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(raw_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse({"error": "Could not decode image"}, status_code=400)

        t0 = time.perf_counter()
        detections = detector.detect(
            frame,
            conf_threshold=payload.conf_threshold,
            iou_threshold=payload.iou_threshold
        )
        duration_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "detections": detections,
            "count": len(detections),
            "inference_ms": round(duration_ms, 1),
            "image_size": {"width": frame.shape[1], "height": frame.shape[0]}
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video frame object recognition.
    Each connection gets its own DetectionTracker for temporal stabilization.
    """
    global detector
    await websocket.accept()

    tracker = DetectionTracker(
        iou_threshold=0.25,
        bbox_smoothing=0.55,
        conf_smoothing=0.45,
        min_hits_to_confirm=3,
        max_misses=8,
    )

    try:
        while True:
            data = await websocket.receive_json()
            raw_b64 = data.get("image", "")
            conf_thresh = float(data.get("conf_threshold", 0.35))
            iou_thresh = float(data.get("iou_threshold", 0.45))

            if not raw_b64:
                continue

            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]

            try:
                img_bytes = base64.b64decode(raw_b64)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception:
                await websocket.send_json({"error": "Invalid frame payload"})
                continue

            if frame is None:
                await websocket.send_json({"error": "Frame decode failed"})
                continue

            t0 = time.perf_counter()
            raw_detections = detector.detect(
                frame,
                conf_threshold=max(0.15, conf_thresh - 0.10),
                iou_threshold=iou_thresh
            )
            stabilized = tracker.update(raw_detections)
            proc_ms = (time.perf_counter() - t0) * 1000.0

            await websocket.send_json({
                "detections": stabilized,
                "count": len(stabilized),
                "inference_ms": round(proc_ms, 1),
                "frame_width": frame.shape[1],
                "frame_height": frame.shape[0]
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket connection error: {e}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on http://localhost:{port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
