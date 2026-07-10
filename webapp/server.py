"""
PlastiScope — microplastic detection web service.

Serves the custom-trained YOLO26 model (ONNX, NMS-free end-to-end) over a
small FastAPI app. The model outputs (1, 300, 6) = [x1, y1, x2, y2, score,
class] in letterboxed 640x640 space; we decode back to original image pixels.

Run:  uvicorn server:app --host 127.0.0.1 --port 8000   (from webapp/)
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

# ─── Configuration ────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT.parent / "models" / "microplastic_yolo26.onnx"
STATIC_DIR = ROOT / "static"

IMG_SIZE = 640              # model input side
CONF_FLOOR = 0.05           # server-side floor; client slider refines above this
MAX_UPLOAD_MB = 25

CLASS_NAMES = ["fiber", "film", "fragment", "pellet"]

# ─── Model session (loaded once at startup) ───────────────────────────────────

session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name

# Warm up so the first user request isn't slow.
session.run(None, {INPUT_NAME: np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)})


# ─── Pre/post-processing ──────────────────────────────────────────────────────

def letterbox(img: Image.Image, size: int = IMG_SIZE):
    """Resize keeping aspect ratio, pad to size x size (Ultralytics-style).

    Returns (tensor NCHW float32/255, scale, pad_x, pad_y).
    """
    w0, h0 = img.size
    scale = min(size / w0, size / h0)
    new_w, new_h = round(w0 * scale), round(h0 * scale)
    resized = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = arr.transpose(2, 0, 1)[None]  # HWC -> NCHW
    return np.ascontiguousarray(tensor), scale, pad_x, pad_y


def decode(raw: np.ndarray, scale: float, pad_x: int, pad_y: int,
           w0: int, h0: int) -> list[dict]:
    """(300, 6) letterboxed [x1,y1,x2,y2,conf,cls] -> original-pixel detections."""
    detections = []
    for x1, y1, x2, y2, conf, cls in raw:
        if conf < CONF_FLOOR:
            continue
        bx1 = max(0.0, min((x1 - pad_x) / scale, w0))
        by1 = max(0.0, min((y1 - pad_y) / scale, h0))
        bx2 = max(0.0, min((x2 - pad_x) / scale, w0))
        by2 = max(0.0, min((y2 - pad_y) / scale, h0))
        if bx2 - bx1 < 1 or by2 - by1 < 1:
            continue
        ci = int(cls)
        detections.append({
            "class_id": ci,
            "class_name": CLASS_NAMES[ci] if 0 <= ci < len(CLASS_NAMES) else str(ci),
            "confidence": round(float(conf), 4),
            "box": [round(float(v), 1) for v in (bx1, by1, bx2, by2)],
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


# ─── API ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="PlastiScope API", docs_url="/api/docs", openapi_url="/api/openapi.json")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH.name,
        "classes": CLASS_NAMES,
        "input_size": IMG_SIZE,
        "conf_floor": CONF_FLOOR,
    }


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Image exceeds {MAX_UPLOAD_MB} MB limit.")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        raise HTTPException(400, "File is not a readable image. Use JPG, PNG, TIFF, BMP or WEBP.")

    w0, h0 = img.size
    t0 = time.perf_counter()
    tensor, scale, pad_x, pad_y = letterbox(img)
    raw = session.run(None, {INPUT_NAME: tensor})[0][0]  # (300, 6)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    detections = decode(raw, scale, pad_x, pad_y, w0, h0)
    counts = {name: 0 for name in CLASS_NAMES}
    for d in detections:
        if d["confidence"] >= 0.25:  # headline counts at default threshold
            counts[d["class_name"]] += 1

    return {
        "detections": detections,
        "counts_at_default": counts,
        "image": {"width": w0, "height": h0, "filename": file.filename},
        "inference_ms": round(elapsed_ms, 1),
    }


# Static frontend — mounted last so /api/* wins.
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
