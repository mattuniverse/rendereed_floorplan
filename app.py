"""
FloorPlan Studio — Furniture Detection Inference API
-----------------------------------------------------
Loads a YOLO11n model (best.pt) once at startup and exposes:
  GET  /health   -> simple liveness check
  POST /predict  -> accepts an image file, returns JSON detections

Environment variables:
  MODEL_PATH        Local path to the model file (default: best.pt)
  MODEL_URL         Optional direct-download URL to fetch the model from
                     if it isn't already present at MODEL_PATH
                     (e.g. a Google Drive "uc?export=download&id=..." link,
                     or a Hugging Face / S3 / GitHub Release URL)
  CONF_THRESHOLD    Confidence threshold, default 0.25
  ALLOWED_ORIGINS   Comma-separated list of allowed origins for CORS,
                     e.g. "https://your-app.vercel.app,http://localhost:3000"
                     Defaults to "*" (open) if not set.
"""

import os
import io
import logging
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("floorplan-inference")

MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
MODEL_URL = os.environ.get("MODEL_URL")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")

model_holder = {"model": None}


def download_model_if_needed():
    if os.path.exists(MODEL_PATH):
        logger.info(f"Model already present at {MODEL_PATH}")
        return
    if not MODEL_URL:
        raise RuntimeError(
            f"Model not found at {MODEL_PATH} and MODEL_URL is not set. "
            "Either bake best.pt into the image/repo, or set MODEL_URL "
            "to a direct-download link."
        )
    logger.info(f"Downloading model from MODEL_URL to {MODEL_PATH} ...")
    response = requests.get(MODEL_URL, stream=True, timeout=120)
    response.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info("Model download complete.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Import here so the module import itself is cheap (helps cold starts
    # on platforms that inspect the app before fully starting it).
    from ultralytics import YOLO

    download_model_if_needed()
    logger.info("Loading YOLO11n model ...")
    model_holder["model"] = YOLO(MODEL_PATH)
    logger.info("Model loaded. Ready to serve predictions.")
    yield
    model_holder["model"] = None


app = FastAPI(title="FloorPlan Studio Inference API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_holder["model"] is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model = model_holder["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded image")

    results = model.predict(image, conf=CONF_THRESHOLD, verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = result.names.get(cls_id, str(cls_id))
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        detections.append({
            "label": label,
            "confidence": round(confidence, 4),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })

    return JSONResponse({
        "count": len(detections),
        "image_size": {"width": image.width, "height": image.height},
        "detections": detections,
    })


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
