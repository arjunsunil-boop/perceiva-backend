import whisper
from fastapi import FastAPI,UploadFile, File, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from logging import getLogger
from pathlib import Path
import tempfile
import os
import io
from kokoro import KPipeline
import soundfile as sf
import numpy as np
from ultralytics import YOLO
import cv2
import uuid





app = FastAPI()

logger = getLogger("uvicorn.error")
access_error = getLogger("uvicorn.access")

IMAGES_DIR = Path("Images")
IMAGES_DIR.mkdir(exist_ok = True)

MODEL_PATH = r"models\best.pt"
CROP_DIR = Path("assets/croppedImages")
UPLOAD_DIR = Path("assets/uploads")
CROP_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PADDING_RATIO = 0.15

pipeline = KPipeline(lang_code="a")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

whisper_model = whisper.load_model("turbo")
yolo_model = YOLO(MODEL_PATH)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/stt-upload")
async def stt_upload(file: UploadFile = File(...)):
    temp_path = None
    try:
        logger.info(f"Processing uploaded audio file: {file.filename}")

        suffix = Path(file.filename).suffix if file.filename else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_path = tmp.name

        result = whisper_model.transcribe(temp_path)
        text = result.get("text", "")
        logger.info(text)
        return {"text": text}
    except Exception as e:
        logger.error(f"Error processing uploaded audio: {str(e)}")
        return {"text": "Error", "error": str(e)}
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except Exception:
                logger.warning(f"Failed to delete temp file: {temp_path}")

@app.post("/tts")
def tts(text: str = Body(..., embed = True)):
    generator = pipeline(text, voice="af_heart")

    chunks = []
    for _, _, audio in generator:
        chunks.append(audio)

    if not chunks:
        return {"ok": False, "error": "No audio generated"}

    full_audio = np.concatenate(chunks)
    buffer = io.BytesIO()
    sf.write(buffer, full_audio, 24000, format="WAV")
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=tts.wav"}
    )


@app.post("/detect-crop")
async def detect_crop(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix or ".jpg"
    upload_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    data = await file.read()
    upload_path.write_bytes(data)

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return {"ok": False, "error": "Invalid image"}

    results = yolo_model.predict(source=image, conf=0.25)

    crop_count = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = result.names[class_id]

            img_h, img_w = image.shape[:2]
            box_w = max(0, x2 - x1)
            box_h = max(0, y2 - y1)
            pad_x = int(box_w * PADDING_RATIO)
            pad_y = int(box_h * PADDING_RATIO)

            x1p = max(0, x1 - pad_x)
            y1p = max(0, y1 - pad_y)
            x2p = min(img_w, x2 + pad_x)
            y2p = min(img_h, y2 + pad_y)

            if x2p <= x1p or y2p <= y1p:
                continue

            cropped = image[y1p:y2p, x1p:x2p]
            crop_filename = f"{class_name}_{crop_count}_{confidence:.2f}.jpg"
            crop_path = CROP_DIR / crop_filename
            cv2.imwrite(str(crop_path), cropped)
            crop_count += 1

    return {"ok": True, "crops": crop_count, "saved_to": str(CROP_DIR)}