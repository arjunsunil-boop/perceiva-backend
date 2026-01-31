import whisper
from fastapi import FastAPI,UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from logging import getLogger
from pathlib import Path
import tempfile
import os
logger = getLogger("uvicorn.error")
access_error = getLogger("uvicorn.access")

IMAGES_DIR = Path("Images")
IMAGES_DIR.mkdir(exist_ok = True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

model = whisper.load_model("turbo")

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

        result = model.transcribe(temp_path)
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

