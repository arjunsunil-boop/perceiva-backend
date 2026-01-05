import whisper
from fastapi import FastAPI
from logging import getLogger

logger = getLogger("uvicorn.error")
access_error = getLogger("uvicorn.access")

app = FastAPI()
model = whisper.load_model("turbo")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/stt")
async def stt(file_name : str):
    logger.info("Processing audio file:")
    result = model.transcribe(file_name)
    logger.info(result["text"])
    return {"text":result["text"]}

