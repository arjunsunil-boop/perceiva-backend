import whisper
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = whisper.load_model("turbo")
result = model.transcribe("C:/Users/ARJUN/Documents/Sound Recordings/Recording (4).m4a")
logger.info(result["text"])  