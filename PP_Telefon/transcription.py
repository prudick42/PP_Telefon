import logging as log
import os
import shutil
from pathlib import Path

import whisper

BASE_DIR = Path(__file__).resolve().parent
VOICE_DIR = BASE_DIR / "Voice"
DEFAULT_AUDIO_FILE = VOICE_DIR / "Mehanik.mp3"


def _ensure_ffmpeg():
    ffmpeg_path = os.getenv("FFMPEG_PATH")
    if ffmpeg_path:
        os.environ["FFMPEG_BINARY"] = ffmpeg_path
        return

    local_ffmpeg = BASE_DIR / "ffmpeg.exe"
    if local_ffmpeg.exists():
        os.environ["PATH"] = str(BASE_DIR) + os.pathsep + os.environ.get("PATH", "")
        os.environ["FFMPEG_BINARY"] = str(local_ffmpeg)
    else:
        detected = shutil.which("ffmpeg")
        if detected:
            os.environ["FFMPEG_BINARY"] = detected


_ensure_ffmpeg()


def transcription(audio_file=None):
    try:
        audio_path = Path(audio_file or os.getenv("PP_TELEFON_AUDIO_FILE", str(DEFAULT_AUDIO_FILE))).expanduser()
        if not audio_path.is_absolute():
            audio_path = (BASE_DIR / audio_path).resolve()

        if not audio_path.exists():
            raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")

        model = whisper.load_model("base")
        log.info("Идёт распознавание...")
        result = model.transcribe(str(audio_path))
        result_text = result["text"].strip()

        log.info(f"Распознанный текст: {result_text}")
        return result_text
    except Exception as exc:
        log.exception("Ошибка транскрибации: %s", exc)
        return ""
        