import logging as log
import os
from pathlib import Path

import torch
import soundfile as sf

from omnivoice import OmniVoice

from AI_mdel import process_with_knowledge
from transcription import transcription

BASE_DIR = Path(__file__).resolve().parent

log.basicConfig(
    filename=str(BASE_DIR / "logis.log"),
    level=log.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    encoding="UTF-8",
)


def _ensure_ffmpeg():
    local_ffmpeg = BASE_DIR / "ffmpeg.exe"
    if local_ffmpeg.exists():
        os.environ["PATH"] = str(BASE_DIR) + os.pathsep + os.environ.get("PATH", "")
        os.environ["FFMPEG_BINARY"] = str(local_ffmpeg)




DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

log.info(f"Using device: {DEVICE}")


log.info("Loading OmniVoice model...")

tts_model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map=DEVICE,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)

log.info("OmniVoice loaded successfully.")


VOICE_PROMPT = {
    "language": "ru",
    "gender": "female",
    "style": "natural",
    "age": "young_adult",
    "emotion": "neutral",
}

OUTPUT_AUDIO = os.getenv(
    "PP_TELEFON_OUTPUT_AUDIO_FILE",
    str(BASE_DIR / "output.wav")
)


def generate_audio(text, output_filename=None):
    try:
        if not text or not str(text).strip():
            log.warning("TTS: empty text")
            return None

        output_path = Path(output_filename or OUTPUT_AUDIO).expanduser()

        if not output_path.is_absolute():
            output_path = (BASE_DIR / output_path).resolve()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        log.info(f"TTS: generating audio ({len(text)} chars)")

        # audio = tts_model.generate(
        #     text=str(text),
        #     voice=VOICE_PROMPT
        # )

        REFERENCE_AUDIO = str(BASE_DIR / "Voice" / "Record.wav")
        REFERENCE_TEXT = ("Прокуроры начали масштабное расследование обвинений в договорных матчах и незаконных ставках. Разные конструкции телескопов работают по-разному и имеют свои сильные и слабые стороны.Мы можем продолжать улучшать подготовку хороших юристов.Обратная связь должна быть своевременной и точной на протяжении всего проекта.Люди также оценивают расстояние, используя относительные размеры объектов.Церкви не должны поощрять это или представлять это как что-то безобидное.Узнайте о настройке беспроводной сети.Вы можете есть их свежими, вареными или ферментированными.")
        audio = tts_model.generate(
            text=str(text),
            ref_audio=REFERENCE_AUDIO,
            ref_text=REFERENCE_TEXT,
            num_step=16,
            speed=1.05
        )


        sf.write(
            str(output_path),
            audio[0],
            24000
        )

        log.info(f"TTS saved: {output_path}")

        return str(output_path)

    except Exception as e:
        log.error(f"TTS error: {e}")
        return None



def main():
    text = transcription()

    llm_resp = process_with_knowledge(text)

    print(llm_resp)

    output_file = generate_audio(llm_resp)

    print(output_file)


if __name__ == "__main__":
    main()