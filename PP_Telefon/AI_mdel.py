import logging as log
import os
from pathlib import Path

import requests
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = Path(os.getenv("PP_TELEFON_DB_DIR", str(BASE_DIR / "db"))).expanduser()
if not DB_DIR.is_absolute():
    DB_DIR = (BASE_DIR / DB_DIR).resolve()
DB_DIR.mkdir(parents=True, exist_ok=True)

# LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1").rstrip("/")
# LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "local-model")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://10.68.98.206:11434/v1").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")


def get_context(query, persist_dir=None):
    results = []
    persist_dir = Path(persist_dir or DB_DIR)
    if not persist_dir.is_absolute():
        persist_dir = (BASE_DIR / persist_dir).resolve()

    try:
        log.info(f"SEARCH: Подключаюсь к базе по пути {persist_dir}")
        db = Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)
        results = db.similarity_search(query, k=3)

        if not results:
            log.warning("SEARCH: Поиск вернул пустой список.")
            return "Информация не найдена."

        context = "\n".join([doc.page_content for doc in results])
        log.info(f"SEARCH: Нашел {len(results)} фрагментов.")
        return context

    except Exception as e:
        log.error(f"SEARCH: Ошибка при поиске: {e}")
        return "Ошибка при обращении к базе знаний."


def process_with_knowledge(transcribed_text):
    log.info("LLM: Ищу данные в базе...")
    context = get_context(transcribed_text)

    prompt = f"Контекст из базы знаний: {context}\n\nВопрос клиента: {transcribed_text}"

    data = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Ты — ВЕЖЛИВЫЙ автоответчик. Используй ТОЛЬКО контекст базы знаний. Если ответа нет в базе, так и скажи. Не задавай вопросов.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }

    try:
        response = requests.post(f"{OLLAMA_URL}/chat/completions", json=data, timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        return f"Ошибка сервера: {response.text}"
    except Exception as e:
        return f"Ошибка связи с Ollama: {e}"