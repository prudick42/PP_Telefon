import os
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_FILE = Path(os.getenv("PP_TELEFON_KNOWLEDGE_FILE", str(BASE_DIR / "knowledge.xlsx"))).expanduser()
DB_DIR = Path(os.getenv("PP_TELEFON_DB_DIR", str(BASE_DIR / "db"))).expanduser()


def build_db_from_excel(file_path=None, persist_dir=None):
    file_path = Path(file_path or KNOWLEDGE_FILE)
    if not file_path.is_absolute():
        file_path = (BASE_DIR / file_path).resolve()

    persist_dir = Path(persist_dir or DB_DIR)
    if not persist_dir.is_absolute():
        persist_dir = (BASE_DIR / persist_dir).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"Файл базы знаний не найден: {file_path}")

    df = pd.read_excel(file_path)

    docs = []
    for _, row in df.iterrows():
        content = (
            f"Сотрудник: {row['Имя']}. "
            f"Должность: {row['Должность']}. "
            f"Телефон: {row['Номер телефона']}. "
        )
        docs.append(Document(page_content=content))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=str(persist_dir))
    print("База знаний обновлена успешно!")


if __name__ == "__main__":
    build_db_from_excel()