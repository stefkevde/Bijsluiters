"""
Stap 1: lees de documenten in data/, hak ze in chunks, en bereken er
lokaal embeddings voor via Ollama. Het resultaat wordt weggeschreven
naar embeddings.json — een simpele lokale "vectorstore".

Gebruik: python ingest.py
"""

import glob
import json
import os
import sys

from config import DATA_DIR, INDEX_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from ollama_client import get_embedding, check_connection


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Knip tekst in overlappende stukken, op woordgrenzen."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk_words = []
        char_count = 0
        i = start
        while i < len(words) and char_count < size:
            chunk_words.append(words[i])
            char_count += len(words[i]) + 1
            i += 1
        chunks.append(" ".join(chunk_words))
        if i >= len(words):
            break
        # ga terug voor de overlap, zodat context niet hard afbreekt
        overlap_words = max(1, overlap // 6)  # ruwe schatting: ~6 tekens/woord
        start = max(start + 1, i - overlap_words)
    return chunks


def load_documents(data_dir: str) -> list[tuple[str, str]]:
    """Geeft een lijst van (bestandsnaam, inhoud) terug voor alle .txt bestanden."""
    paths = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not paths:
        print(f"Geen .txt bestanden gevonden in {data_dir}")
        sys.exit(1)
    docs = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read()))
    return docs


def main():
    if not check_connection():
        print(
            "Kan Ollama niet bereiken op localhost. "
            "Start Ollama eerst (zie README.md) en probeer opnieuw."
        )
        sys.exit(1)

    documents = load_documents(DATA_DIR)
    print(f"{len(documents)} document(en) gevonden. Chunken en embedden...")

    index = []
    for filename, text in documents:
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            print(f"  - {filename} chunk {i + 1}/{len(chunks)}")
            embedding = get_embedding(chunk)
            index.append(
                {
                    "source": filename,
                    "chunk_id": i,
                    "text": chunk,
                    "embedding": embedding,
                }
            )

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f)

    print(f"\nKlaar. {len(index)} chunks opgeslagen in {INDEX_PATH}")


if __name__ == "__main__":
    main()
