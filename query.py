"""
Stap 2: stel een vraag. Het script zoekt de meest relevante chunks op
(lokaal, via cosine similarity) en stuurt die als context mee naar het
lokale chat-model, met bronvermelding.

Gebruik: python query.py "Wat zijn de contra-indicaties van paracetamol?"
"""

import json
import sys

import numpy as np

from config import INDEX_PATH, TOP_K
from ollama_client import get_embedding, chat, check_connection

SYSTEM_PROMPT = """\
Je bent een interne informatie-assistent voor apotheekmedewerkers.
Beantwoord de vraag UITSLUITEND op basis van de meegegeven fragmenten.
Als het antwoord niet in de fragmenten staat, zeg dat expliciet —
verzin niets bij. Vermeld altijd uit welk document (bron) je antwoord komt.
Dit is een interne hulptool, geen vervanging voor de officiële bijsluiter
of een gesprek met een apotheker/arts."""


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_index(path: str) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Geen index gevonden op {path}. Draai eerst: python ingest.py")
        sys.exit(1)


def top_matches(question_embedding: list[float], index: list[dict], k: int) -> list[dict]:
    q_vec = np.array(question_embedding)
    scored = []
    for entry in index:
        sim = cosine_similarity(q_vec, np.array(entry["embedding"]))
        scored.append((sim, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored[:k]]


def build_prompt(question: str, matches: list[dict]) -> str:
    context_blocks = []
    for m in matches:
        context_blocks.append(f"[Bron: {m['source']}]\n{m['text']}")
    context = "\n\n---\n\n".join(context_blocks)
    return f"Context:\n{context}\n\nVraag: {question}\n\nAntwoord (met bronvermelding):"


def main():
    if len(sys.argv) < 2:
        print('Gebruik: python query.py "je vraag hier"')
        sys.exit(1)

    if not check_connection():
        print(
            "Kan Ollama niet bereiken op localhost. "
            "Start Ollama eerst (zie README.md) en probeer opnieuw."
        )
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    index = load_index(INDEX_PATH)

    print("Vraag embedden en relevante fragmenten opzoeken...")
    question_embedding = get_embedding(question)
    matches = top_matches(question_embedding, index, TOP_K)

    print("Gevonden bronnen:", ", ".join(sorted({m["source"] for m in matches})))
    print("Antwoord genereren...\n")

    prompt = build_prompt(question, matches)
    answer = chat(prompt, system=SYSTEM_PROMPT)

    print("=" * 60)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
