"""
Dunne wrapper rond de lokale Ollama HTTP-API.

Geen SDK's, geen cloud-calls — puur requests naar localhost. Zo is meteen
zichtbaar in de code zelf dat er niets naar buiten gaat.
"""

import requests

from config import OLLAMA_URL, EMBED_MODEL, CHAT_MODEL


def get_embedding(text: str) -> list[float]:
    """Vraag een embedding op bij het lokale embedding-model."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def chat(prompt: str, system: str | None = None) -> str:
    """Stuur een prompt naar het lokale chat-model en geef het antwoord terug.

    We gebruiken stream=False voor de eenvoud (dit is een demo-CLI, geen
    productie-app), en temperature=0 omdat we hier feitelijke, herleidbare
    antwoorden willen — geen creatieve variatie.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def check_connection() -> bool:
    """Simpele check of Ollama lokaal bereikbaar is, voor duidelijke foutmeldingen."""
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return True
    except requests.exceptions.RequestException:
        return False
