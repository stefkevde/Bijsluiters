"""
Centrale configuratie voor het project.

Alles draait lokaal via Ollama (http://localhost:11434) — er verlaat
geen enkel stukje data deze machine. Dat is bewust: bijsluiter- en
medicatie-gerelateerde vragen kunnen herleidbaar zijn naar een patiënt,
dus een externe API (OpenAI, Anthropic, ...) is hier geen optie.
"""

import os

# Ollama draait standaard lokaal op deze poort
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Klein, quantized embedding-model (draait vlot op een laptop zonder GPU)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

# Klein, quantized chat-model. Llama 3.1 8B in 4-bit (Q4_K_M) is een goede
# balans tussen kwaliteit en snelheid op consumenten-hardware.
# Trek 'm eventueel naar Mistral 7B als je nog lichter wil gaan.
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.1:8b-instruct-q4_K_M")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "embeddings.json")

CHUNK_SIZE = 800       # tekens per chunk
CHUNK_OVERLAP = 100    # overlap tussen chunks, zodat context niet afbreekt
TOP_K = 3              # aantal chunks dat als context meegaat naar het model
