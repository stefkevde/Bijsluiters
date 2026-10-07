# Local AI Assistant for Medication Information (Demo)

A small, fully **local RAG system** (Retrieval-Augmented Generation) that answers questions based on a set of documents — designed with a pharmacy context in mind.

## Why local?

In the pharmacy sector, a question about medication can easily be linked to a patient ("what can I combine with my blood pressure medication?"). Sending that data through an external API (OpenAI, Anthropic, ...) can quickly raise GDPR concerns — especially when considering the AI Act, which introduces additional requirements for AI applications dealing with sensitive medical contexts.

This project therefore runs entirely through [**Ollama**](https://ollama.com) on your own machine:

- No document, question, or answer leaves the device.
- Both the embedding model and the language model are **quantized** (4-bit), allowing them to run smoothly on a regular laptop without a GPU.
- The code itself makes this explicit: there are no cloud API calls (`ollama_client.py` only communicates with `localhost`).

This is deliberately a **demo/proof of concept**, not a production-ready tool. The example documents in `data/` are fictional, simplified placeholder texts — **not official patient leaflets** — and are solely intended to demonstrate the pipeline.

## How it works

1. **`ingest.py`** — reads the `.txt` files from `data/`, splits them into overlapping chunks, and calculates a local embedding for each chunk through Ollama. Everything is stored in `embeddings.json` (a simple local "vector store" — no separate database is required at this scale).
2. **`query.py`** — embeds the user's question, searches for the most relevant chunks using cosine similarity, and sends them as context to the local language model, with instructions to **always cite the source** and **not make anything up** if the answer is not contained in the documents.

This deliberately does *not* immediately send the question to a language model: RAG makes sense here because answers should be **traceable** to a specific document — crucial in a medical/pharmacy context.

## Installation

```bash
# 1. Install Ollama (macOS/Linux/Windows): see https://ollama.com/download

# 2. Pull the two models (stored locally in quantized form)
ollama pull nomic-embed-text
ollama pull llama3.1:8b-instruct-q4_K_M

# 3. Install Python dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Step 1: Ingest and embed the documents (once, or again when documents change)
python ingest.py

# Step 2: Ask a question
python query.py "What should I be aware of when taking paracetamol?"
```

Example output:

```markdown
Embedding the question and searching for relevant passages...
Sources found: paracetamol_info.txt
Generating answer...

============================================================
According to paracetamol_info.txt, extra caution is advised
for people with liver problems, and simultaneous use with
other paracetamol-containing products is discouraged. For
complete and up-to-date information, the document itself
refers to the BCFI or the official patient leaflet.
============================================================
```

## Next steps (if this were taken beyond a demo)

- Replace word-count-based chunking with something more robust, such as sentence-based chunking.
- Replace `embeddings.json` with a proper local vector database (e.g. Chroma or SQLite + sqlite-vec) once the number of documents grows.
- Add tracing/logging (which chunks were used, how long processing took, how many tokens were used) — implementing observability from day one, as explicitly mentioned in the requirements for this position.
- Connect an actual authorized data source (e.g. BCFI) instead of separate `.txt` files.

## Disclaimer

The content in `data/` is fictional and intended solely to demonstrate the technical pipeline. This is **not medical advice** and is not a substitute for the official patient leaflet, BCFI, or consultation with a pharmacist or physician.
