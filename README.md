# Lokale AI-assistent voor medicatie-info (demo)

Een klein, volledig **lokaal draaiend** RAG-systeem (Retrieval-Augmented
Generation) dat vragen beantwoordt op basis van een set documenten —
gedacht vanuit een apotheekcontext.

## Waarom lokaal?

In de apotheeksector kan een vraag over medicatie makkelijk herleidbaar zijn
naar een patiënt ("wat mag ik combineren met mijn bloeddrukmedicatie?").
Die data via een externe API (OpenAI, Anthropic, ...) sturen is dan al snel
een GDPR-vraagstuk — zeker met de AI Act erbij, die net extra eisen stelt
aan AI-toepassingen die met gevoelige, medische context werken.

Dit project draait daarom volledig via **[Ollama](https://ollama.com)** op
je eigen machine:
- Geen document, geen vraag en geen antwoord verlaat het toestel.
- Het embedding-model en het taalmodel zijn beide **quantized** (4-bit),
  zodat ze vlot draaien op een gewone laptop zonder GPU.
- De code zelf maakt dit zichtbaar: er zit geen enkele cloud-API-call in
  (`ollama_client.py` praat alleen met `localhost`).

Dit is bewust een **demo/proof-of-concept**, geen productieklare tool. De
voorbeelddocumenten in `data/` zijn fictieve, vereenvoudigde placeholder-
teksten — géén officiële bijsluiters — puur om de pijplijn te tonen.

## Hoe het werkt

1. **`ingest.py`** — leest de `.txt`-bestanden in `data/`, hakt ze in
   overlappende stukken (chunks), en berekent er lokaal een embedding voor
   via Ollama. Alles wordt weggeschreven naar `embeddings.json` (een simpele
   lokale "vectorstore" — geen aparte databank nodig voor deze schaal).
2. **`query.py`** — embedt je vraag, zoekt via cosine similarity de meest
   relevante chunks op, en stuurt die als context naar het lokale
   taalmodel, met de instructie om **altijd de bron te vermelden** en
   **niets te verzinnen** als het antwoord niet in de documenten staat.

Dit is bewust *niet* meteen naar een taalmodel gooien: RAG is hier zinvol
omdat je wil dat antwoorden **herleidbaar** zijn naar een concreet document
— cruciaal in een medische/apotheekcontext.

## Installatie

```bash
# 1. Ollama installeren (macOS/Linux/Windows): zie https://ollama.com/download

# 2. De twee modellen ophalen (worden lokaal quantized bewaard)
ollama pull nomic-embed-text
ollama pull llama3.1:8b-instruct-q4_K_M

# 3. Python-dependencies
pip install -r requirements.txt
```

## Gebruik

```bash
# Stap 1: documenten inlezen en embedden (eenmalig, of opnieuw bij wijzigingen)
python ingest.py

# Stap 2: een vraag stellen
python query.py "Waar moet ik op letten bij paracetamol?"
```

Voorbeelduitvoer:

```
Vraag embedden en relevante fragmenten opzoeken...
Gevonden bronnen: paracetamol_info.txt
Antwoord genereren...

============================================================
Volgens paracetamol_info.txt is extra voorzichtigheid aangewezen bij
leverproblemen, en wordt gelijktijdig gebruik met andere paracetamol-
bevattende producten afgeraden. Voor de volledige, actuele informatie
verwijst het document zelf door naar het BCFI of de officiële bijsluiter.
============================================================
```

## Volgende stappen (als dit verder zou gaan dan een demo)

- Chunking vervangen door iets robuuster dan woord-tellen (bv. op zinsgrens).
- `embeddings.json` vervangen door een echte lokale vectordatabank
  (bv. Chroma of SQLite + sqlite-vec) zodra het aantal documenten groeit.
- Tracing/logging toevoegen (welke chunks werden gebruikt, hoe lang het
  duurde, hoeveel tokens) — observability vanaf dag één, zoals in de
  functie-eisen van deze vacature ook expliciet gevraagd wordt.
- Een echte, geautoriseerde databron aankoppelen (bv. BCFI) in plaats van
  losse `.txt`-bestanden.

## Disclaimer

De inhoud in `data/` is fictief en enkel bedoeld om de technische pijplijn
te demonstreren. Dit is geen medisch advies en geen vervanging voor de
officiële bijsluiter, het BCFI, of een gesprek met een apotheker of arts.
