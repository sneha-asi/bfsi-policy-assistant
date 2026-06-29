# BFSI Policy Assistant — local RAG (FAISS + Ollama + Whisper)

A retrieval-augmented chatbot that answers **only** from a static set of BFSI
policy and terms-and-conditions PDFs. Everything runs locally: no API keys, no
data leaves the machine.

## Stack
- **Extraction:** PyMuPDF, with a **Tesseract OCR fallback** for scanned pages
  (one of the 12 docs is a scanned fraud-dispute policy).
- **Chunking:** sentence-aware, ~120-word windows with 30-word overlap so clauses
  stay whole.
- **Embeddings:** Ollama `nomic-embed-text` (768-d), L2-normalized.
- **Vector store:** **FAISS** `IndexFlatIP` (exact cosine — the corpus is small and
  static, so no approximation needed).
- **Retrieval:** hybrid **dense (FAISS) + BM25**, fused with Reciprocal Rank Fusion.
- **Abstention:** gate on the best dense cosine score; out-of-corpus questions are
  refused instead of hallucinated.
- **Generation:** Ollama `llama3.2:3b`, grounded prompt, low temperature, citations.
- **Voice:** `faster-whisper` (`tiny.en`) for mic / audio-file questions.
- **UI:** Flask web chat (text + mic) and a CLI.

## Why these choices
- **FAISS Flat over an ANN index:** ~28 chunks; exact search is instant and has
  zero recall loss. ANN (IVF/HNSW) only pays off at large scale.
- **Hybrid + RRF:** dense embeddings miss exact tokens that matter in BFSI
  (amounts, "3 working days", KYC/AML/EMI); BM25 catches them. RRF combines the
  two by rank, so neither score scale dominates and there's nothing to tune.
- **Abstention before generation:** the requirement is "answer only from the
  docs". The cosine gate (0.55) cleanly separates in-corpus (~0.67–0.76) from
  out-of-corpus (~0.38–0.47), so we never invoke the LLM on irrelevant queries.
- **Small local models:** the host has 8GB RAM, so `llama3.2:3b` + `nomic-embed-text`
  + Whisper `tiny.en` are deliberately sized to fit.

## Run
```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
# needs: ollama serving, `ollama pull llama3.2:3b nomic-embed-text`,
#        system `tesseract` and `ffmpeg`
./.venv/bin/python ingest.py        # build the FAISS index (run once)
./.venv/bin/python app.py           # web UI at http://127.0.0.1:5099
./.venv/bin/python cli.py           # terminal chat
./.venv/bin/python smoke_test.py    # grounded answers + abstention check
```

## Layout
```
docs/          12 BFSI policy / T&C PDFs (the static corpus)
index/         FAISS index + chunk metadata (built by ingest.py)
src/extract.py extraction + OCR fallback
src/chunk.py   sentence-aware chunking
src/embed.py   Ollama embeddings
src/store.py   FAISS wrapper (build/save/load/search)
src/bm25.py    from-scratch Okapi BM25
src/retrieve.py hybrid retrieval + RRF + abstention
src/generate.py grounded Ollama generation
src/voice.py   Whisper transcription
src/rag.py     end-to-end orchestration
app.py / cli.py interfaces
```
