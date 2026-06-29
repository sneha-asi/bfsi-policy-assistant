# Low-Level Design: BFSI Policy Assistant

**Location:** `~/Documents/bfsi-rag`
**Companion:** see `PRD.md` for product scope.

## 1. Architecture overview

```
                          INGEST (run once, static corpus)
  docs/*.pdf
     |  extract.py  (PyMuPDF text; Tesseract OCR fallback for scans)
     v
  pages [{page, text, ocr}]
     |  chunk.py    (sentence-aware, 120-word windows, 30-word overlap)
     v
  chunks [{id, doc_id, title, page, ocr, text}]
     |  embed.py    (Ollama nomic-embed-text, L2-normalized, 768-d)
     v
  vectors (n x 768)  --->  store.py  --->  index/faiss.index + index/chunks.json
                            (FAISS IndexFlatIP)

                          QUERY (per request)
  question (text)  <-----------------------------  voice.py (faster-whisper) <- audio
     |
     |  retrieve.py
     |    dense:  embed query -> FAISS top-12  (cosine via inner product)
     |    sparse: bm25.py top-12               (exact-token match)
     |    fuse:   Reciprocal Rank Fusion -> top-5
     |    gate:   best cosine >= 0.55 ? proceed : ABSTAIN
     v
  generate.py  (Ollama llama3.2:3b, grounded prompt, temp 0.1) + citations
     v
  rag.py -> {answer, grounded, confidence, sources[]}  -> app.py / cli.py
```

## 2. Module responsibilities

| Module | Responsibility | Key choices |
|--------|----------------|-------------|
| `src/extract.py` | PDF -> per-page text | PyMuPDF; if a page has < 40 chars, render at 3x and OCR with Tesseract; keep whichever is longer. Marks `ocr=True`. |
| `src/chunk.py` | text -> overlapping chunks | Sentence split on `.?!`/blank lines; pack to ~120 words; carry a 30-word tail into the next chunk so straddling clauses stay whole. |
| `src/embed.py` | text -> vectors | Ollama `nomic-embed-text`; L2-normalize so inner product == cosine. |
| `src/store.py` | persist + search vectors | FAISS `IndexFlatIP`; saves `faiss.index` + `chunks.json` metadata. |
| `src/bm25.py` | keyword scorer | From-scratch Okapi BM25 (k1=1.5, b=0.75); idf floored at 0. |
| `src/retrieve.py` | hybrid retrieval + abstain | Dense + BM25 fused by RRF (k=60); abstention gate on best cosine (0.55). |
| `src/generate.py` | grounded answer | Ollama `llama3.2:3b`, system prompt restricts to context, temp 0.1, num_ctx 4096, asks for `[Source N]` citations. |
| `src/voice.py` | speech -> text | faster-whisper `tiny.en`, int8 CPU, lazy-loaded and cached. |
| `src/rag.py` | orchestration | retrieve -> abstain-or-generate -> assemble answer + structured sources. |
| `ingest.py` | build index | Run once over `docs/`. |
| `app.py` | web UI | Flask; `/ask` (JSON), `/voice` (multipart audio). |
| `cli.py` | terminal | text input; `:voice <path>` for audio. |

## 3. Data model

**Chunk** (also the FAISS metadata record, stored in `index/chunks.json`):
```json
{
  "id": "health-insurance-policy#1",
  "doc_id": "health-insurance-policy",
  "title": "Health Insurance Policy",
  "page": 1,
  "ocr": false,
  "text": "..."
}
```
The FAISS row index aligns 1:1 with the position in `chunks.json`, so a search
hit's integer id maps directly back to its metadata.

**Answer** (returned by `rag.answer`):
```json
{
  "answer": "string",
  "grounded": true,
  "confidence": 0.692,
  "sources": [{"n": 1, "title": "...", "page": 1, "ocr": false}]
}
```

## 4. Key design decisions and rationale

### 4.1 FAISS `IndexFlatIP` (exact), not an ANN index
The corpus is ~28 chunks and static. Exact flat search is instant, needs no
training, and has zero recall loss. Approximate indexes (IVF, HNSW) only earn
their complexity at large scale and would add recall error here for no benefit.
Inner product on L2-normalized vectors is exactly cosine similarity.

### 4.2 Hybrid retrieval (dense + BM25) fused with RRF
Dense embeddings capture meaning but miss exact tokens that dominate BFSI
queries: amounts, "3 working days", section numbers, acronyms (KYC, AML, EMI,
NEFT). BM25 nails those literal matches. We fuse the two ranked lists with
Reciprocal Rank Fusion: `score(d) = sum 1/(60 + rank_in_list)`. RRF uses only
the rank within each list, so the different score scales of cosine and BM25
cannot dominate each other, and there is nothing to tune per corpus.

### 4.3 Abstention before generation
The product rule is "answer only from the documents". We gate on the best dense
cosine score: if it is below `0.55` the query is treated as out-of-corpus and we
return a fixed refusal **without calling the LLM**. The threshold sits in a clean
gap measured on this corpus: in-corpus best cosine ~0.67-0.76, out-of-corpus
~0.38-0.47. This makes hallucination structurally unlikely rather than relying on
the model to behave.

### 4.4 Grounded generation with low temperature
The system prompt forbids outside knowledge, instructs concise/precise numeric
answers, and asks for `[Source N]` citations. Temperature 0.1 keeps answers
factual and repeatable. The authoritative citation list is the structured
`sources` array built from retrieval, independent of the model's prose.

### 4.5 OCR fallback for scanned PDFs
One document is a scan with no text layer. Rather than dropping it, any page with
< 40 extracted characters is rendered at 3x and passed through Tesseract; the
chunk is flagged `ocr=True` so the UI can show "(OCR)".

### 4.6 Small local models sized for 8GB RAM
`llama3.2:3b` (2GB), `nomic-embed-text` (274MB), and Whisper `tiny.en` (~75MB)
are deliberately chosen to all fit and run on the host CPU with no cloud calls.

## 5. Request flows

**Typed (`POST /ask`):** JSON `{question}` -> `RAG.answer` -> retrieve (dense +
BM25 + RRF) -> abstain check -> generate -> JSON `{answer, grounded, confidence,
sources}`.

**Voice (`POST /voice`):** multipart audio -> saved to a temp file ->
`voice.transcribe` (Whisper) -> transcript treated as the question -> same
pipeline -> JSON also includes the transcribed `question`. In the browser, the
mic uses `MediaRecorder` to capture WebM and uploads the blob.

## 6. Configuration constants

| Constant | Value | Where |
|----------|-------|-------|
| Chunk size / overlap | 120 / 30 words | `chunk.py` |
| OCR trigger | page < 40 chars | `extract.py` |
| OCR render zoom | 3.0x | `extract.py` |
| Embed model | `nomic-embed-text` | `embed.py` |
| Candidates per retriever | 12 | `retrieve.py` |
| RRF constant | 60 | `retrieve.py` |
| Abstention cosine | 0.55 | `retrieve.py` |
| Gen model / temperature | `llama3.2:3b` / 0.1 | `generate.py` |
| Whisper model | `tiny.en` (int8, CPU) | `voice.py` |
| Web port | 5099 | `app.py` |

## 7. External dependencies

- **Ollama** serving locally with `llama3.2:3b` and `nomic-embed-text` pulled.
- System **tesseract** (OCR) and **ffmpeg** (audio decode for Whisper).
- Python venv (Homebrew Python is PEP-668 externally managed).

## 8. Failure modes and handling

| Condition | Behaviour |
|-----------|-----------|
| Out-of-corpus question | Abstain with fixed refusal; LLM never called. |
| Scanned page, no text layer | OCR fallback; chunk flagged `ocr`. |
| Empty question / no audio | 400 from the API. |
| Whisper cannot transcribe | 400 "could not transcribe". |
| Index missing | `store.load` raises; instruct to run `ingest.py`. |

## 9. Known limitations

- The model sometimes appends its own `[Source N]` line in the prose; the
  reliable citations are the structured `sources` list, not that line.
- No streaming output yet; the full answer appears at once.
- Corpus is fixed at ingest time; runtime upload is future work.
