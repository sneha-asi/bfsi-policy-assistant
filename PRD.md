# Product Requirements Document: BFSI Policy Assistant

**Status:** Built and verified (2026-06-28)
**Owner:** Sneha Gupta

## 1. Summary

BFSI Policy Assistant is a fully local retrieval-augmented (RAG) chatbot that
answers customer and staff questions about a bank/insurer's policy and
terms-and-conditions documents. It answers **only** from the loaded documents,
cites its sources, and refuses questions it cannot ground in the corpus.
Questions can be typed or spoken.

## 2. Problem

BFSI policy and T&C documents are long, dense, and rarely change ("stagnant").
Customers and frontline staff need exact answers (waiting periods, fees,
reporting windows, minimum balances) without reading the whole document or
risking a wrong answer. A general chatbot would hallucinate; a keyword search
returns pages, not answers. The product needs a grounded assistant that is
accurate, cites its source, and stays inside the document set.

## 3. Goals

- Answer natural-language questions strictly from a fixed set of BFSI PDFs.
- Always show which document and page an answer came from (auditability).
- Refuse (abstain) when the answer is not in the documents, instead of guessing.
- Support both typed and voice questions.
- Run entirely locally: no API keys, no customer data leaving the machine
  (important for BFSI data-protection requirements).

## 4. Non-goals

- No live/changing data feeds. The corpus is treated as static and indexed once.
- No multi-tenant accounts, auth, or persistence of user chats in v1.
- No fine-tuning of models; we use off-the-shelf local models.
- No general-knowledge Q&A; out-of-corpus questions are intentionally refused.

## 5. Users and use cases

- **Customer / self-service:** "What is the waiting period for pre-existing
  diseases?", "How long do I have to report a fraudulent transaction?"
- **Frontline support agent:** quick, cited lookups during a call.
- **Compliance / ops:** confirm exact policy wording with a page reference.

## 6. Corpus

12 BFSI policy / T&C PDFs (health, motor, life insurance; home loan; savings and
fixed-deposit terms; credit card agreement; KYC/AML; claims; grievance; privacy;
and one **scanned** fraud-dispute policy). The scanned document forces an OCR
path so image-only PDFs are not silently dropped. (These are clearly-labelled
synthetic "Aegis" demo documents, not real customer data.)

## 7. Functional requirements

| # | Requirement |
|---|-------------|
| F1 | Ingest a folder of PDFs (including scanned ones via OCR) into a searchable index. |
| F2 | Answer a typed question from the indexed documents. |
| F3 | Answer a spoken question (microphone or audio file) via speech-to-text. |
| F4 | Show source document + page for every grounded answer. |
| F5 | Abstain with a clear message when no relevant content is found. |
| F6 | Surface a confidence signal for each answer. |
| F7 | Provide a web UI and a CLI. |

## 8. Non-functional requirements

- **Privacy:** 100% local inference (embeddings, generation, transcription).
- **Hardware fit:** must run on an 8GB-RAM Mac (drives small-model choices).
- **Accuracy bias:** prefer abstaining over a wrong answer (precision over recall).
- **Latency:** typed answers within a few seconds on CPU for this corpus size.
- **Reproducibility:** deterministic-leaning generation (low temperature).

## 9. Success criteria

- In-corpus factual questions answered correctly with a correct citation.
- Out-of-corpus questions reliably refused (no hallucinated answers).
- Voice questions transcribed accurately and routed through the same pipeline.

Verified on 2026-06-28: correct cited answers (36-month pre-existing-disease
wait, 3-working-day fraud reporting window, Rs 10,000 / 5,000 minimum balance,
1% FD premature-withdrawal penalty); out-of-corpus questions abstained; a
speech sample was transcribed exactly and answered.

## 10. Stack (summary)

Local only: FAISS (vector index), Ollama (`nomic-embed-text` embeddings +
`llama3.2:3b` generation), faster-whisper (`tiny.en` voice), PyMuPDF + Tesseract
(extraction/OCR), Flask (web UI). See `LLD.md` for the design.

## 11. Future work

- Streaming token output in the web UI.
- In-browser document upload to extend the corpus at runtime.
- An evaluation harness with ground-truth Q&A and adversarial near-miss queries.
- Answer-citation cleanup (strip the model's inline source line; rely on the
  structured sources list only).
