"""Build the FAISS index from the PDFs in docs/.

Run once (the corpus is static): extract -> chunk -> embed -> FAISS -> save.
"""
from __future__ import annotations

from pathlib import Path

from src.chunk import chunk_document
from src.embed import embed_texts
from src.extract import extract_pages
from src.store import VectorStore

DOCS_DIR = Path(__file__).resolve().parent / "docs"


def _title(path: Path) -> str:
    return path.stem.replace("-", " ").replace("_", " ").title()


def main() -> None:
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs in {DOCS_DIR}")

    all_chunks: list[dict] = []
    for pdf in pdfs:
        pages = extract_pages(pdf)
        ocr_pages = sum(1 for p in pages if p["ocr"])
        chunks = chunk_document(pdf.stem, _title(pdf), pages)
        all_chunks.extend(chunks)
        flag = f"  (OCR on {ocr_pages} page(s))" if ocr_pages else ""
        print(f"  {pdf.name}: {len(pages)} pages -> {len(chunks)} chunks{flag}")

    print(f"\nEmbedding {len(all_chunks)} chunks with Ollama nomic-embed-text...")
    vectors = embed_texts([c["text"] for c in all_chunks])

    store = VectorStore.build(vectors, all_chunks)
    store.save()
    print(f"Done. FAISS index: {store.index.ntotal} vectors, dim {vectors.shape[1]}.")


if __name__ == "__main__":
    main()
