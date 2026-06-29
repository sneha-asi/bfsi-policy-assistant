"""End-to-end RAG: retrieve -> abstain-or-generate -> answer + citations."""
from __future__ import annotations

from pathlib import Path

from .chunk import chunk_document
from .embed import embed_texts
from .extract import extract_pages
from .generate import REFUSAL, generate
from .retrieve import Retriever
from .store import VectorStore


class RAG:
    def __init__(self):
        self.retriever = Retriever(VectorStore.load())

    def add_pdf(self, pdf_path: str | Path, persist: bool = True) -> dict:
        """Ingest a new PDF at runtime: extract -> chunk -> embed -> index."""
        pdf_path = Path(pdf_path)
        doc_id = pdf_path.stem
        title = doc_id.replace("-", " ").replace("_", " ").title()
        pages = extract_pages(pdf_path)
        if not pages:
            return {"title": title, "chunks": 0, "ocr_pages": 0}
        chunks = chunk_document(doc_id, title, pages)
        vectors = embed_texts([c["text"] for c in chunks])
        self.retriever.add_chunks(vectors, chunks)
        if persist:
            self.retriever.store.save()
        return {
            "title": title,
            "chunks": len(chunks),
            "ocr_pages": sum(1 for p in pages if p["ocr"]),
        }

    def answer(self, question: str, top_k: int = 5) -> dict:
        r = self.retriever.retrieve(question, top_k=top_k)
        if not r["grounded"]:
            return {
                "answer": REFUSAL,
                "grounded": False,
                "confidence": round(r["best_cosine"], 3),
                "sources": [],
            }
        text = generate(question, r["hits"])
        sources = [
            {"n": i + 1, "title": h["title"], "page": h["page"], "ocr": h["ocr"]}
            for i, h in enumerate(r["hits"])
        ]
        return {
            "answer": text,
            "grounded": True,
            "confidence": round(r["best_cosine"], 3),
            "sources": sources,
        }
