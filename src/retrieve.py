"""Hybrid retrieval: dense (FAISS) + BM25, fused with Reciprocal Rank Fusion.

Why RRF instead of adding the raw scores: cosine and BM25 live on different
scales, so summing them lets one dominate. RRF only uses each list's *rank*, so
the two signals combine fairly and it needs no per-corpus tuning.

Abstention: the chatbot must answer ONLY from the docs. We gate on the best
dense cosine score; if nothing clears the threshold the query is out-of-corpus
and we refuse instead of letting the LLM improvise.
"""
from __future__ import annotations

from .bm25 import BM25
from .embed import embed_query
from .store import VectorStore

# Calibrated for the nomic-embed-text BFSI distribution (see calibrate.py).
ABSTAIN_COSINE = 0.55
RRF_K = 60          # standard RRF damping constant
DENSE_K = 12        # candidates pulled from each retriever before fusion


class Retriever:
    def __init__(self, store: VectorStore):
        self.store = store
        self.bm25 = BM25([c["text"] for c in store.chunks])

    def add_chunks(self, vectors, new_chunks: list[dict]) -> None:
        """Add new chunks to the store and rebuild BM25 (cheap for this corpus)."""
        self.store.add(vectors, new_chunks)
        self.bm25 = BM25([c["text"] for c in self.store.chunks])

    def retrieve(self, query: str, top_k: int = 5) -> dict:
        # --- dense ---
        qvec = embed_query(query)
        dense = self.store.search(qvec, DENSE_K)
        best_cosine = dense[0][1] if dense else 0.0
        dense_ranked = [c["id"] for c, _ in dense]

        # --- sparse ---
        bm_scores = self.bm25.scores(query)
        bm_ranked = [
            self.store.chunks[i]["id"]
            for i in sorted(range(len(bm_scores)), key=lambda j: bm_scores[j], reverse=True)[:DENSE_K]
            if bm_scores[i] > 0
        ]

        # --- reciprocal rank fusion ---
        fused: dict[str, float] = {}
        for rank, cid in enumerate(dense_ranked):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        for rank, cid in enumerate(bm_ranked):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)

        by_id = {c["id"]: c for c in self.store.chunks}
        order = sorted(fused, key=fused.get, reverse=True)[:top_k]
        hits = [by_id[cid] for cid in order]

        return {
            "hits": hits,
            "best_cosine": best_cosine,
            "grounded": best_cosine >= ABSTAIN_COSINE,
        }
