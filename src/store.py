"""FAISS vector store + chunk metadata, persisted to disk.

The corpus is small and static ("stagnant" policy/T&C docs), so an exact
IndexFlatIP is the right call: zero approximation error, no training, instant
build. With L2-normalized vectors inner product == cosine similarity.
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

INDEX_DIR = Path(__file__).resolve().parent.parent / "index"
_FAISS_PATH = INDEX_DIR / "faiss.index"
_META_PATH = INDEX_DIR / "chunks.json"


class VectorStore:
    def __init__(self, index: faiss.Index, chunks: list[dict]):
        self.index = index
        self.chunks = chunks

    @classmethod
    def build(cls, vectors: np.ndarray, chunks: list[dict]) -> "VectorStore":
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        return cls(index, chunks)

    def add(self, vectors: np.ndarray, new_chunks: list[dict]) -> None:
        """Append more vectors + their metadata (used by live uploads)."""
        self.index.add(vectors.astype("float32"))
        self.chunks.extend(new_chunks)

    def save(self) -> None:
        INDEX_DIR.mkdir(exist_ok=True)
        faiss.write_index(self.index, str(_FAISS_PATH))
        _META_PATH.write_text(json.dumps(self.chunks, ensure_ascii=False))

    @classmethod
    def load(cls) -> "VectorStore":
        if not _FAISS_PATH.exists():
            raise FileNotFoundError("Index not found. Run ingest.py first.")
        index = faiss.read_index(str(_FAISS_PATH))
        chunks = json.loads(_META_PATH.read_text())
        return cls(index, chunks)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[dict, float]]:
        q = query_vec.reshape(1, -1).astype("float32")
        scores, idxs = self.index.search(q, min(k, len(self.chunks)))
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            out.append((self.chunks[idx], float(score)))
        return out
