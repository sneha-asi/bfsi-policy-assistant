"""Embeddings via local Ollama (nomic-embed-text).

nomic-embed-text is a 274MB model that runs comfortably in 8GB RAM. We L2-normalize
every vector so that an inner-product FAISS index gives exact cosine similarity.
"""
from __future__ import annotations

import numpy as np
import ollama

EMBED_MODEL = "nomic-embed-text"


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts -> (n, dim) float32, L2-normalized."""
    out = []
    for t in texts:
        resp = ollama.embeddings(model=EMBED_MODEL, prompt=t)
        out.append(resp["embedding"])
    arr = np.asarray(out, dtype="float32")
    return _normalize(arr)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query -> (dim,) float32, L2-normalized."""
    return embed_texts([text])[0]
