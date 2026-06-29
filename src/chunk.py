"""Sentence-aware chunking.

Policy/T&C text is dense with clauses, so naive fixed-width splits cut sentences
in half and hurt retrieval. We split on sentence boundaries and pack sentences
into ~chunk_size-word windows with a small overlap so a clause that straddles a
window boundary still appears whole in one chunk.
"""
from __future__ import annotations

import re

# Split on ., !, ? or newline followed by whitespace/end. Keeps abbreviations
# mostly intact for this corpus while staying dependency-free.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")

CHUNK_WORDS = 120      # target words per chunk
OVERLAP_WORDS = 30     # words carried over between consecutive chunks


def _sentences(text: str) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text)
    parts = [s.strip() for s in _SENT_SPLIT.split(text)]
    return [s for s in parts if s]


def chunk_document(doc_id: str, title: str, pages: list[dict]) -> list[dict]:
    """Turn extracted pages into overlapping, sentence-aligned chunks."""
    chunks: list[dict] = []
    for pg in pages:
        sents = _sentences(pg["text"])
        cur: list[str] = []
        cur_len = 0
        for sent in sents:
            wlen = len(sent.split())
            if cur and cur_len + wlen > CHUNK_WORDS:
                chunks.append(_make(doc_id, title, pg["page"], pg["ocr"], cur, len(chunks)))
                # start next chunk with the tail of the previous one (overlap)
                cur, cur_len = _tail(cur, OVERLAP_WORDS)
            cur.append(sent)
            cur_len += wlen
        if cur:
            chunks.append(_make(doc_id, title, pg["page"], pg["ocr"], cur, len(chunks)))
    return chunks


def _tail(sents: list[str], n_words: int) -> tuple[list[str], int]:
    out: list[str] = []
    total = 0
    for sent in reversed(sents):
        wlen = len(sent.split())
        if total + wlen > n_words and out:
            break
        out.insert(0, sent)
        total += wlen
    return out, total


def _make(doc_id, title, page, ocr, sents, idx) -> dict:
    return {
        "id": f"{doc_id}#{idx}",
        "doc_id": doc_id,
        "title": title,
        "page": page,
        "ocr": ocr,
        "text": " ".join(sents),
    }
