"""Tiny Okapi BM25, written from scratch.

Dense embeddings capture meaning but miss exact tokens that matter a lot in
BFSI text: amounts, section numbers, acronyms (KYC, AML, EMI, NEFT). BM25 is a
keyword scorer that nails those. We fuse the two in retrieve.py.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = [tokenize(d) for d in docs]
        self.N = len(self.corpus)
        self.doc_len = [len(d) for d in self.corpus]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.tf = [Counter(d) for d in self.corpus]
        df: Counter[str] = Counter()
        for d in self.corpus:
            for term in set(d):
                df[term] += 1
        # BM25+ style idf, floored at 0 so common terms never go negative.
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = [0.0] * self.N
        for i in range(self.N):
            dl = self.doc_len[i]
            tf = self.tf[i]
            s = 0.0
            for term in q:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += idf * (freq * (self.k1 + 1)) / denom
            out[i] = s
        return out
