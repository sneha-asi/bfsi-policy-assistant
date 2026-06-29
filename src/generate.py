"""Grounded answer generation with local Ollama (llama3.2:3b).

The model is instructed to answer ONLY from the supplied context and to cite the
source documents. We keep temperature low for factual, repeatable answers. If
retrieval already decided the query is out-of-corpus we never even call the LLM.
"""
from __future__ import annotations

import ollama

GEN_MODEL = "llama3.2:3b"

REFUSAL = (
    "I can only answer from the loaded BFSI policy and terms documents, "
    "and I couldn't find anything relevant to that. Please rephrase or ask "
    "about a covered policy (e.g. health/motor/life insurance, home loan, "
    "savings/FD terms, credit card, KYC/AML, claims, grievance, privacy, fraud)."
)

_SYSTEM = (
    "You are a careful BFSI policy assistant. Answer the user's question using "
    "ONLY the numbered context passages provided. Do not use outside knowledge. "
    "If the answer is not in the context, say you don't have that information. "
    "Be concise and precise with numbers, durations and amounts. After the "
    "answer, cite the sources you used as [Source N]."
)


def _format_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(f"[{i}] ({h['title']}, p.{h['page']})\n{h['text']}")
    return "\n\n".join(blocks)


def generate(question: str, hits: list[dict]) -> str:
    context = _format_context(hits)
    user = (
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, then list the [Source N] you used."
    )
    resp = ollama.chat(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0.1, "num_ctx": 4096},
    )
    return resp["message"]["content"].strip()
