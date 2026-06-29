"""Quick end-to-end check: grounded answers, citations, and abstention."""
from src.rag import RAG

IN_CORPUS = [
    "What is the waiting period for pre-existing diseases in the health policy?",
    "How long do I have to report a fraudulent or disputed transaction?",
    "What is the minimum balance for a savings account?",
    "What is the lock-in / premature withdrawal penalty on a fixed deposit?",
]
OUT_OF_CORPUS = [
    "What's the weather in Mumbai tomorrow?",
    "Who won the 2022 FIFA World Cup?",
]


def main():
    rag = RAG()
    print("===== IN-CORPUS =====")
    for q in IN_CORPUS:
        r = rag.answer(q)
        print(f"\nQ: {q}\ngrounded={r['grounded']} conf={r['confidence']}")
        print(r["answer"])
        print("Sources:", ", ".join(f"{s['title']} p.{s['page']}" for s in r["sources"]))

    print("\n===== OUT-OF-CORPUS (should abstain) =====")
    for q in OUT_OF_CORPUS:
        r = rag.answer(q)
        print(f"\nQ: {q}\ngrounded={r['grounded']} conf={r['confidence']}")
        print(r["answer"][:120])


if __name__ == "__main__":
    main()
