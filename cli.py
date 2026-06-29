"""Terminal chat. Type a question, or `:voice path/to/audio.wav` to ask by voice."""
from __future__ import annotations

from src.rag import RAG


def _print(res: dict) -> None:
    print(f"\n{res['answer']}")
    if res["sources"]:
        srcs = ", ".join(f"[{s['n']}] {s['title']} p.{s['page']}" for s in res["sources"])
        print(f"\nSources: {srcs}")
    print(f"(confidence {res['confidence']})\n")


def main() -> None:
    print("BFSI Policy Assistant. Ask about loaded policies/T&Cs. Ctrl-C to quit.")
    rag = RAG()
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.startswith(":voice "):
            from src.voice import transcribe
            q = transcribe(q[len(":voice "):].strip())
            print(f"(heard: {q})")
        _print(rag.answer(q))


if __name__ == "__main__":
    main()
