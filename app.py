"""Flask web UI: typed questions and voice (mic) questions.

/ask    -> JSON {question} -> grounded answer + sources
/voice  -> multipart audio -> Whisper transcript -> same answer pipeline
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.rag import RAG

app = Flask(__name__)
rag = RAG()

DOCS_DIR = Path(__file__).resolve().parent / "docs"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    question = (request.json or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "empty question"}), 400
    res = rag.answer(question)
    res["question"] = question
    return jsonify(res)


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    pdfs = [f for f in files if (f.filename or "").lower().endswith(".pdf")]
    if not pdfs:
        return jsonify({"error": "no PDF files"}), 400
    added = []
    for f in pdfs:
        dest = DOCS_DIR / Path(f.filename).name
        f.save(dest)
        added.append(rag.add_pdf(dest))
    total = rag.retriever.store.index.ntotal
    return jsonify({"added": added, "total_chunks": total})


@app.route("/voice", methods=["POST"])
def voice():
    from src.voice import transcribe

    audio = request.files.get("audio")
    if not audio:
        return jsonify({"error": "no audio"}), 400
    suffix = Path(audio.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        audio.save(tmp.name)
        question = transcribe(tmp.name)
    if not question:
        return jsonify({"error": "could not transcribe"}), 400
    res = rag.answer(question)
    res["question"] = question
    return jsonify(res)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5099, debug=False)
