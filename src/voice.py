"""Voice input via faster-whisper (CTranslate2).

tiny.en is ~75MB and transcribes on CPU in well under 8GB RAM, which suits this
machine. Loaded lazily so the text-only path never pays the model-load cost.
"""
from __future__ import annotations

from functools import lru_cache

from faster_whisper import WhisperModel

WHISPER_MODEL = "tiny.en"


@lru_cache(maxsize=1)
def _model() -> WhisperModel:
    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")


def transcribe(audio_path: str) -> str:
    """Transcribe an audio file (wav/webm/mp3/...) to text."""
    segments, _ = _model().transcribe(audio_path, beam_size=1, language="en")
    return " ".join(seg.text.strip() for seg in segments).strip()
