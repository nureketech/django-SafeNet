"""Stream audio -> transcript (optional, ``faster-whisper``).

Stream transcripts usually arrive as text and are analysed with
``source_type="stream_transcript"`` (evidence ``source: "transcript"``). For raw
audio, ``transcribe_audio`` uses faster-whisper when it is installed
(``pip install faster-whisper``); otherwise :class:`TranscriberUnavailable` explains why.
"""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache


class TranscriberUnavailable(RuntimeError):
    """faster-whisper is not installed."""


def available() -> tuple[bool, str]:
    if importlib.util.find_spec("faster_whisper") is None:
        return False, "faster-whisper is not installed (`pip install faster-whisper`)"
    return True, ""


@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel

    return WhisperModel(os.environ.get("SAFENET_WHISPER_MODEL", "small"), device="cpu", compute_type="int8")


def transcribe_audio(path: str, language: str | None = None) -> str:
    """Transcribe an audio/video file (Russian or Kazakh speech) into text."""
    ok, reason = available()
    if not ok:
        raise TranscriberUnavailable(reason)
    segments, _info = _model().transcribe(path, language=language, vad_filter=True)
    return " ".join(segment.text.strip() for segment in segments)
