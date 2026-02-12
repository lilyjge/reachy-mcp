"""Shared context for sentiment: last STT result and conversation (user said / robot said).

STT loop writes here; speak tool appends robot output. Sentiment loop reads only.
No duplicate mic usage: voice input for sentiment is the latest STT transcript.
"""

from __future__ import annotations

import threading
import time
from typing import Any

_lock = threading.Lock()
_last_stt: dict[str, Any] = {}
_conversation_buffer: list[dict[str, Any]] = []
_max_buffer = 20


def set_last_stt(text: str) -> None:
    """Store the latest STT transcript (called by STT loop after each transcription)."""
    with _lock:
        _last_stt["text"] = text
        _last_stt["timestamp"] = time.time()


def get_last_stt() -> dict[str, Any]:
    """Return the latest STT result for sentiment. Keys: text, timestamp."""
    with _lock:
        return {
            "text": _last_stt.get("text", ""),
            "timestamp": _last_stt.get("timestamp", 0.0),
        }


def append_user_said(text: str) -> None:
    """Append a user utterance (from STT). Call after transcribing."""
    if not text or not text.strip():
        return
    with _lock:
        _conversation_buffer.append({
            "role": "user",
            "content": text.strip(),
            "at": time.time(),
        })
        if len(_conversation_buffer) > _max_buffer:
            _conversation_buffer.pop(0)


def append_robot_said(text: str) -> None:
    """Append a robot utterance (from speak tool)."""
    if not text or not text.strip():
        return
    with _lock:
        _conversation_buffer.append({
            "role": "robot",
            "content": text.strip(),
            "at": time.time(),
        })
        if len(_conversation_buffer) > _max_buffer:
            _conversation_buffer.pop(0)


def get_conversation_buffer(n: int = 10) -> list[dict[str, Any]]:
    """Return last n entries (user/robot said only)."""
    with _lock:
        return list(_conversation_buffer[-n:])
