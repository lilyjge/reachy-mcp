"""Background sentiment analysis loop: face, last STT, conversation (user/robot said) -> emotion agent -> robot emotion.

Uses shared context from STT loop and speak tool; no separate mic recording.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from time import monotonic

import cv2

from dotenv import load_dotenv

load_dotenv()

from reachy_mini import ReachyMini

from controller import vision
from controller import controls
from controller.sentiment_context import get_last_stt, get_conversation_buffer

# Configuration from env
SENTIMENT_LOOP_INTERVAL = float(os.environ.get("SENTIMENT_LOOP_INTERVAL", "5.0"))
SENTIMENT_DEBOUNCE_SEC = float(os.environ.get("SENTIMENT_DEBOUNCE_SEC", "3.0"))
SENTIMENT_ENABLED = os.environ.get("SENTIMENT_ENABLED", "true").lower() in ("1", "true", "yes")
SENTIMENT_CONVERSATION_MESSAGES = int(os.environ.get("SENTIMENT_CONVERSATION_MESSAGES", "10"))

_state_lock = threading.Lock()
_current_emotion: str | None = None
_last_emotion_time: float = 0.0
_emotion_agent = None


def _capture_frame_and_analyze_face(mini: ReachyMini) -> dict | None:
    """Capture one frame, save to disk, run DeepFace emotion analysis. Returns emotion dict or None on error."""
    try:
        _REACHY_DIR = vision._IMAGES_DIR / "reachy"
        _REACHY_DIR.mkdir(parents=True, exist_ok=True)
        path = _REACHY_DIR / f"sentiment_{monotonic():.3f}.jpg"
        _ = mini.media.get_frame()
        frame = mini.media.get_frame()
        if not cv2.imwrite(str(path), frame):
            return None
        result = vision.analyze_face(str(path))
        if isinstance(result, list) and len(result) > 0:
            return result[0]
        if isinstance(result, dict):
            return result
        return None
    except Exception as e:
        print(f"Sentiment loop: face analysis error: {e}", file=sys.stderr)
        return None


def _build_context_message(facial_emotion: dict | None, last_stt: dict, conversation: list[dict]) -> str:
    """Build the message for the emotion agent from face, last STT text, and user/robot conversation."""
    parts = []

    if facial_emotion and isinstance(facial_emotion, dict):
        em = facial_emotion.get("emotion") or facial_emotion.get("dominant_emotion")
        if isinstance(em, dict):
            parts.append("Face emotions (scores): " + ", ".join(f"{k}={v}" for k, v in em.items()))
        elif isinstance(em, str):
            parts.append(f"Dominant face emotion: {em}.")
        else:
            parts.append(str(facial_emotion))
    else:
        parts.append("No face detected or analysis failed.")

    text = (last_stt.get("text") or "").strip()
    if text:
        parts.append(f"User last said (from mic): {text}")
    else:
        parts.append("User has not said anything recently (no STT).")

    if conversation:
        lines = []
        for m in conversation:
            role = m.get("role", "?")
            content = (m.get("content") or "")[:300]
            lines.append(f"  {role}: {content}")
        parts.append("Recent conversation (user = speech from mic, robot = what the robot said):\n" + "\n".join(lines))
    else:
        parts.append("No recent conversation.")

    return "\n\n".join(parts)


def _should_update_emotion(new_emotion: str, _confidence: float = 1.0) -> bool:
    """True if we should call play_emotion (debounce + different from current)."""
    if not new_emotion:
        return False
    with _state_lock:
        now = time.time()
        if _current_emotion == new_emotion:
            return False
        if now - _last_emotion_time < SENTIMENT_DEBOUNCE_SEC:
            return False
        return True


def _mark_emotion_played(emotion: str) -> None:
    with _state_lock:
        global _current_emotion, _last_emotion_time
        _current_emotion = emotion
        _last_emotion_time = time.time()


def _try_play_emotion(mini: ReachyMini, emotion: str) -> bool:
    """Called by emotion agent tool. Returns True if emotion was played (debounce applied)."""
    if not emotion or not emotion.strip():
        return False
    emotion = emotion.strip()
    if not _should_update_emotion(emotion, 1.0):
        return False
    try:
        controls.play_emotion(mini, emotion)
        _mark_emotion_played(emotion)
        return True
    except Exception as e:
        print(f"Sentiment loop: play_emotion error: {e}", file=sys.stderr)
        return False


def _get_emotion_agent(mini: ReachyMini):
    """Lazy-create the pydantic emotion agent (needs mini for get_mini)."""
    global _emotion_agent
    if _emotion_agent is None:
        from controller.emotion_agent import make_emotion_agent
        get_mini = lambda: mini  # noqa: E731
        _emotion_agent = make_emotion_agent(get_mini, _try_play_emotion)
    return _emotion_agent


def _analyze_sentiment(mini: ReachyMini) -> None:
    """Gather face + last STT + conversation from shared context; run emotion agent; agent calls play_emotion if appropriate."""
    if not controls.list_emotions():
        return

    facial_emotion = _capture_frame_and_analyze_face(mini)
    last_stt = get_last_stt()
    conversation = get_conversation_buffer(n=SENTIMENT_CONVERSATION_MESSAGES)

    context_message = _build_context_message(facial_emotion, last_stt, conversation)

    try:
        agent = _get_emotion_agent(mini)
        agent.run_sync(context_message)
    except Exception as e:
        print(f"Sentiment loop: emotion agent error: {e}", file=sys.stderr)


def run_sentiment_loop(
    mini: ReachyMini,
    stop_event: threading.Event | None = None,
) -> None:
    """Run the sentiment analysis loop in the current thread. Stops when stop_event is set."""
    stop = stop_event or threading.Event()
    while not stop.is_set():
        try:
            _analyze_sentiment(mini)
        except Exception as e:
            if not stop.is_set():
                print(f"Sentiment loop error: {e}", file=sys.stderr)
                time.sleep(1.0)
        for _ in range(int(SENTIMENT_LOOP_INTERVAL / 0.5)):
            if stop.is_set():
                break
            time.sleep(0.5)


def start_sentiment_loop(mini: ReachyMini) -> tuple[threading.Thread | None, threading.Event | None]:
    """Start the sentiment loop in a daemon thread if SENTIMENT_ENABLED is true.
    Returns (thread, stop_event). If disabled, returns (None, None). Call stop_event.set() to stop."""
    if not SENTIMENT_ENABLED:
        return (None, None)
    stop_event = threading.Event()
    thread = threading.Thread(
        target=run_sentiment_loop,
        args=(mini,),
        kwargs={"stop_event": stop_event},
        daemon=True,
    )
    thread.start()
    return (thread, stop_event)
