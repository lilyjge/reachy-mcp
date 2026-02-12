"""Pydantic agent that chooses and plays robot emotion from context (face, user speech, conversation)."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Lazy model init to avoid import-time API calls
_model: Any = None


def _get_model():
    global _model
    if _model is None:
        from pydantic_ai.models.groq import GroqModel
        from pydantic_ai.providers.groq import GroqProvider
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY required for emotion agent")
        _model = GroqModel(
            "meta-llama/llama-4-scout-17b-16e-instruct",
            provider=GroqProvider(api_key=api_key),
        )
    return _model


def make_emotion_agent(
    get_mini: Callable[[], Any],
    try_play_emotion: Callable[[Any, str], bool],
) -> Any:
    """Build an Agent with list_emotions and play_emotion tools. Uses Groq.
    try_play_emotion(mini, emotion) returns True if emotion was played (debouncing inside)."""
    from pydantic_ai import Agent

    from controller import controls

    def list_emotions() -> dict[str, str]:
        """List available robot emotions (name -> description)."""
        return controls.list_emotions()

    def play_emotion(emotion: str) -> str:
        """Play the given emotion on the robot. Use one of the names from list_emotions.
        May be skipped if the same emotion was played recently (debounce)."""
        if not emotion or not emotion.strip():
            return "No emotion name provided."
        names = controls.list_emotions()
        name = emotion.strip()
        if name not in names:
            for k in names:
                if k.lower() == name.lower():
                    name = k
                    break
            else:
                return f"Unknown emotion. Use one of: {', '.join(names.keys())}."
        mini = get_mini()
        if mini is None:
            return "Robot not available."
        if try_play_emotion(mini, name):
            return f"Playing emotion: {name}."
        return "Skipped (same emotion or debounce)."

    instructions = """You are an emotion controller for a robot. You will receive context about the user: their facial expression, what they last said (from speech), and recent conversation (user and robot turns).

Your only job is to call play_emotion with exactly one emotion name from the list_emotions result. Choose the emotion that best matches the user's apparent state (e.g. happy face and cheerful words -> joy; frustrated tone -> empathy or concern). If context is empty or unclear, prefer a neutral or calm emotion. Call play_emotion exactly once per turn with your chosen emotion."""

    return Agent(
        _get_model(),
        tools=[list_emotions, play_emotion],
        instructions=instructions,
    )
