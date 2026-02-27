from pyexpat.errors import messages
import sys
import time
from collections.abc import Callable

from . import controller
from reachy_mini import ReachyMini
from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from typing import Any


def _log_tool_entered(name: str, **params: Any) -> None:
    """Log as soon as the server's event loop invokes this tool (for latency debugging)."""
    ts = time.strftime("%H:%M:%S", time.localtime())
    parts = [f"{k}={v!r}" for k, v in params.items()]
    params_str = ", ".join(parts) if parts else ""
    msg = f"[{ts}] MCP tool entered: {name}({params_str})" if params_str else f"[{ts}] MCP tool entered: {name}"
    # Truncate very long values (e.g. base64 image strings) so the log line stays readable
    if len(msg) > 500:
        msg = msg[:497] + "..."
    print(msg, flush=True, file=sys.stderr)


def register_robot_tools(mcp: FastMCP, get_mini: Callable[[], ReachyMini]):
    """Register all robot tools.

    Motion tools use the daemon HTTP API (non-blocking, cancellable, task-tracked).
    Media/vision tools use the ReachyMini SDK directly.
    """

    # ── Motion tools (via daemon HTTP API) ──────────────────────────
    @mcp.tool()
    def move_to_audio() -> str:
        """Move Reachy Mini's head towards the direction of arrival of the last recorded audio."""
        _log_tool_entered("move_to_audio")
        return controller.move_to_audio(get_mini())

    @mcp.tool()
    def stop_move() -> str:
        """Stop all currently running moves on the robot."""
        _log_tool_entered("stop_move")
        messages = controller._stop_all_moves()
        if not messages:
            return "No moves running."
        return "; ".join(messages)

    @mcp.tool()
    def reset_head() -> str:
        """Stops all movement, resets head position to default"""
        _log_tool_entered("reset_head")
        controller._go_to_default()
        return "Reset done."

    @mcp.tool()
    def move_head_left() -> str:
        """Move the head to the left."""
        _log_tool_entered("move_head_left")
        return controller.move_head_left()

    @mcp.tool()
    def move_head_right() -> str:
        """Move the head to the right."""
        _log_tool_entered("move_head_right")
        return controller.move_head_right()

    @mcp.tool()
    def list_emotions() -> list[str]:
        """List the available emotions that can be played."""
        _log_tool_entered("list_emotions")
        return controller._get_available_emotions()

    @mcp.tool()
    def play_emotion(emotion: str) -> str:
        """Play an emotion.

        Args:
            emotion: The name of the emotion to play (use list_emotions to see available ones).
        """
        _log_tool_entered("play_emotion", emotion=emotion)
        controller._play_emotion(emotion)
        return f"Emotion '{emotion}' started."

    # ── Media / vision tools (via SDK) ──────────────────────────────

    @mcp.tool()
    def take_picture(for_text_only_model: bool = True) -> tuple[str, Image | str]:
        """Take a picture with Reachy Mini's camera.

        Every capture is saved under images/ (cleared on server shutdown).

        Args:
            for_text_only_model: If True, return a text description of the image
                instead of the image itself. Use this when the client model does
                not accept images (e.g. text-only LLM). If False, return the image
                for multimodal models.
        Returns:
            tuple[str, Image | str]: The path to the image and the image or text description of the image.
        """
        _log_tool_entered("take_picture", for_text_only_model=for_text_only_model)
        return controller.take_picture(get_mini(), for_text_only_model)

    @mcp.tool()
    def describe_image(image: str, question: str = "What is in the image?") -> Any:
        """Get a short text description of an image (e.g. from take_picture).

        Use when the model does not accept images.

        Args:
            image: Either a local filename eg returned by take_picture, absolute path,
                    or an HTTP(S) URL to an image which will be downloaded and cached.
            question: The question to ask the model. Defaults to "What is in the image?"
        Returns:
            tuple[str, str]: The path to the image and the text description of the image.
        """
        _log_tool_entered("describe_image", image=image, question=question)
        return controller.describe_image(image, question)

    @mcp.tool()
    def detect_faces(image: str) -> Any:
        """Detect faces in an image.

        Args:
            image: Either a local filename eg returned by take_picture, absolute path,
                    or an HTTP(S) URL to an image which will be downloaded and cached.
        """
        _log_tool_entered("detect_faces", image=image)
        return controller.detect_faces(image)

    @mcp.tool()
    def analyze_face(image: str) -> Any:
        """Analyze the face in the image using the DeepFace model.

        Args:
            image: Either a local filename eg returned by take_picture, absolute path,
                    or an HTTP(S) URL to an image which will be downloaded and cached.
        """
        _log_tool_entered("analyze_face", image=image)
        return controller.analyze_face(image)

    @mcp.tool()
    def save_image_person(image: str, person_name: str) -> str:
        """If a name is provided for a person in the image, use this to save an image of a person.

        Args:
            image: Either a local filename eg returned by take_picture, absolute path,
                    or an HTTP(S) URL to an image which will be downloaded and cached.
            person_name: The name of the person to save the image of. The image is copied
                to `images/people/<person_name>` with a unique filename.
        """
        _log_tool_entered("save_image_person", image=image, person_name=person_name)
        return controller.save_image_person(image, person_name)

    @mcp.tool()
    def speak(text: str, forcefully_interrupt: bool = False) -> str:
        """Speak words using text to speech with Reachy Mini's speaker.

        Args:
            text: The text to speak.
            forcefully_interrupt: If True and the robot is currently speaking, stop
                the current speech, clear any queued speech, and speak immediately.
                If False (default) and the robot is speaking, queue this request to
                execute after the current speech finishes.
        """
        _log_tool_entered("speak", text=text, forcefully_interrupt=forcefully_interrupt)
        controller.speak(get_mini(), text, forcefully_interrupt)
        return "Done"