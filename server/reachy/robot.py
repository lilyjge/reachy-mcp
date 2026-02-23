import sys
import time
import math
from collections.abc import Callable

import httpx

from . import controller
from reachy_mini import ReachyMini
from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from typing import Any

_DAEMON_URL = "http://localhost:8000/api"
_daemon = httpx.Client(base_url=_DAEMON_URL, timeout=30.0)

# Dataset name used by the emotions library for recorded moves
_EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"


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


def _get_running_moves() -> list[dict[str, str]]:
    """Return list of currently running moves from the daemon."""
    resp = _daemon.get("/move/running")
    resp.raise_for_status()
    print("get running moves", resp)
    return resp.json()

def _go_to(head_x: float = 0,
        head_y: float = 0,
        head_z: float = 0,
        head_roll: float = 0,
        head_pitch: float = 0,
        head_yaw: float = 0,
        head_mm: bool = False,
        head_degrees: bool = True,
        antenna_x: float = 0,
        antenna_y: float = 0,
        body_yaw: float | None = 0.0,
        duration: float = 0.5,
        method: str = "minjerk",) -> str:
        _ensure_no_moves_running()

        # Unit conversions
        if head_mm:
            head_x /= 1000.0
            head_y /= 1000.0
            head_z /= 1000.0
        if head_degrees:
            head_roll = math.radians(head_roll)
            head_pitch = math.radians(head_pitch)
            head_yaw = math.radians(head_yaw)

        payload: dict[str, Any] = {
            "head_pose": {
                "x": head_x,
                "y": head_y,
                "z": head_z,
                "roll": head_roll,
                "pitch": head_pitch,
                "yaw": head_yaw,
            },
            "antennas": [antenna_x, antenna_y],
            "duration": duration,
            "interpolation": method,
        }
        if body_yaw is not None:
            payload["body_yaw"] = body_yaw

        resp = _daemon.post("/move/goto", json=payload)
        resp.raise_for_status()

def _go_to_default() -> str:
    """Go to the default position."""
    return _go_to()

def _stop_all_moves() -> list[str]:
    """Stop all currently running moves. Returns list of stop messages."""
    running = _get_running_moves()
    messages = []
    for m in running:
        try:
            stop_resp = _daemon.post("/move/stop", json={"uuid": m["uuid"]})
            stop_resp.raise_for_status()
            print("stop move", stop_resp)
            messages.append(stop_resp.json().get("message", f"Stopped {m['uuid']}"))
        except httpx.HTTPStatusError:
            messages.append(f"Failed to stop {m['uuid']}")
    return messages


def _ensure_no_moves_running() -> None:
    """If any moves are currently running, stop them all before proceeding."""
    if _get_running_moves():
        _stop_all_moves()
        time.sleep(0.05)


def register_robot_tools(mcp: FastMCP, get_mini: Callable[[], ReachyMini]):
    """Register all robot tools.

    Motion tools use the daemon HTTP API (non-blocking, cancellable, task-tracked).
    Media/vision tools use the ReachyMini SDK directly.
    """

    # ── Motion tools (via daemon HTTP API) ──────────────────────────

    @mcp.tool()
    def goto_target(
        head_x: float = 0,
        head_y: float = 0,
        head_z: float = 0,
        head_roll: float = 0,
        head_pitch: float = 0,
        head_yaw: float = 0,
        head_mm: bool = False,
        head_degrees: bool = True,
        antenna_x: float = 0,
        antenna_y: float = 0,
        body_yaw: float | None = 0.0,
        duration: float = 0.5,
        method: str = "minjerk",
    ) -> str:
        """
        Move Reachy Mini to desired position.
        Args:
            head_x (float): X coordinate of the position.
            head_y (float): Y coordinate of the position.
            head_z (float): Z coordinate of the position.
            head_roll (float): Roll angle
            head_pitch (float): Pitch angle
            head_yaw (float): Yaw angle
            head_mm (bool): If True, convert position from millimeters to meters.
            head_degrees (bool): If True, interpret roll, pitch, and yaw as degrees; otherwise as radians.
            antenna_x (float): X coordinate of the antenna position.
            antenna_y (float): Y coordinate of the antenna position.
            duration (float): Duration of the movement in seconds.
            method (str): Interpolation method ("linear", "minjerk", "ease", "cartoon"). Default is "minjerk".
            body_yaw (float | None): Body yaw angle in radians. Use None to keep the current yaw.
        """
        _log_tool_entered(
            "goto_target",
            head_x=head_x, head_y=head_y, head_z=head_z,
            head_roll=head_roll, head_pitch=head_pitch, head_yaw=head_yaw,
            head_mm=head_mm, head_degrees=head_degrees, antenna_x=antenna_x, antenna_y=antenna_y,
            body_yaw=body_yaw,
            duration=duration, method=method,
        )
        resp = _go_to(
            head_x=head_x,
            head_y=head_y,
            head_z=head_z,
            head_roll=head_roll,
            head_pitch=head_pitch,
            head_yaw=head_yaw,
            head_mm=head_mm,
            head_degrees=head_degrees,
            antenna_x=antenna_x,
            antenna_y=antenna_y,
            body_yaw=body_yaw,
            duration=duration,
            method=method,
        )
        return f"Move started: {resp}"

    @mcp.tool()
    def stop_move() -> str:
        """Stop all currently running moves on the robot."""
        _log_tool_entered("stop_move")
        messages = _stop_all_moves()
        _go_to_default()
        if not messages:
            return "No moves running."
        return "; ".join(messages)

    @mcp.tool()
    def wake_up() -> str:
        """Wake up the robot (enable motors and move to initial pose)."""
        _log_tool_entered("wake_up")
        _ensure_no_moves_running()
        resp = _daemon.post("/move/play/wake_up")
        resp.raise_for_status()
        print("wake up", resp)
        return f"Wake up started: {resp.json()['uuid']}"

    @mcp.tool()
    def go_to_sleep() -> str:
        """Put the robot to sleep (move to sleep pose and disable motors)."""
        _log_tool_entered("go_to_sleep")
        _ensure_no_moves_running()
        resp = _daemon.post("/move/play/goto_sleep")
        resp.raise_for_status()
        print("go to sleep", resp)
        return f"Sleep started: {resp.json()['uuid']}"
    
    @mcp.tool()
    def list_emotions() -> dict[str, str]:
        """List all emotions available in the emotions library."""
        _log_tool_entered("list_emotions")
        # Try to list from the daemon's recorded move datasets
        try:
            resp = _daemon.get(f"/move/recorded-move-datasets/list/{_EMOTIONS_DATASET}")
            resp.raise_for_status()
            print("list emotions", resp)
            moves = resp.json()
            return {m: m for m in moves}
        except httpx.HTTPStatusError:
            # Fall back to SDK-based listing
            return controller.list_emotions()

    @mcp.tool()
    def play_emotion(emotion: str) -> str:
        """Play an emotion.

        Args:
            emotion: The name of the emotion to play (use list_emotions to see available ones).
        """
        _log_tool_entered("play_emotion", emotion=emotion)
        _ensure_no_moves_running()
        # Play via daemon's recorded move dataset endpoint
        try:
            resp = _daemon.post(
                f"/move/play/recorded-move-dataset/{_EMOTIONS_DATASET}/{emotion}"
            )
            resp.raise_for_status()
            print("play emotion", resp)
            return f"Emotion started: {resp.json()['uuid']}"
        except httpx.HTTPStatusError as e:
            # Fall back to SDK if the daemon endpoint fails
            if e.response.status_code == 404:
                return f"Failed to play emotion: {emotion}"
            raise

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