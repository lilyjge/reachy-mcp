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
    """Register all robot tools. FastMCP handles sync tools automatically."""

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
            duration (float): Duration of the movement in seconds.
            method (InterpolationTechnique): Interpolation method to use ("linear", "minjerk", "ease", "cartoon"). Default is "minjerk".
            body_yaw (float | None): Body yaw angle in radians. Use None to keep the current yaw.
        """
        _log_tool_entered(
            "goto_target",
            head_x=head_x, head_y=head_y, head_z=head_z,
            head_roll=head_roll, head_pitch=head_pitch, head_yaw=head_yaw,
            head_mm=head_mm, head_degrees=head_degrees, body_yaw=body_yaw,
            duration=duration, method=method,
        )
        controller.goto_target(
            get_mini(), head_x, head_y, head_z, head_roll, head_pitch, head_yaw,
            head_mm, head_degrees, body_yaw, duration, method,
        )
        return "Done"

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

    @mcp.tool()
    def list_emotions() -> dict[str, str]:
        """List all emotions available in the emotions library."""
        _log_tool_entered("list_emotions")
        return controller.list_emotions()

    @mcp.tool()
    def play_emotion(emotion: str) -> str:
        """Play an emotion."""
        _log_tool_entered("play_emotion", emotion=emotion)
        return controller.play_emotion(get_mini(), emotion)