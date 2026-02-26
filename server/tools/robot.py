from collections.abc import Callable

from ..reachy import controller
from reachy_mini import ReachyMini
from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from typing import Any

def register_robot_tools(mcp: FastMCP, get_mini: Callable[[], ReachyMini]):
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
        print("calling goto_target")
        controller.goto_target(get_mini(), head_x, head_y, head_z, head_roll, head_pitch, head_yaw, head_mm, head_degrees, body_yaw, duration, method)
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
        print("calling take_picture")
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
        print("calling describe_image with image: " + image + " and question: " + question)
        return controller.describe_image(image, question)


    @mcp.tool()
    def detect_faces(image: str) -> Any:
        """Detect faces in an image.

        Args:
            image: Either a local filename eg returned by take_picture, absolute path, 
                    or an HTTP(S) URL to an image which will be downloaded and cached.
        """
        print("calling detect_faces with image: " + image)
        return controller.detect_faces(image)

    @mcp.tool()
    def analyze_face(image: str) -> Any:
        """Analyze the face in the image using the DeepFace model.

        Args:
            image: Either a local filename eg returned by take_picture, absolute path, 
                    or an HTTP(S) URL to an image which will be downloaded and cached.
        """
        print("calling analyze_face with image: " + image)
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
        print("calling save_image_person with image: " + image + " and person_name: " + person_name)
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

        Runs in a background thread to avoid blocking the FastMCP event loop.
        """
        print(f"calling speak with text: {text}, forcefully_interrupt: {forcefully_interrupt}")
        controller.speak(get_mini(), text, forcefully_interrupt)
        return "Done"


    @mcp.tool()
    def list_emotions() -> dict[str, str]:
        """List all emotions available in the emotions library."""
        print("calling list_emotions")
        return controller.list_emotions()

    @mcp.tool()
    def play_emotion(emotion: str) -> str:
        """Play an emotion.

        Runs in a background thread to avoid AsyncToSync being used
        from the same thread as the FastMCP async event loop.
        """
        print("calling play_emotion with emotion: " + emotion)
        return controller.play_emotion(get_mini(), emotion)
