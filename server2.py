"""FastMCP quickstart example.

Run from the repository root:
    uv run examples/snippets/servers/fastmcp_quickstart.py
"""

from time import sleep
from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import cv2
from pocket_tts import TTSModel
import scipy.io.wavfile
from sound_play import play

mcp = FastMCP("Reachy Mini Robot")

@mcp.tool()
def move(
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
):
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
    with ReachyMini() as mini:
        mini.goto_target(
            head=create_head_pose(
                x=head_x,
                y=head_y,
                z=head_z,
                roll=head_roll,
                pitch=head_pitch,
                yaw=head_yaw,
                mm=head_mm,
                degrees=head_degrees,
            ),
            body_yaw=body_yaw,
            duration=duration,
            method=method,
        )


@mcp.tool()
def take_picture() -> None:
    """Take a picture with Reachy Mini's camera."""
    with ReachyMini() as mini:
        sleep(1)
        frame = mini.media.get_frame()
        print(cv2.imwrite("reachy2.jpg", frame))
        return Image(path="reachy2.jpg")


@mcp.tool()
def speak(text: str) -> None:
    """Speak words using text to speech with Reachy Mini's speaker."""
    tts_model = TTSModel.load_model()
    voice_state = tts_model.get_state_for_audio_prompt(
        "cosette"
    )
    audio = tts_model.generate_audio(voice_state, text)
    # Audio is a 1D torch tensor containing PCM data.
    scipy.io.wavfile.write("output.wav", tts_model.sample_rate, audio.numpy())
    play("output.wav")

# Run with streamable HTTP transport
if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=5000)