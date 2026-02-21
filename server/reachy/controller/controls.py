"""
NO LONGER USED IN robot.py CAN BE DEPRICATED IN THE FUTURE
"""

from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini.utils import create_head_pose
from threading import Thread

EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"
recorded_emotions = RecordedMoves(EMOTIONS_DATASET)
move_names = recorded_emotions.list_moves()
moves_and_descriptions = {name: recorded_emotions.get(name).description for name in move_names}


def _play_emotion_worker(mini: ReachyMini, emotion: str) -> None:
    """Internal helper to play an emotion in a separate thread."""
    move = recorded_emotions.get(emotion)
    mini.play_move(move, initial_goto_duration=0.7)

def play_emotion(mini: ReachyMini, emotion: str):
    """Play an emotion.

    Runs in a background thread to avoid AsyncToSync being used
    from the same thread as the FastMCP async event loop.
    """
    if emotion not in moves_and_descriptions:
        return "Emotion not found! Use list_emotions to get the list of available emotions."
    else:
        t = Thread(target=_play_emotion_worker, args=(mini, emotion))
        t.start()
        t.join()
        return "Done playing emotion."

def list_emotions() -> dict[str, str]:
    """List all emotions available in the emotions library."""
    return moves_and_descriptions

def goto_target(
    mini: ReachyMini,
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