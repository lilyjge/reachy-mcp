from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini.utils import create_head_pose
# from threading import Thread
import threading
from typing import Any
import httpx
import time
import math
import json
import random
from pathlib import Path

import cv2
import numpy as np

from server.reachy import controller

_DAEMON_URL = "http://localhost:8000/api"
_daemon = httpx.Client(base_url=_DAEMON_URL, timeout=30.0)

_EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"
recorded_emotions = RecordedMoves(_EMOTIONS_DATASET)
# move_names = recorded_emotions.list_moves()
# moves_and_descriptions = {name: recorded_emotions.get(name).description for name in move_names}

_MOVE_TYPES_PATH = Path(__file__).parent / "move_types.json"
moves_: dict[str, list[str]] = json.loads(_MOVE_TYPES_PATH.read_text())

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
        method: str = "minjerk",
        keep_head_positon: dict | None = None) -> str:

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
            } if not keep_head_positon else keep_head_positon,
            "antennas": [antenna_x, antenna_y],
            "duration": duration,
            "interpolation": method,
        }
        if body_yaw is not None:
            payload["body_yaw"] = body_yaw

        resp = _daemon.post("/move/goto", json=payload)
        resp.raise_for_status()

def _get_head_pose() -> dict[str, float]:
    """Get the current head pose from the daemon, as a dict with keys x/y/z/roll/pitch/yaw.
    Returns None if the head pose is not available.
    """
    resp = _daemon.get("/state/present_head_pose")
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()

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


def listening_pose(mini: ReachyMini, listening: threading.Event) -> None:
    """Move to a listening pose (head slightly down, facing forward)."""
    curr_head = _get_head_pose()
    while listening.is_set():
        _go_to(antenna_x=math.radians(-20), antenna_y=math.radians(20), duration=1, keep_head_positon=curr_head)
        _wait_for_moves_to_finish()
        _go_to(antenna_x=0, antenna_y=0, duration=1, keep_head_positon=curr_head)
        _wait_for_moves_to_finish()

def listening_worker(mini: ReachyMini, stop: threading.Event, listening: threading.Event) -> None:
    """Worker function to run the listening pose in a separate thread."""
    while not stop.is_set():
        if not listening.wait(timeout=0.1):
            continue
        listening_pose(mini, listening)
        time.sleep(0.4)  # small delay to avoid busy looping

def _play_emotion(emotion: str) -> str:
    """Play an emotion from the recorded moves dataset."""
    _stop_all_moves()
    # Resolve emotion type to a specific move variant
    variants = moves_.get(emotion)
    emotion_to_play = emotion
    if variants:
        emotion_to_play = random.choice(variants)
    else:
        raise ValueError(f"No variants found for emotion: {emotion}")
    # Play via daemon's recorded move dataset endpoint
    try:
        resp = _daemon.post(
            f"/move/play/recorded-move-dataset/{_EMOTIONS_DATASET}/{emotion_to_play}"
        )
        resp.raise_for_status()
        print("play emotion", resp)
        return f"Emotion started: {resp.json()['uuid']}"
    except httpx.HTTPStatusError:
        raise ValueError(f"Failed to play emotion: {emotion}")

def _doa_degrees_to_head_yaw(angle_degrees: float) -> float:
    """Convert DoA angle frame to Reachy head_yaw degrees.

    DoA frame:
      - 0   = robot left
      - 90  = robot front
      - 180 = robot right
      - 270 = robot back

    head_yaw frame:
      - 0    = front
      - +90  = left
      - -90  = right
      - ±180 = back
    """
    yaw = 90.0 - (angle_degrees % 360.0)
    return ((yaw + 180.0) % 360.0) - 180.0

def move_to_audio(mini: ReachyMini, doa: tuple[float, bool] | None = None) -> str:
    print("Getting direction of arrival from audio...")
    if doa is None:
        print("No DoA provided, fetching from mini.media.get_DoA()")
        doa = mini.media.get_DoA()

    if doa is not None:
        angle, is_valid = doa
        if is_valid:
            angle_degrees = math.degrees(angle)
            head_yaw = _doa_degrees_to_head_yaw(angle_degrees)
            print(
                f"Direction of Arrival (DoA) angle: {angle_degrees:.2f} degrees"
                f" -> head_yaw: {head_yaw:.2f} degrees"
            )
            _go_to(head_yaw=head_yaw, duration=2)
            center_to_face(mini, head_yaw=head_yaw)
        else:
            print("DoA data is invalid.")
    else:
        print("No Audio direction detected.")
    return "Moved head towards audio source."



# --- Face-centering constants ---
# Max refinement iterations to avoid infinite loops.
_CENTER_MAX_ITERS = 10
# Middle target box size as fraction of frame dimensions.
_CENTER_BOX_FRAC = 0.20
# Fixed incremental nudges (degrees) applied each iteration.
_CENTER_YAW_STEP_DEG = 2.0
_CENTER_PITCH_STEP_DEG = 2.0
# Minimum face pixel size to consider.
_CENTER_MIN_FACE_PX = 60


def _detect_largest_face(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return (x, y, w, h) of the largest face in *frame*, or None."""
    from .vision import _get_frontal_face_cascade

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = _get_frontal_face_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(_CENTER_MIN_FACE_PX, _CENTER_MIN_FACE_PX),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return None
    # Pick the largest face by area
    areas = [w * h for (_, _, w, h) in faces]
    idx = int(np.argmax(areas))
    return tuple(faces[idx])  # type: ignore[return-value]

def _wait_for_moves_to_finish(timeout: float = 3, sleep_timer: float = 0.5) -> bool:
    """Wait until daemon reports no running moves. Returns False on timeout."""
    deadline = time.time() + timeout
    while _get_running_moves():
        if time.time() >= deadline:
            return False
        time.sleep(sleep_timer)
    return True

def center_to_face(
    mini: ReachyMini,
    head_yaw: float | None = None,
) -> str:
    """Capture frames and nudge yaw/pitch until face center reaches middle box."""
    print("Centering head to face...")
    _wait_for_moves_to_finish(timeout=8)
    curr_pos = _get_head_pose()
    yaw = curr_pos["yaw"] if head_yaw is None else head_yaw
    pitch = curr_pos["pitch"]
    no_face_iters = 0

    while True and no_face_iters < _CENTER_MAX_ITERS:
        _ = mini.media.get_frame()
        frame = mini.media.get_frame()
        if frame is None:
            result = "Error: could not capture frame from camera."
            break

        img_h, img_w = frame.shape[:2]
        box_w = img_w * _CENTER_BOX_FRAC
        box_h = img_h * _CENTER_BOX_FRAC
        box_left = int((img_w - box_w) / 2.0)
        box_right = int((img_w + box_w) / 2.0)
        box_top = int((img_h - box_h) / 2.0)
        box_bottom = int((img_h + box_h) / 2.0)

        try:
            face = _detect_largest_face(frame)
        except Exception as e:
            print(f"Error during face detection: {e}")
            continue
        if face is None:
            print("No face detected in frame.")
            no_face_iters += 1
            continue
        try:
            x, y, fw, fh = face
        except TypeError:
            continue
        face_cx = x + fw / 2.0
        face_cy = y + fh / 2.0

        if box_left <= face_cx <= box_right and box_top <= face_cy <= box_bottom:
            print("Face found and head centred on it.")
            break

        delta_yaw = 0.0
        delta_pitch = 0.0
        if face_cx > box_right:
            delta_yaw = -_CENTER_YAW_STEP_DEG
        elif face_cx < box_left:
            delta_yaw = _CENTER_YAW_STEP_DEG
        if face_cy > box_bottom:
            delta_pitch = -_CENTER_PITCH_STEP_DEG
        elif face_cy < box_top:
            delta_pitch = _CENTER_PITCH_STEP_DEG

        yaw += delta_yaw
        pitch -= delta_pitch
        _wait_for_moves_to_finish()
        _go_to(head_yaw=yaw, head_pitch=pitch, duration=0.02)

    print("Moved head to center on face.")
