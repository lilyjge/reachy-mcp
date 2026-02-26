"""Standalone face tracking helpers for Reachy Mini.

Run directly:
    python modules/face_tracking.py --mode center
    python modules/face_tracking.py --mode audio
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from typing import Any

import cv2
import httpx
import numpy as np
from reachy_mini import ReachyMini

_DAEMON_URL = "http://localhost:8000/api"
_daemon = httpx.Client(base_url=_DAEMON_URL, timeout=30.0)

# --- Face-centering constants ---
_CENTER_MAX_ITERS = 30
_CENTER_BOX_FRAC = 0.20
_CENTER_YAW_STEP_DEG = 2.0
_CENTER_PITCH_STEP_DEG = 2.0
_CENTER_MIN_FACE_PX = 60

_frontal_face_cascade: cv2.CascadeClassifier | None = None


def _get_running_moves() -> list[dict[str, str]]:
    """Return list of currently running moves from the daemon."""
    resp = _daemon.get("/move/running")
    resp.raise_for_status()
    return resp.json()

def _get_head_pose():
    """Get the current head pose from the daemon, as a dict with keys x/y/z/roll/pitch/yaw.
    Returns None if the head pose is not available.
    """
    resp = _daemon.get("/state/present_head_pose")
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()

def _go_to(
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
) -> None:
    """Move head/body/antennas via daemon goto endpoint."""
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


def _wait_for_moves_to_finish(timeout: float = 3.0, sleep: float = 0.1) -> bool:
    """Wait until daemon reports no running moves. Returns False on timeout."""
    deadline = time.time() + timeout
    while _get_running_moves():
        if time.time() >= deadline:
            return False
        time.sleep(sleep)
    return True


def _get_frontal_face_cascade() -> cv2.CascadeClassifier:
    """Lazy-load OpenCV Haar cascade for frontal face detection."""
    global _frontal_face_cascade
    if _frontal_face_cascade is None:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if cascade.empty():
            raise RuntimeError("Failed to load haarcascade_frontalface_default.xml")
        _frontal_face_cascade = cascade
    return _frontal_face_cascade


def _detect_largest_face(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return (x, y, w, h) of the largest face in frame, or None."""
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
    areas = [w * h for (_, _, w, h) in faces]
    idx = int(np.argmax(areas))
    return tuple(faces[idx])  # type: ignore[return-value]


def center_to_face(
    mini: ReachyMini,
    settle_time: float = 0.1,
    show_preview: bool = True,
    window_name: str = "Reachy Face Tracking",
) -> str:
    """Capture frames and nudge yaw/pitch until face center reaches middle box."""
    curr_pos = _get_head_pose()
    yaw = curr_pos["yaw"]
    pitch = curr_pos["pitch"]
    result = "Face found and head adjusted towards it (max iterations reached)."

    try:
        while True:
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

            face = _detect_largest_face(frame)
            if face is None:
                # print(f"center_to_face iter {i}: no face detected")
                if show_preview:
                    preview = frame.copy()
                    cv2.rectangle(
                        preview,
                        (box_left, box_top),
                        (box_right, box_bottom),
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        preview,
                        "No face detected",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(window_name, preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        result = "Stopped by user."
                        break
                continue

            x, y, fw, fh = face
            face_cx = x + fw / 2.0
            face_cy = y + fh / 2.0

            if show_preview:
                preview = frame.copy()
                cv2.rectangle(
                    preview,
                    (box_left, box_top),
                    (box_right, box_bottom),
                    (0, 255, 255),
                    2,
                )
                cv2.rectangle(preview, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
                cv2.circle(
                    preview,
                    (int(face_cx), int(face_cy)),
                    4,
                    (0, 0, 255),
                    thickness=-1,
                )
                cv2.imshow(window_name, preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    result = "Stopped by user."
                    break

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
    finally:
        if show_preview:
            cv2.destroyWindow(window_name)

    return result

def main() -> None:
    with ReachyMini() as mini:
        print(center_to_face(mini))


if __name__ == "__main__":
    main()
