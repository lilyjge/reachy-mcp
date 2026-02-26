from __future__ import annotations

import argparse
import json
import math
import random
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
from reachy_mini import ReachyMini

_DAEMON_URL = "http://localhost:8000/api"
_daemon = httpx.Client(base_url=_DAEMON_URL, timeout=30.0)
_EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"
_MOVE_TYPES_PATH = (
    Path(__file__).resolve().parent.parent
    / "server"
    / "reachy"
    / "controller"
    / "move_types.json"
)
moves_: dict[str, list[str]] = json.loads(_MOVE_TYPES_PATH.read_text())

emotions_list = ["loving", "attentive", "confused", "frustrated"]

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
        print("play emotion: ", resp)
        return f"Emotion started: {resp.json()['uuid']}"
    except httpx.HTTPStatusError:
        raise ValueError(f"Failed to play emotion: {emotion}")


def _stop_all_moves() -> list[str]:
    """Stop all currently running moves. Returns list of stop messages."""
    running = _get_running_moves()
    messages = []
    for m in running:
        try:
            stop_resp = _daemon.post("/move/stop", json={"uuid": m["uuid"]})
            stop_resp.raise_for_status()
            messages.append(stop_resp.json().get("message", f"Stopped {m['uuid']}"))
        except httpx.HTTPStatusError:
            messages.append(f"Failed to stop {m['uuid']}")
    return messages


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

def wait_for_headpat(
    min_change_seconds: float = 0.5,
    poll_interval: float = 0.05,
    quiet_reset_seconds: float = 0.3,
) -> str:
    """Wait for a headpat-like oscillation and play the loving emotion."""
    original_pose = _get_head_pose()
    if original_pose is None:
        raise RuntimeError("Head pose is not available from the daemon.")

    print("ready")
    angle_threshold_rad = math.radians(0.1)
    position_threshold_m = 0.0005  # 5 mm
    thresholds = {
        "pitch": angle_threshold_rad,
        "yaw": angle_threshold_rad,
        "x": position_threshold_m,
        "y": position_threshold_m,
        "z": position_threshold_m,
    }

    direction_seen: dict[str, set[str]] = {axis: set() for axis in thresholds}
    movement_started_at: float | None = None
    last_movement_at: float | None = None

    while True:
        now = time.monotonic()
        current_pose = _get_head_pose()
        if current_pose is None:
            time.sleep(poll_interval)
            continue

        moving_now = False
        for axis, threshold in thresholds.items():
            delta = current_pose[axis] - original_pose[axis]
            if delta >= threshold:
                direction_seen[axis].add("positive")
                moving_now = True
            elif delta <= -threshold:
                direction_seen[axis].add("negative")
                moving_now = True

        if moving_now:
            if movement_started_at is None:
                movement_started_at = now
            last_movement_at = now

            changed_both_directions = any(
                {"positive", "negative"}.issubset(directions)
                for directions in direction_seen.values()
            )
            sustained = now - movement_started_at >= min_change_seconds
            if sustained and changed_both_directions:
                print(_play_emotion(random.choice(emotions_list)))
                _wait_for_moves_to_finish(timeout=10.0, sleep=0.1)
                _go_to()
                _wait_for_moves_to_finish(timeout=10.0, sleep=0.1)

                # Reset baseline/state and continue waiting for a new deviation.
                latest_pose = _get_head_pose()
                if latest_pose is not None:
                    original_pose = latest_pose
                movement_started_at = None
                last_movement_at = None
                direction_seen = {axis: set() for axis in thresholds}
                time.sleep(poll_interval)
                continue

        elif (
            movement_started_at is not None
            and last_movement_at is not None
            and (now - last_movement_at) >= quiet_reset_seconds
        ):
            movement_started_at = None
            last_movement_at = None
            direction_seen = {axis: set() for axis in thresholds}

        time.sleep(poll_interval)

def main() -> None:
    with ReachyMini() as mini:
        _go_to()
        print(wait_for_headpat())


if __name__ == "__main__":
    main()
