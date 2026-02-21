"""Automatic head tracking loop that activates after eye contact.

Similar to the STT loop, this runs continuously in the background:
1. Waits for eye contact with the user
2. Starts tracking the user's head movements
3. Stops when activity is detected (agent spawns process or calls tools)
4. Returns to waiting for eye contact

This makes the robot feel natural and responsive without requiring explicit tool calls.
"""

from __future__ import annotations
import time
import logging
import threading
from typing import Optional
import numpy as np
import cv2

from reachy_mini import ReachyMini
from .vision import wait_for_eye_contact

logger = logging.getLogger(__name__)

# Load OpenCV Haar cascades for face detection
_face_cascade = None

def _get_face_cascade():
    """Lazy-load OpenCV face cascade."""
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_cascade


# Tracking parameters
HEAD_TRACKING_FPS = 20  # Hz - Update rate for head tracking
SMOOTHING_FACTOR = 0.3  # Lower = smoother but more lag (0-1)
HEAD_YAW_SCALE = 35.0  # degrees - Max yaw when head is at edge (side to side)
DEAD_ZONE = 0.05  # Ignore small movements near center (0-1, as fraction of frame)
ACTIVITY_CHECK_INTERVAL = 0.2  # How often to check if activity occurred (seconds)
ACTIVITY_TIMEOUT = 3.0  # If activity within this time, stop tracking (seconds)


def _detect_head_position(frame: np.ndarray) -> Optional[tuple[float, float]]:
    """Detect head position in the frame using OpenCV face detection.
    
    Returns:
        (x, y) normalized position (-1 to 1, 0 is center) or None if no face detected
    """
    face_cascade = _get_face_cascade()
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame.shape[:2]
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        return None
    
    # Use the largest face (closest to camera)
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    
    # Calculate face center
    face_center_x = x + fw / 2
    face_center_y = y + fh / 2
    
    # Convert to normalized coordinates (-1 to 1, center is 0)
    # Invert y because image coords are top-down but we want bottom-up
    norm_x = (face_center_x / w - 0.5) * 2.0  # 0-1 -> -1 to 1
    norm_y = -(face_center_y / h - 0.5) * 2.0  # 0-1 -> -1 to 1 (inverted)
    
    return (norm_x, norm_y)


def _apply_dead_zone(value: float) -> float:
    """Apply dead zone to ignore small movements near center."""
    if abs(value) < DEAD_ZONE:
        return 0.0
    # Scale to maintain smooth motion outside dead zone
    sign = 1.0 if value > 0 else -1.0
    scaled = (abs(value) - DEAD_ZONE) / (1.0 - DEAD_ZONE)
    return sign * scaled


def _calculate_target_angles(norm_x: float, smooth_x: float) -> tuple[float, float]:
    """Convert normalized head position to robot target angles.
    
    Args:
        norm_x: Normalized horizontal position (-1 to 1)
        smooth_x: Previous smoothed x
        
    Returns:
        (yaw, new_smooth_x) in degrees and smoothed value
    """
    # Apply dead zone
    norm_x = _apply_dead_zone(norm_x)
    
    # Apply smoothing (only x-axis for yaw)
    smooth_x = SMOOTHING_FACTOR * norm_x + (1 - SMOOTHING_FACTOR) * smooth_x
    
    # Convert to angles - only yaw (side to side), no pitch (up/down)
    # Positive X (right) -> positive yaw (turn right)
    yaw = smooth_x * HEAD_YAW_SCALE
    
    return (yaw, smooth_x)


def _update_robot_pose(mini: ReachyMini, yaw: float) -> None:
    """Update robot head position to follow the target angles (yaw only)."""
    from reachy_mini.utils import create_head_pose
    
    try:
        # Create target pose with only yaw (side to side), pitch=0 (look straight)
        target_pose = create_head_pose(
            x=0, y=0, z=0,
            roll=0,
            pitch=0,  # Always look straight ahead, no up/down
            yaw=yaw,
            degrees=True
        )
        
        # Send to robot with short duration for smooth following
        mini.goto_target(
            head=target_pose,
            duration=0.2,  # Fast updates for smooth tracking
            method="linear"  # Linear for responsive tracking
        )
        # Note: We don't call mark_activity() here because head tracking
        # IS the idle behavior that replaces breathing. Only tool calls
        # should mark activity to pause both breathing and head tracking.
            
    except Exception as e:
        logger.error(f"Error updating robot pose: {e}")


def _head_tracking_loop(mini: ReachyMini, stop_event: threading.Event, movement_manager) -> None:
    """Main loop: wait for eye contact, track head, stop on activity, repeat."""
    logger.info("Head tracking loop started")
    
    frame_time = 1.0 / HEAD_TRACKING_FPS
    
    while not stop_event.is_set():
        try:
            # Phase 1: Wait for eye contact
            logger.info("Waiting for eye contact to start head tracking...")
            if not wait_for_eye_contact(mini, stop_event):  # Wait until eye contact or stop
                # wait_for_eye_contact returns False if stop_event was set
                break
            
            logger.info("Eye contact detected! Starting head tracking...")
            
            # Notify movement manager that head tracking is active (pauses breathing)
            if movement_manager:
                movement_manager.set_head_tracking_active(True)
            
            # Phase 2: Track head until activity detected
            smooth_x = 0.0
            target_yaw = 0.0
            last_activity_check = time.monotonic()
            next_update = time.monotonic()
            
            # Flush first frame
            _ = mini.media.get_frame()
            
            tracking_active = True
            while tracking_active and not stop_event.is_set():
                current_time = time.monotonic()
                
                # Check for recent activity periodically
                if current_time - last_activity_check >= ACTIVITY_CHECK_INTERVAL:
                    last_activity_check = current_time
                    if movement_manager:
                        time_since_activity = movement_manager.get_time_since_last_activity()
                        # Stop tracking if there's been recent activity (tool calls, speech, etc.)
                        if time_since_activity < ACTIVITY_TIMEOUT:
                            logger.info("Activity detected, stopping head tracking")
                            tracking_active = False
                            break
                
                # Rate limiting
                if current_time < next_update:
                    time.sleep(0.01)
                    continue
                
                next_update = current_time + frame_time
                
                try:
                    # Get frame from robot camera
                    frame = mini.media.get_frame()
                    if frame is None:
                        continue
                    
                    # Detect head position
                    head_pos = _detect_head_position(frame)
                    
                    if head_pos is None:
                        # No face detected - reset to neutral gradually
                        target_yaw *= 0.9
                        if abs(target_yaw) < 1.0:
                            target_yaw = 0.0
                    else:
                        # Calculate target angles from head position (x-axis only for yaw)
                        norm_x, norm_y = head_pos
                        target_yaw, smooth_x = _calculate_target_angles(norm_x, smooth_x)
                    
                    # Update robot pose (yaw only, no pitch)
                    _update_robot_pose(mini, target_yaw)
                    
                except Exception as e:
                    logger.error(f"Error in head tracking frame: {e}")
                    time.sleep(0.1)  # Avoid tight error loop
            
            # Tracking stopped, return head to neutral before waiting for next eye contact
            if movement_manager:
                movement_manager.set_head_tracking_active(False)
            
            if not stop_event.is_set():
                try:
                    from reachy_mini.utils import create_head_pose
                    neutral_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
                    mini.goto_target(head=neutral_pose, duration=0.8, method="minjerk")
                    logger.info("Returned to neutral position")
                    # Wait for movement to complete
                    time.sleep(0.8)
                except Exception as e:
                    logger.error(f"Error returning to neutral: {e}")
                
                # Small delay before next eye contact detection
                time.sleep(0.5)
        
        except Exception as e:
            logger.error(f"Error in head tracking loop: {e}", exc_info=True)
            time.sleep(1.0)  # Avoid rapid error loop
    
    logger.info("Head tracking loop stopped")


def start_head_tracking_loop(mini: ReachyMini, movement_manager) -> tuple[threading.Thread, threading.Event]:
    """Start the head tracking loop in a background thread.
    
    Args:
        mini: ReachyMini instance
        movement_manager: MovementManager instance for activity detection
        
    Returns:
        (thread, stop_event) - Use stop_event.set() to stop the loop
    """
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_head_tracking_loop,
        args=(mini, stop_event, movement_manager),
        daemon=True,
        name="HeadTrackingLoop"
    )
    thread.start()
    return thread, stop_event
