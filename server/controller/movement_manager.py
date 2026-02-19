"""Movement manager with breathing idle behavior.

Manages continuous breathing animation when the robot is idle.
"""

from __future__ import annotations
import time
import logging
import threading
from typing import Tuple
from queue import Empty, Queue

import numpy as np
from numpy.typing import NDArray

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.motion.move import Move
from .breathing import BreathingMove


logger = logging.getLogger(__name__)

# Configuration constants
CONTROL_LOOP_FREQUENCY_HZ = 100.0  # Hz - Target frequency for the movement control loop


class MovementManager:
    """Coordinate moves with idle breathing behavior.

    Responsibilities:
    - Run a real-time loop that manages the robot's idle breathing animation
    - Automatically start breathing after idle_inactivity_delay when no moves are active
    - Provide thread-safe APIs for controlling robot activity

    Timing:
    - All time calculations use time.monotonic() for stability
    - Loop attempts 100 Hz updates

    Concurrency:
    - External threads communicate via _command_queue
    - Thread-safe activity tracking and control
    """

    def __init__(self, current_robot: ReachyMini):
        """Initialize movement manager."""
        self.current_robot = current_robot

        # Single timing source for durations
        self._now = time.monotonic

        # Movement state
        self._last_activity_time = self._now()
        self._current_move: Move | None = None
        self._move_start_time: float | None = None
        
        # Track last pose for smooth transitions
        neutral_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self._last_head_pose = neutral_pose
        self._last_antennas: Tuple[float, float] = (0.0, 0.0)

        # Configuration
        self.idle_inactivity_delay = 2.0  # seconds before starting breathing
        self.target_frequency = CONTROL_LOOP_FREQUENCY_HZ
        self.target_period = 1.0 / self.target_frequency

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._breathing_active = False

        # Cross-thread signalling
        self._command_queue: "Queue[Tuple[str, any]]" = Queue()
        self._shared_state_lock = threading.Lock()
        self._shared_last_activity_time = self._last_activity_time

    def start(self) -> None:
        """Start the movement manager loop in a thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()
        logger.info("Movement manager started with breathing enabled")

    def stop(self) -> None:
        """Stop the movement manager loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("Movement manager stopped")

    def mark_activity(self) -> None:
        """Mark the robot as active, preventing idle breathing."""
        with self._shared_state_lock:
            self._shared_last_activity_time = self._now()

    def is_idle(self) -> bool:
        """Return True when the robot has been inactive longer than the idle delay."""
        with self._shared_state_lock:
            last_activity = self._shared_last_activity_time

        return self._now() - last_activity >= self.idle_inactivity_delay

    def _poll_commands(self) -> None:
        """Process queued commands from other threads."""
        while True:
            try:
                command, payload = self._command_queue.get_nowait()
            except Empty:
                break
            self._handle_command(command, payload)

    def _handle_command(self, command: str, payload: any) -> None:
        """Handle a command from the queue."""
        if command == "mark_activity":
            self._last_activity_time = self._now()
            # If we're breathing, stop it
            if self._breathing_active:
                self._current_move = None
                self._move_start_time = None
                self._breathing_active = False
                logger.debug("Breathing stopped due to activity")

    def _control_loop(self) -> None:
        """Main control loop running at target frequency."""
        logger.debug("Movement manager control loop started")
        
        next_tick = self._now()
        
        while not self._stop_event.is_set():
            current_time = self._now()
            
            # Wait until next tick
            if current_time < next_tick:
                sleep_time = next_tick - current_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                current_time = self._now()
            
            next_tick = current_time + self.target_period

            # Poll commands
            self._poll_commands()

            # Sync activity time
            with self._shared_state_lock:
                self._shared_last_activity_time = self._last_activity_time

            # Check if we should start breathing
            idle_duration = current_time - self._last_activity_time
            should_breathe = idle_duration >= self.idle_inactivity_delay

            if should_breathe and not self._breathing_active:
                # Start breathing
                self._start_breathing()
            elif not should_breathe and self._breathing_active:
                # Stop breathing (activity detected elsewhere)
                self._stop_breathing()

            # Update current move (breathing) if active
            if self._current_move is not None and self._move_start_time is not None:
                elapsed = current_time - self._move_start_time
                try:
                    result = self._current_move.evaluate(elapsed)
                    if result is not None:
                        head_pose, antennas, body_yaw = result
                        
                        # Store for smooth transitions
                        if head_pose is not None:
                            self._last_head_pose = head_pose
                        if antennas is not None:
                            self._last_antennas = (float(antennas[0]), float(antennas[1]))
                        
                        # Send to robot
                        self.current_robot.set_target(
                            head=head_pose,
                            antennas=antennas,
                            body_yaw=body_yaw
                        )
                except Exception as e:
                    logger.error(f"Error evaluating breathing move: {e}", exc_info=True)

        logger.debug("Movement manager control loop exited")

    def _start_breathing(self) -> None:
        """Start the breathing animation."""
        logger.info("Starting breathing animation")
        
        # Create breathing move starting from current pose
        self._current_move = BreathingMove(
            interpolation_start_pose=self._last_head_pose,
            interpolation_start_antennas=self._last_antennas,
            interpolation_duration=1.0
        )
        self._move_start_time = self._now()
        self._breathing_active = True

    def _stop_breathing(self) -> None:
        """Stop the breathing animation."""
        logger.debug("Stopping breathing animation")
        self._current_move = None
        self._move_start_time = None
        self._breathing_active = False
