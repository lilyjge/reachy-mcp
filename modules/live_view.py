"""Live camera viewer for Reachy Mini.

Displays the camera feed in a window. Press 'q' to quit.

Note: The daemon must be running before executing this script.

Usage:
    python modules/live_view.py
    python modules/live_view.py --backend gstreamer
"""

import argparse
import time

import cv2
from reachy_mini import ReachyMini


def main() -> None:
    """Show a live camera feed from Reachy Mini."""
    with ReachyMini(media_backend="gstreamer") as mini:
        # Wait for the first frame
        frame = mini.media.get_frame()
        start_time = time.time()
        while frame is None:
            if time.time() - start_time > 20:
                print("Timeout: Failed to grab frame within 20 seconds.")
                exit(1)
            print("Waiting for camera...")
            frame = mini.media.get_frame()
            time.sleep(1)

        print("Live view started. Press 'q' to exit.")
        try:
            while True:
                # time.sleep(0.5)
                frame = mini.media.get_frame()
                if frame is not None:
                    cv2.imshow("Reachy Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cv2.destroyAllWindows()
            print("Live view stopped.")


if __name__ == "__main__":
    main()
