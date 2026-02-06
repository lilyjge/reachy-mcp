from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import numpy as np
import time

with ReachyMini() as mini:
    #Move everything at once
    mini.goto_target(
        head=create_head_pose(z=10, mm=True),    # Up 10mm
        antennas=np.deg2rad([45, 45]),           # Antennas out
        body_yaw=np.deg2rad(30),                 # Turn body
        duration=2.0,                            # Take 2 seconds
        method="minjerk"                         # Smooth acceleration
    )

    mini.goto_target(
        head=create_head_pose(z=0, mm=True),    # Up 10mm
        antennas=np.deg2rad([0, 0]),           # Antennas out
        body_yaw=np.deg2rad(0),                 # Turn body
        duration=3.0,                            # Take 2 seconds
        method="cartoon"                         # Smooth acceleration
    )
    time.sleep(3)

    import cv2
    frame = mini.media.get_frame()
    print(frame)
    print(cv2.imwrite("reachy2.jpg", frame))