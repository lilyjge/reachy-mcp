"""FastMCP quickstart example.

Run from the repository root:
    uv run examples/snippets/servers/fastmcp_quickstart.py
"""

from contextlib import asynccontextmanager
import controller
from controller.stt_loop import start_stt_loop
from controller.movement_manager import MovementManager
from reachy_mini import ReachyMini
import shutil
from fastmcp import FastMCP
from tools import register_background_tools, register_robot_tools

# Set by lifespan when the server starts; tools resolve it via getter so registration can happen before run().
mini = None
_stt_stop = None
_stt_thread = None
_movement_manager = None


@asynccontextmanager
async def lifespan(server):
    global mini, _stt_stop, _stt_thread, _movement_manager
    controller._IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    with ReachyMini() as m:
        mini = m
        
        # Initialize and start movement manager for breathing
        _movement_manager = MovementManager(mini)
        _movement_manager.start()
        
        # Set the movement manager in controls and vision so tools can mark activity
        from controller.controls import set_movement_manager
        from controller.vision import set_movement_manager as set_vision_movement_manager
        set_movement_manager(_movement_manager)
        set_vision_movement_manager(_movement_manager)
        
        _stt_thread, _stt_stop = start_stt_loop(mini)
        try:
            yield
        finally:
            # Stop movement manager
            if _movement_manager is not None:
                _movement_manager.stop()
            
            # Stop STT loop before closing ReachyMini, then join briefly
            if _stt_stop is not None:
                _stt_stop.set()
            if _stt_thread is not None:
                _stt_thread.join(timeout=3.0)
    if controller._IMAGES_DIR.is_dir():
        try:
            shutil.rmtree(controller._IMAGES_DIR)
        except OSError:
            pass


def main():
    mcp = FastMCP("Reachy Mini Robot", lifespan=lifespan)
    register_robot_tools(mcp, lambda: mini)
    register_background_tools(mcp)
    mcp.run(transport="streamable-http", port=5001)


if __name__ == "__main__":
    main()