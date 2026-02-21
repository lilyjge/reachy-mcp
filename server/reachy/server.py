"""FastMCP quickstart example.

Run from the repository root:
    uv run examples/snippets/servers/fastmcp_quickstart.py
"""

from contextlib import asynccontextmanager
from . import controller
from .controller.stt_loop import start_stt_loop
from reachy_mini import ReachyMini
import argparse
import shutil
from fastmcp import FastMCP
from .robot import register_robot_tools
# Set by lifespan when the server starts; tools resolve it via getter so registration can happen before run().
mini = None
_stt_stop = None
_stt_thread = None


@asynccontextmanager
async def lifespan(server):
    global mini, _stt_stop, _stt_thread
    controller._IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    with ReachyMini() as m:
        mini = m
        _stt_thread, _stt_stop = start_stt_loop(mini)
        try:
            yield
        finally:
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
    parser = argparse.ArgumentParser(description="Reachy Mini MCP Server")
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Run with sim (no STT/camera lifespan)",
    )
    args = parser.parse_args()

    if args.sim:
        # Sim mode: no STT loop or camera lifespan
        with ReachyMini() as m:
            mcp = FastMCP("Reachy Mini Robot")
            register_robot_tools(mcp, lambda: m)
            mcp.run(transport="streamable-http", port=5001, stateless_http=True)
    else:
        # Real hardware mode: use lifespan with STT loop and camera
        mcp = FastMCP("Reachy Mini Robot", lifespan=lifespan)
        register_robot_tools(mcp, lambda: mini)
        mcp.run(transport="streamable-http", port=5001, stateless_http=True)


if __name__ == "__main__":
    main()