"""FastMCP quickstart example.

Run from the repository root:
    uv run examples/snippets/servers/fastmcp_quickstart.py
"""

from contextlib import asynccontextmanager
import controller
from reachy_mini import ReachyMini
import shutil
from fastmcp import FastMCP
from tools import register_robot_tools

# Set by lifespan when the server starts; tools resolve it via getter so registration can happen before run().
mini = None


@asynccontextmanager
async def lifespan(server):
    global mini
    controller._IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    with ReachyMini() as m:
        mini = m
        yield
    if controller._IMAGES_DIR.is_dir():
        try:
            shutil.rmtree(controller._IMAGES_DIR)
        except OSError:
            pass


def main():
    mcp = FastMCP("Reachy Mini Robot", lifespan=lifespan)
    register_robot_tools(mcp, lambda: mini)
    mcp.run(transport="streamable-http", port=5000)


if __name__ == "__main__":
    main()