"""FastMCP quickstart example.

Run from the repository root:
    uv run examples/snippets/servers/fastmcp_quickstart.py
"""

from contextlib import asynccontextmanager
from . import controller
from .controller.stt_loop import start_stt_loop
from reachy_mini import ReachyMini
import shutil
from fastmcp import FastMCP
from .robot import register_robot_tools
import os
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


def main(sim: bool = False, tts_elevenlabs: bool = False, tts_voice: str | None = None) -> None:
    port = int(os.environ.get("REACHY_MCP_PORT", "5001"))

    # Configure TTS engine/voice for the controller TTS subprocess.
    if tts_voice:
        os.environ["TTS_VOICE"] = tts_voice
    os.environ.setdefault("TTS_VOICE", os.environ.get("TTS_VOICE", "autumn"))

    if tts_elevenlabs:
        os.environ["TTS_ENGINE"] = "elevenlabs"
    else:
        os.environ.setdefault("TTS_ENGINE", "groq")

    if sim:
        # Sim mode: no STT loop or camera lifespan
        with ReachyMini() as m:
            mcp = FastMCP("Reachy Mini Robot")
            register_robot_tools(mcp, lambda: m)
            mcp.run(transport="streamable-http", port=port, stateless_http=True)
    else:
        # Real hardware mode: use lifespan with STT loop and camera
        mcp = FastMCP("Reachy Mini Robot", lifespan=lifespan)
        register_robot_tools(mcp, lambda: mini)
        mcp.run(transport="streamable-http", port=port, stateless_http=True)