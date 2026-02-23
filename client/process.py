"""
Kernel process server: MCP tools to launch worker agents.

Workers run in a separate subprocess (python -m client.worker) so the agent has its own
event loop and no contention with the main process (uvicorn). That avoids the long delay
between the agent calling a tool and the MCP server (5001) receiving the request.

Using subprocess instead of multiprocessing avoids importing process.py (which has FastMCP)
in the worker process, giving a cleaner environment.
"""
import logging
import os
from client.utils import all_mcp_servers
import subprocess
import sys
import uuid
from fastmcp import FastMCP

logger = logging.getLogger(__name__)

callback_url = "http://localhost:8765/event"
_worker_processes: dict[str, subprocess.Popen] = {}
_worker_system_prompts: dict[str, str] = {}
mcp = FastMCP("rosaOS Kernel")


@mcp.tool(description=f"""Spawn a background worker that runs an LLM agent with the given system prompt.

    The worker runs in a separate subprocess with access to the MCP tools (e.g. robot
    at 5001). When the worker completes, it POSTs to the client callback. Use the
    returned worker_id to correlate the callback.

    Args:
        system_prompt: Instructions for the worker agent (e.g. surveillance task, patrol, etc.).
        robots: List of robots for the worker to have access to through their MCP servers.
            Must be one of the following: {", ".join([name for name in all_mcp_servers.keys()])}

    Returns:
        worker_id: The worker will include this in its completion callback.
    """)
def launch_process(system_prompt: str, robots: list[str] = ["reachy-mini"]) -> str:
    """Spawn a background worker that runs an LLM agent with the given system prompt.

    The worker runs in a separate subprocess with access to the MCP tools (e.g. robot
    at 5001). When the worker completes, it POSTs to the client callback. Use the
    returned worker_id to correlate the callback.

    Args:
        system_prompt: Instructions for the worker agent (e.g. surveillance task, patrol, etc.).
        robots: List of robots for the worker to have access to through their MCP servers.

    Returns:
        worker_id: The worker will include this in its completion callback.
    """
    worker_id = str(uuid.uuid4())
    logger.info("Launching worker subprocess worker_id=%s system_prompt=%s", worker_id, system_prompt)

    try:
        env = os.environ.copy()
        env["WORKER_ID"] = worker_id
        env["WORKER_SYSTEM_PROMPT"] = system_prompt
        env["CALLBACK_URL"] = callback_url

        # Pass through the MCP server (robot) names this worker should have access to.
        valid_robots: list[str] = []
        for name in robots:
            if name in all_mcp_servers:
                valid_robots.append(name)
            else:
                logger.warning("Requested robot %s is not a known MCP server; ignoring", name)
        if not valid_robots:
            valid_robots = ["reachy-mini"]
        env["WORKER_MCP_SERVERS"] = ",".join(valid_robots)

        _worker_processes[worker_id] = subprocess.Popen(
            [sys.executable, "-m", "client.worker"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=sys.stderr,
        )
        _worker_system_prompts[worker_id] = system_prompt
        logger.debug("Worker subprocess started worker_id=%s", worker_id)
        return "Worker launched successfully. Worker ID: " + worker_id
    except Exception as e:
        logger.exception("Failed to launch worker worker_id=%s: %s", worker_id, e)
        return "Failed to launch worker: " + str(e)

def start_process_server() -> None:
    mcp.run(transport="streamable-http", port=7001)