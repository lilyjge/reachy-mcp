"""
Kernel process server: MCP tools to launch worker agents.

Workers run in a separate subprocess (python -m client.worker) so the agent has its own
event loop and no contention with the main process (uvicorn). That avoids the long delay
between the agent calling a tool and the MCP server (5001) receiving the request.

We keep 1-2 pool workers running; launch_process prefers an idle pool worker and only
spawns a one-off subprocess when all pool workers are busy.
"""
import json
import logging
import os
from client.kernel_utils import validated_mcp_servers
import subprocess
import sys
import uuid
from fastmcp import FastMCP

logger = logging.getLogger(__name__)

callback_url = "http://localhost:8765/event"
POOL_SIZE = 2

# Pool: long-lived workers that accept tasks via stdin. Each slot: process, stdin pipe, busy, worker_id.
_pool_slots: list[dict] = []
# One-off workers (worker_id -> Popen) when pool is full
_worker_processes: dict[str, subprocess.Popen] = {}
_worker_system_prompts: dict[str, str] = {}
mcp = FastMCP("rosaOS Kernel")


def _ensure_pool() -> None:
    """Start pool workers up to POOL_SIZE; remove dead ones."""
    global _pool_slots
    # Drop dead slots
    _pool_slots = [s for s in _pool_slots if s["process"].poll() is None]
    while len(_pool_slots) < POOL_SIZE:
        env = os.environ.copy()
        env["WORKER_POOL"] = "1"
        env["CALLBACK_URL"] = callback_url
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "client.worker"],
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=sys.stderr,
                text=False,
            )
            _pool_slots.append({
                "process": proc,
                "stdin": proc.stdin,
                "busy": False,
                "worker_id": None,
            })
            logger.debug("Started pool worker %s", len(_pool_slots))
        except Exception as e:
            logger.warning("Failed to start pool worker: %s", e)
            break


def _dispatch_to_pool(worker_id: str, system_prompt: str, mcp_servers: list[str]) -> bool:
    """Send task to an idle pool worker. Returns True if dispatched, False if no idle worker."""
    _ensure_pool()
    for slot in _pool_slots:
        if not slot["busy"]:
            task = {
                "worker_id": worker_id,
                "system_prompt": system_prompt,
                "mcp_servers": mcp_servers,
            }
            try:
                slot["stdin"].write((json.dumps(task) + "\n").encode())
                slot["stdin"].flush()
                slot["busy"] = True
                slot["worker_id"] = worker_id
                _worker_system_prompts[worker_id] = system_prompt
                return True
            except (BrokenPipeError, OSError) as e:
                logger.warning("Pool worker write failed: %s", e)
                slot["busy"] = False
                slot["worker_id"] = None
                _ensure_pool()
                continue
    return False


def mark_worker_done(worker_id: str) -> str:
    """Called when a worker finishes (callback received). Frees pool slot or removes one-off. Returns system prompt of the worker."""
    for slot in _pool_slots:
        if slot.get("worker_id") == worker_id:
            slot["busy"] = False
            slot["worker_id"] = None
            prompt = _worker_system_prompts.pop(worker_id, None)
            logger.debug("Pool slot freed for worker_id=%s", worker_id)
            return prompt
    _worker_processes.pop(worker_id, None)
    return _worker_system_prompts.pop(worker_id, None)


@mcp.tool(description=f"""Spawn a background worker that runs an LLM agent with the given system prompt.

    The worker runs in a separate subprocess with access to the MCP tools (e.g. robot
    at 5001). When the worker completes, it POSTs to the client callback. Use the
    returned worker_id to correlate the callback.

    Args:
        system_prompt: Instructions for the worker agent (e.g. surveillance task, patrol, etc.).
        robots: List of robots for the worker to have access to through their MCP servers.
            Must be one of the following: {", ".join([name for name in validated_mcp_servers.keys()])}

    Returns:
        worker_id: The worker will include this in its completion callback.
    """)
def launch_process(system_prompt: str, robots: list[str] = ["reachy-mini"]) -> str:
    """Spawn a background worker that runs an LLM agent with the given system prompt.

    Prefers an idle pool worker (1-2 long-lived subprocesses); if both are busy, spawns a one-off subprocess.
    When the worker completes, it POSTs to the client callback. Use the returned worker_id to correlate the callback.

    Args:
        system_prompt: Instructions for the worker agent (e.g. surveillance task, patrol, etc.).
        robots: List of robots for the worker to have access to through their MCP servers.

    Returns:
        worker_id: The worker will include this in its completion callback.
    """
    worker_id = str(uuid.uuid4())
    valid_robots: list[str] = []
    for name in robots:
        if name in validated_mcp_servers:
            valid_robots.append(name)
        else:
            logger.warning("Requested robot %s is not a known MCP server; ignoring", name)
    if not valid_robots:
        valid_robots = ["reachy-mini"]

    try:
        if _dispatch_to_pool(worker_id, system_prompt, valid_robots):
            logger.info("Dispatched to pool worker_id=%s", worker_id)
            return "Worker launched successfully (pool). Worker ID: " + worker_id
    except Exception as e:
        logger.warning("Pool dispatch failed, falling back to one-off: %s", e)

    logger.info("Launching one-off worker subprocess worker_id=%s", worker_id)
    try:
        env = os.environ.copy()
        env["WORKER_ID"] = worker_id
        env["WORKER_SYSTEM_PROMPT"] = system_prompt
        env["CALLBACK_URL"] = callback_url
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

@mcp.tool()
def get_processes() -> list[dict]:
    """Get the list and tasks of all currently running processes."""
    return [{"worker_id": worker_id, "system_prompt": system_prompt} for worker_id, system_prompt in _worker_system_prompts.items()]

@mcp.tool()
def terminate_process(worker_id: str, force: bool = False) -> str:
    """
    Terminate the process with the given worker_id.
    Args:
        worker_id: The ID of the process to terminate.
        force: If True, the process will be terminated even if it is a pool worker.
            Keep this False in most cases unless it is absolutely necessary.
    Returns:
        A message indicating the result of the termination.
    """
    # Check pool slots first (task running in a long-lived worker)
    for i, slot in enumerate(_pool_slots):
        if slot.get("worker_id") == worker_id:
            if not force:
                return (
                    "This task is running in a pool worker (long-lived process). "
                    "Terminating the task would be inefficient."
                    "Use force=True to terminate the task anyway."
                )
            prompt = _worker_system_prompts.pop(worker_id, "") or "(pool task)"
            try:
                slot["process"].terminate()
            except OSError:
                pass
            _pool_slots.pop(i)
            logger.info("Terminated pool worker task worker_id=%s", worker_id)
            preview = (prompt[:200] + "…") if len(prompt) > 200 else (prompt or "?")
            return f"Process in pool worker with the task '{preview}' terminated successfully."
    # One-off subprocess
    if worker_id not in _worker_processes:
        return "Process not found."
    prompt = _worker_system_prompts.get(worker_id, "")
    try:
        _worker_processes[worker_id].terminate()
    except OSError:
        pass
    mark_worker_done(worker_id)
    logger.info("Terminated one-off worker worker_id=%s", worker_id)
    preview = (prompt[:200] + "…") if prompt and len(prompt) > 200 else (prompt or "?")
    return f"Process with the task '{preview}' terminated successfully."

def start_process_server() -> None:
    logger.info("Starting process manager MCP server")
    _ensure_pool()
    mcp.run(transport="streamable-http", port=7001)