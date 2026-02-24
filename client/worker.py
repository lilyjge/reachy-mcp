"""
Run the agent in a separate process so it has its own event loop and no contention
with the main process (uvicorn). Invoked by process.launch_process() via subprocess.

Modes:
- One-shot (default): Env WORKER_ID, WORKER_SYSTEM_PROMPT; run one task, callback, exit.
- Pool: Env WORKER_POOL=1, CALLBACK_URL; read tasks from stdin (one JSON object per line:
  {"worker_id": "...", "system_prompt": "...", "mcp_servers": ["reachy-mini", ...]}), run each, callback, repeat until stdin closes.

Env (one-shot): WORKER_ID, WORKER_SYSTEM_PROMPT
Env (optional): CALLBACK_URL (default http://localhost:8765/event),
                WORKER_MCP_SERVERS (comma-separated MCP server names; default "reachy-mini")
"""
import asyncio
import json
import os
import sys
from client.common import agent_worker
from client.worker_utils import make_agent
import httpx
# Configure anyio to use asyncio backend explicitly before any imports that use it
os.environ.setdefault("ANYIO_BACKEND", "asyncio")

# Set up event loop policy before any event loop is created (same as __main__.py)
if sys.platform == "win32":
    def _make_connection_reset_safe_policy():
        base_policy = asyncio.DefaultEventLoopPolicy()

        class SafePolicy(asyncio.DefaultEventLoopPolicy):
            def new_event_loop(self):
                loop = base_policy.new_event_loop()
                default_handler = loop.default_exception_handler

                def handler(loop, context):
                    exc = context.get("exception")
                    if isinstance(exc, ConnectionResetError):
                        return
                    default_handler(context)

                loop.set_exception_handler(handler)
                return loop

        return SafePolicy()

    asyncio.set_event_loop_policy(_make_connection_reset_safe_policy())


def _run_one_task(worker_id: str, system_prompt: str, mcp_servers: list[str], callback_url: str) -> bool:
    """Run a single agent task and POST callback. Returns success."""
    agent = make_agent(system_prompt, mcp_servers=mcp_servers)
    first_message = (
        "Complete the assigned task to the best of your ability. "
        "If the task cannot be completed, be absolutely sure to have thought really hard and exhausted all possibilities before reporting to the user. "
        "Begin your task."
    )
    result, success = agent_worker(agent, first_message, [])
    task_preview = (system_prompt[:200] + "…") if len(system_prompt) > 200 else system_prompt
    task_ctx = (
        f"Task was successful. System prompt: {task_preview}. Agent output: {result.output}"
        if success
        else f"Task was not successful. System prompt: {task_preview}. Error: {result.output}"
    )
    payload = {"worker_id": worker_id, "message": task_ctx, "done": success}
    try:
        httpx.post(callback_url, json=payload, timeout=10.0).raise_for_status()
    except httpx.HTTPError as e:
        print(f"Callback failed: {e}", file=sys.stderr)
    return success


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    callback_url = os.environ.get("CALLBACK_URL", "http://localhost:8765/event")

    if os.environ.get("WORKER_POOL", "").strip().lower() in ("1", "true"):
        # Pool mode: read tasks from stdin (one JSON line per task)
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    task = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Invalid task JSON: {e}", file=sys.stderr)
                    continue
                worker_id = task.get("worker_id")
                system_prompt = task.get("system_prompt", "")
                mcp_servers = task.get("mcp_servers") or ["reachy-mini"]
                if not worker_id:
                    print("Task missing worker_id", file=sys.stderr)
                    continue
                _run_one_task(worker_id, system_prompt, mcp_servers, callback_url)
        except KeyboardInterrupt:
            pass  # exit gracefully when parent is interrupted (e.g. Ctrl+C)
        sys.exit(0)

    # One-shot mode
    worker_id = os.environ.get("WORKER_ID")
    system_prompt = os.environ.get("WORKER_SYSTEM_PROMPT", "")
    mcp_servers_env = os.environ.get("WORKER_MCP_SERVERS", "")
    if mcp_servers_env:
        mcp_servers = [name.strip() for name in mcp_servers_env.split(",") if name.strip()]
    else:
        mcp_servers = ["reachy-mini"]

    success = _run_one_task(worker_id, system_prompt, mcp_servers, callback_url)
    sys.exit(0 if success else 1)
