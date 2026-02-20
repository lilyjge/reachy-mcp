"""
Run the agent in a separate process so it has its own event loop and no contention
with the main process (uvicorn). Invoked by process.launch_process() via subprocess.

Env (required): WORKER_ID, WORKER_SYSTEM_PROMPT
Env (optional): CALLBACK_URL (default http://localhost:8765/event),
                 MCP_SERVER_URL (default http://localhost:5001/mcp)
"""
import asyncio
import os
import sys
from client.utils import _make_agent, _agent_worker
import httpx
# Configure anyio to use asyncio backend explicitly before any imports that use it
# This ensures anyio's cancel scopes work correctly with the event loop
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

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    worker_id = os.environ.get("WORKER_ID")
    system_prompt = os.environ.get("WORKER_SYSTEM_PROMPT", "")
    callback_url = os.environ.get("CALLBACK_URL", "http://localhost:8765/event")

    if not worker_id:
        print("WORKER_ID env required", file=sys.stderr)
        sys.exit(1)

    agent = _make_agent(system_prompt)
    first_message = (
        "Complete the assigned task to the best of your ability. "
        "If the task cannot be completed, be absolutely sure to have thought really hard and exhausted all possibilities before reporting to the user. "
        "Begin your task."
    )
    result, success = _agent_worker(agent, first_message, [])
    task_preview = (system_prompt[:100] + "…") if len(system_prompt) > 100 else system_prompt
    task_ctx = f"Task was successful. System prompt: {task_preview}. Agent output: {result.output}" if success else f"Task was not successful. System prompt: {task_preview}. Error: {result.output}"
    payload = {"worker_id": worker_id, "message": task_ctx, "done": success}
    try:
        httpx.post(callback_url, json=payload, timeout=10.0).raise_for_status()
    except httpx.HTTPError as e:
        print(f"Callback failed: {e}", file=sys.stderr)
    sys.exit(0 if success else 1)
