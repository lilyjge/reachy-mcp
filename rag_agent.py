"""
Main Pydantic AI agent for the Reachy Mini robot.

Run normally: starts HTTP server with chat, callback endpoint, and push-on-event.
  python rag_agent.py
  Callback URL is hardcoded in the MCP server (http://localhost:8765/event).

Run as worker (spawned by MCP spawn_background_instance):
  python rag_agent.py --worker --worker-id UUID --system-prompt "Your task..."
  Uses same MCP tools (including callback). Callback instructions are in the first user message, not system prompt.
"""

from __future__ import annotations

import argparse
import queue
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Model and MCP
# ---------------------------------------------------------------------------

model = None
try:
    import httpx
    _local_base_url = "https://localhost:6000/v1"
    # Quick health check: if the local server isn't reachable, fall back to Groq.
    # Many OpenAI-compatible servers expose GET /v1/models.
    _models_url = _local_base_url.rstrip("/") + "/models"
    httpx.get(_models_url, timeout=2.0, verify=False).raise_for_status()
    model = OpenAIResponsesModel(
        "openai/gpt-oss-20b",
        provider=OpenAIProvider(
            base_url=_local_base_url,
            api_key="foo",
        ),
    )
except Exception as e:
    print("error: " + str(e))
    model = GroqModel(
        "openai/gpt-oss-20b",
        provider=GroqProvider(
            api_key=os.environ.get("GROQ_API_KEY"),
        ),
    )

print(model)
mcp_server = MCPServerStreamableHTTP("http://localhost:5001/mcp")
mcp_server2 = MCPServerStreamableHTTP("http://localhost:9090/mcp")


BASE_INSTRUCTIONS = """
You are a LLM controlling a Reachy Mini robot, a friendly and helpful robot.
Use tools to physically interact with the user.
As a LLM, you don't have image capabilities, but the robot have image tools, so work with these tools and image paths.
You are a text only model, so remember that when you call the take picture tool.
Use the describe_image tool to answer questions about images.
Use the analyze face tool to analyze faces in images.
If you are provided with an image with a name, save the image with the name.
If you take a picture and the image matches a named person, save the image with the name too.

You also have access to a TurtleBot4 robot with the following capabilities:
- Navigation: drive forward/backward, rotate, navigate to positions
- Docking: dock to charging station (dock action) and undock from charging station (undock action)
- Sensors: camera feed, LIDAR scans for obstacle detection
- Actions: wall_follow, drive_distance, rotate_angle, drive_arc, and more
You can coordinate actions between the Reachy Mini and TurtleBot to accomplish complex tasks.

When the user asks you to perform a task, think hard about the task and the best way to perform it.
Also think about whether the task should be performed in a background task or not. 
If it should be performed in a background task, use the spawn_background_instance tool to spawn a background task.
When you receive a [Worker callback] message, that is a report from a background task—respond to the user about it (e.g. "I just saw Alice on camera!").
When you receive a [User said] message, the user spoke into the robot's microphone—respond to what they said.
If you want to send a message to the user, always use the speak tool instead of outputting text.
Simulate real conversation flow.
""".strip()


def _make_agent(extra_instructions: str = "") -> Agent:
    instructions = BASE_INSTRUCTIONS
    if extra_instructions:
        instructions = f"{instructions}\n\n{extra_instructions}"
    return Agent(
        model,
        toolsets=[mcp_server, mcp_server2],
        instructions=instructions,
    )


# ---------------------------------------------------------------------------
# Worker mode: argparse, first user message has callback instructions
# ---------------------------------------------------------------------------

def _run_worker(worker_id: str, system_prompt: str) -> None:
    """Run worker task; all stdout/stderr go to the log file from the spawn side."""
    def log(msg: str) -> None:
        print(f"[worker {worker_id[:8]}] {msg}", flush=True)

    log(f"started (cwd={os.getcwd()}, executable={sys.executable})")
    try:
        agent = _make_agent(system_prompt)
        first_message = (
            f"Your worker_id is {worker_id}. When you have completed your task, you MUST call "
            f'the callback tool with your result message, done=True, and worker_id="{worker_id}". '
            "If the task cannot be completed, be absolutely sure to have thought really hard and have exhausted all possibilities and tried everything before calling the callback tool."
            "Then you are finished. Now begin your task."
        )
        log("calling agent.run_sync ...")
        result = agent.run_sync(first_message)
        log(f"result: {result.output or '(no output)'}")
    except TimeoutError as e:
        import traceback
        log(f"timeout error during MCP initialization or task execution: {e}")
        log("attempting to send callback about timeout...")
        # Try to notify about the timeout via callback if possible
        try:
            import httpx
            httpx.post(
                "http://localhost:8765/event",
                json={
                    "worker_id": worker_id,
                    "message": f"Worker timed out during initialization or execution. This may indicate MCP servers are slow/unavailable or the task took too long.",
                    "done": True,
                },
                timeout=5.0,
            )
        except Exception:
            pass
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        import traceback
        log(f"error: {e}")
        # Try to send error callback
        try:
            import httpx
            httpx.post(
                "http://localhost:8765/event",
                json={
                    "worker_id": worker_id,
                    "message": f"Worker encountered error: {type(e).__name__}: {e}",
                    "done": True,
                },
                timeout=5.0,
            )
        except Exception:
            pass
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)


# ---------------------------------------------------------------------------
# Main mode: FastAPI, POST /event runs agent and pushes; GET /updates; POST /chat
# ---------------------------------------------------------------------------

_outgoing_lock = threading.Lock()
_outgoing_messages: list[dict] = []  # pushed when event triggers agent response
_message_history: list = []


def _push_outgoing(role: str, content: str, worker_id: str | None = None, done: bool | None = None) -> None:
    with _outgoing_lock:
        print("pushing outgoing message: " + content)
        _outgoing_messages.append({
            "role": role,
            "content": content,
            "worker_id": worker_id,
            "done": done,
            "at": datetime.now(timezone.utc).isoformat(),
        })


def _pop_outgoing() -> list[dict]:
    """Return all pending outgoing messages and clear the buffer."""
    with _outgoing_lock:
        out = list(_outgoing_messages)
        _outgoing_messages.clear()
        return out


def main_app():
    from fastapi import FastAPI
    from fastapi.responses import FileResponse

    agent = _make_agent()
    app = FastAPI()
    port = int(os.environ.get("RAG_AGENT_PORT", "8765"))

    event_queue: queue.Queue[dict] = queue.Queue()

    def _event_worker() -> None:
        """Background thread: consume events and run agent so we don't block the callback (avoids MCP deadlock)."""
        while True:
            try:
                payload = event_queue.get()
                print("event_queue.get: " + str(payload))
                if payload is None:
                    break
                if payload.get("type") == "speech":
                    worker_message = f"[User said] {payload.get('text', '')}"
                else:
                    worker_id = payload.get("worker_id", "?")
                    message = payload.get("message", "")
                    done = payload.get("done", False)
                    worker_message = f"[Worker callback] {message} (worker_id={worker_id}, done={done}). Inform the user."
                try:
                    print("running agent with message: " + worker_message)
                    result = agent.run_sync(worker_message, message_history=_message_history)
                    _message_history.clear()
                    _message_history.extend(result.all_messages())
                    output = (result.output or "").strip()
                    _push_outgoing("model", output, worker_id=payload.get("worker_id"), done=payload.get("done"))
                except Exception as e:
                    import traceback
                    error_msg = f"[Error processing event] {type(e).__name__}: {e}"
                    print(f"Agent error: {error_msg}")
                    traceback.print_exc()
                    _push_outgoing("model", error_msg, worker_id=payload.get("worker_id"), done=payload.get("done"))
            except Exception as e:
                import traceback
                print(f"Event worker error: {type(e).__name__}: {e}")
                traceback.print_exc()

    _event_thread = threading.Thread(target=_event_worker, daemon=True)
    _event_thread.start()

    @app.post("/event")
    def post_event(payload: dict):
        """Receive callback from background workers. Queues the event and returns immediately."""
        event_queue.put_nowait(payload)
        return {"ok": True}

    @app.post("/stt")
    def post_stt(payload: dict):
        """Receive transcribed speech from the robot's mic (server STT loop). Queues as user speech and runs the agent; response is pushed to /updates."""
        text = (payload.get("text") or "").strip()
        print("received transcribed speech: " + text)
        if not text:
            return {"ok": False, "error": "missing or empty text"}
        event_queue.put_nowait({"type": "speech", "text": text})
        return {"ok": True}

    @app.get("/updates")
    def get_updates():
        """Return new outgoing messages (e.g. agent responses triggered by worker callbacks) and clear the queue."""
        return {"messages": _pop_outgoing()}

    @app.post("/chat")
    def post_chat(payload: dict):
        """Send a message and get a response. Body: {"prompt": "..."}."""
        prompt = payload.get("prompt", "")
        result = agent.run_sync(prompt, message_history=_message_history)
        _message_history.clear()
        _message_history.extend(result.all_messages())
        return {"output": result.output}

    # Simple chat UI
    _ui_path = Path(__file__).resolve().parent / "client_ui.html"

    @app.get("/")
    def index():
        if _ui_path.is_file():
            return FileResponse(_ui_path, media_type="text/html")
        return {"message": "Rag agent. POST /chat with {\"prompt\": \"...\"}, GET /updates for event-driven messages."}

    return app, port


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reachy Mini rag agent (main or worker).")
    parser.add_argument("--worker", action="store_true", help="Run as background worker")
    parser.add_argument("--worker-id", default=os.environ.get("RAG_AGENT_WORKER_ID"), help="Worker ID (required in worker mode)")
    parser.add_argument("--system-prompt", default=os.environ.get("RAG_AGENT_WORKER_SYSTEM_PROMPT", ""), help="Worker task instructions")
    parser.add_argument("--system-prompt-file", help="Read system prompt from file (overrides --system-prompt)")
    args = parser.parse_args()

    if args.worker:
        wid = args.worker_id
        prompt = args.system_prompt
        if args.system_prompt_file:
            path = Path(args.system_prompt_file)
            if path.is_file():
                prompt = path.read_text(encoding="utf-8").strip()
        if not wid:
            print("Error: --worker-id required in worker mode", file=sys.stderr)
            sys.exit(1)
        _run_worker(wid, prompt)
        raise SystemExit(0)

    app, port = main_app()
    import uvicorn
    uvicorn.run(app, port=port)
