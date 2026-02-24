"""
Main Pydantic AI agent for the Reachy Mini robot.
"""

import queue
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from client.utils import _agent_worker, KERNEL_INSTRUCTIONS, model
from client.process import mark_worker_done
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP

_outgoing_lock = threading.Lock()
_outgoing_messages: list[dict] = []  # pushed when event triggers agent response
_message_history: list = []

# For chat client interfaces
def _push_outgoing(role: str, content: str, worker_id: str | None = None, done: bool | None = None) -> None:
    with _outgoing_lock:
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
    app = FastAPI()
    port = int(os.environ.get("RAG_AGENT_PORT", "8765"))

    event_queue: queue.Queue[dict] = queue.Queue()

    def _event_worker() -> None:
        """Single dedicated thread: consume events and run agent. The agent is created in this
        thread so all asyncio/anyio state (locks, event loop) lives in one thread, avoiding
        'bound to a different event loop' and cancel-scope errors when using run_sync().
        """
        agent = Agent(
            model,
            toolsets=[MCPServerStreamableHTTP("http://localhost:7001/mcp")],
            instructions=KERNEL_INSTRUCTIONS,
            retries=10,
        )
        while True:
            try:
                payload = event_queue.get()
                print("event_queue.get: " + str(payload))
                if payload is None:
                    break
                result_queue = None
                if payload.get("type") == "speech":
                    worker_message = f"[User said] {payload.get('text', '')}"
                elif payload.get("type") == "chat":
                    worker_message = payload.get("prompt", "")
                    result_queue = payload.get("result_queue")
                else:
                    worker_id = payload.get("worker_id", "?")
                    message = payload.get("message", "")
                    done = payload.get("done", False)
                    mark_worker_done(worker_id)
                    worker_message = f"[Worker callback] {message} (worker_id={worker_id}, done={done})."
                print("running agent with message: " + worker_message)
                try:
                    result, success = _agent_worker(agent, worker_message, _message_history)
                    if not success:
                        error_msg = (result.output or "Unknown error").strip()
                        print(f"Agent run failed: {error_msg}")
                        if result_queue is not None:
                            result_queue.put({"error": error_msg})
                        else:
                            _push_outgoing("model", f"Error: {error_msg}", worker_id=payload.get("worker_id"), done=payload.get("done"))
                        continue
                    _message_history.clear()
                    _message_history.extend(result.all_messages())
                    output = (result.output or "").strip()
                    print("agent output: " + output)
                    if result_queue is not None:
                        result_queue.put({"output": result.output})
                    else:
                        _push_outgoing("model", output)
                except Exception as e:
                    import traceback
                    error_msg = f"[Error] {type(e).__name__}: {e}"
                    print(f"Agent error: {error_msg}")
                    traceback.print_exc()
                    if result_queue is not None:
                        result_queue.put({"error": str(e)})
                    else:
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

    # For chat client interfaces
    @app.get("/updates")
    def get_updates():
        """Return new outgoing messages (e.g. agent responses triggered by worker callbacks) and clear the queue."""
        return {"messages": _pop_outgoing()}

    @app.post("/chat")
    def post_chat(payload: dict):
        """Send a message and get a response. Body: {"prompt": "..."}. Runs agent in the same thread as event worker."""
        prompt = payload.get("prompt", "")
        result_queue = queue.Queue()
        event_queue.put_nowait({"type": "chat", "prompt": prompt, "result_queue": result_queue})
        response = result_queue.get()
        if response.get("error"):
            return {"output": "Error: " + response["error"]}
        return {"output": response["output"]}

    # Simple chat UI
    _ui_path = Path(__file__).resolve().parent / "client_ui.html"

    @app.get("/")
    def index():
        if _ui_path.is_file():
            return FileResponse(_ui_path, media_type="text/html")
        return {"message": "Rag agent. POST /chat with {\"prompt\": \"...\"}, GET /updates for event-driven messages."}

    return app, port
