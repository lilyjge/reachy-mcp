"""Background worker spawn and callback tools for the MCP server."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastmcp import FastMCP

# worker_id still pending (not yet callback with done=True)
_pending_workers: list[str] = []
# worker_id -> Popen (so we can detect exit without callback)
_worker_procs: dict[str, subprocess.Popen] = {}
# worker_id -> system_prompt (for reaper message when worker dies early)
_worker_system_prompts: dict[str, str] = {}
_workers_lock = threading.Lock()
callback_url = "http://localhost:8765/event"
_reaper_started = False
_reaper_thread: threading.Thread | None = None


def _start_reaper_if_needed() -> None:
    global _reaper_started, _reaper_thread
    if _reaper_started:
        return
    with _workers_lock:
        if _reaper_started:
            return
        _reaper_started = True
    _reaper_thread = threading.Thread(target=_reaper_loop, daemon=True)
    _reaper_thread.start()


def _notify_worker_died(worker_id: str, exit_code: int | None, message: str) -> None:
    """POST to client that worker exited without calling callback; hold _workers_lock only around list/dict edits, not HTTP."""
    payload = {
        "worker_id": worker_id,
        "message": message,
        "done": False,
    }
    try:
        httpx.post(callback_url, json=payload, timeout=10.0)
    except httpx.HTTPError:
        pass


def _reaper_loop() -> None:
    """Background thread: when a worker process exits and worker_id is still in _pending_workers, notify client and clean up."""
    while True:
        time.sleep(1.0)
        to_notify: list[tuple[str, str]] = []
        with _workers_lock:
            for worker_id, proc in list(_worker_procs.items()):
                if proc.poll() is not None:
                    exit_code = proc.returncode
                    system_prompt = _worker_system_prompts.pop(worker_id, "")
                    if worker_id in _pending_workers:
                        _pending_workers.remove(worker_id)
                        task_preview = (system_prompt[:200] + "…") if len(system_prompt) > 200 else system_prompt
                        task_ctx = f" Task was: {task_preview!r}" if system_prompt else ""
                        to_notify.append((worker_id, f"Worker process exited without calling callback (exit code {exit_code}).{task_ctx}"))
                    del _worker_procs[worker_id]
        for wid, msg in to_notify:
            _notify_worker_died(wid, None, msg)

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _rag_agent_worker_script() -> Path:
    """Path to rag_agent.py at project root (parent of server/)."""
    return _project_root() / "rag_agent.py"


def _worker_log_dir() -> Path:
    """Directory for worker stdout/stderr logs (for debugging)."""
    d = _project_root() / "logs" / "workers"
    d.mkdir(parents=True, exist_ok=True)
    return d


def register_background_tools(mcp: FastMCP) -> None:
    """Register spawn_background_instance and callback tools."""

    @mcp.tool()
    def spawn_background_instance(system_prompt: str) -> str:
        """Spawn a background worker that runs an LLM agent with the given system prompt.

        The worker is started with --worker-id and --system-prompt (argparse). It has access to
        the same MCP tools as the main client, plus a callback tool. When the worker completes
        its task, it should call callback(message, done=True, worker_id=...) to send a message
        to the main client and then terminate.

        Args:
            system_prompt: Instructions for the worker agent (e.g. surveillance task, patrol, etc.).

        Returns:
            worker_id: The worker must pass this to callback(message, done, worker_id=...).
        """
        worker_id = str(uuid.uuid4())
        with _workers_lock:
            _pending_workers.append(worker_id)
        script = _rag_agent_worker_script()
        if not script.is_file():
            with _workers_lock:
                _pending_workers.remove(worker_id)
            return f"Error: rag_agent script not found at {script}"
        # Use same Python as the server (venv) so the worker has access to installed deps
        python_exe = os.environ.get("PYTHON") or sys.executable
        cmd = [python_exe, str(script), "--worker", "--worker-id", worker_id, "--system-prompt", system_prompt]
        cwd = str(script.parent)
        log_dir = _worker_log_dir()
        log_path = log_dir / f"worker_{worker_id[:8]}.log"
        try:
            log_file = open(log_path, "w", encoding="utf-8")
        except OSError as e:
            with _workers_lock:
                _pending_workers.remove(worker_id)
            return f"Error creating worker log file: {e}"
        try:
            log_file.write(
                f"[{datetime.now(timezone.utc).isoformat()}] spawn\n"
                f"cmd: {cmd}\n"
                f"cwd: {cwd}\n"
                f"---\n"
            )
            log_file.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,
            )
            log_file.close()
        except Exception as e:
            log_file.close()
            with _workers_lock:
                _pending_workers.remove(worker_id)
            return f"Error spawning worker: {e}"
        _start_reaper_if_needed()
        with _workers_lock:
            _worker_procs[worker_id] = proc
            _worker_system_prompts[worker_id] = system_prompt
        print(f"Worker spawned. worker_id={worker_id} log={log_path}", file=sys.stderr, flush=True)
        return worker_id

    @mcp.tool()
    def callback(message: str, done: bool, worker_id: str) -> str:
        """Send a message to the main client and optionally signal that the worker is done.

        Call this when the background worker has a result or has finished. The server will
        POST to the main client's callback URL. If done=True, the worker should terminate after this call.

        Args:
            message: Message to inject into the main client's conversation (e.g. "Alice detected").
            done: If True, the worker should exit after this callback; the main client will treat this as worker finished.
            worker_id: The worker_id returned by spawn_background_instance for this worker.
        """
        with _workers_lock:
            if worker_id not in _pending_workers:
                return f"Error: unknown worker_id or worker already finished: {worker_id}"
        url = callback_url
        payload = {"worker_id": worker_id, "message": message, "done": done}
        try:
            r = httpx.post(url, json=payload, timeout=10.0)
            r.raise_for_status()
        except httpx.HTTPError as e:
            return f"Callback delivery failed: {e}"
        if done:
            with _workers_lock:
                _pending_workers.remove(worker_id)
                _worker_procs.pop(worker_id, None)
                _worker_system_prompts.pop(worker_id, None)
        return "Callback delivered."

