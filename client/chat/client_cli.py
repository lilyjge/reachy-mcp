"""
Simple CLI client for the rag agent (POST /chat + poll GET /updates for event-driven messages).

Usage:
  python client_cli.py [--base-url http://localhost:8765]
  Ensure rag_agent.py is running (python rag_agent.py) and the MCP server is up.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import httpx

DEFAULT_BASE = os.environ.get("RAG_AGENT_URL") or "http://localhost:%s" % os.environ.get("RAG_AGENT_PORT", "8765")


def poll_updates(base_url: str, interval: float = 1.5) -> None:
    """Background thread: poll GET /updates and print new messages."""
    while True:
        try:
            r = httpx.get(f"{base_url}/updates", timeout=5.0)
            if r.status_code != 200:
                time.sleep(interval)
                continue
            data = r.json()
            for msg in data.get("messages", []):
                content = msg.get("content", "")
                worker_id = msg.get("worker_id")
                prefix = "[Background] " if worker_id else ""
                print(f"\n{prefix}Robot: {content}\nYou: ", end="", flush=True)
        except (httpx.HTTPError, KeyboardInterrupt):
            pass
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI client for rag agent (chat + event updates).")
    parser.add_argument("--base-url", default=DEFAULT_BASE, help="Base URL of rag_agent server")
    parser.add_argument("--no-poll", action="store_true", help="Disable polling for background event messages")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    if not args.no_poll:
        t = threading.Thread(target=poll_updates, args=(base,), daemon=True)
        t.start()

    print("Rag agent CLI. Type a message and press Enter. Ctrl+C to exit.")
    print("(Background worker callbacks will appear as [Background] Robot: ...)\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            continue
        try:
            r = httpx.post(f"{base}/chat", json={"prompt": prompt}, timeout=60.0)
            r.raise_for_status()
            out = r.json().get("output", "")
            print(f"Robot: {out}\n")
        except httpx.HTTPError as e:
            print(f"Error: {e}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
