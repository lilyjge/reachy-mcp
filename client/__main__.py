# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
import asyncio
import os
import sys
import threading

# Before any event loop is created: on Windows, asyncio's Proactor can raise
# ConnectionResetError when the remote closes the connection during transport
# cleanup. Install a policy so every new loop ignores that in its exception handler.
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


def _parse_args():
    from client.common import GROQ_TOOL_USE_MODELS
    parser = __import__("argparse").ArgumentParser(
        description="rosaOS client: kernel + process manager. Use a local OpenAI-compatible LLM or Groq.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local OpenAI-compatible endpoint (e.g. vLLM). Else use Groq API.",
    )
    parser.add_argument(
        "--endpoint",
        type=int,
        default=int(os.environ.get("LOCAL_LLM_PORT", "6000")),
        metavar="PORT",
        help="Local LLM port when --local (default: 6000, or env LOCAL_LLM_PORT).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
        metavar="MODEL",
        help="Groq model when not --local. Tool-use models: %s (default: openai/gpt-oss-120b)."
        % ", ".join(GROQ_TOOL_USE_MODELS),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("RAG_AGENT_PORT", "8765")),
        metavar="PORT",
        help="Client app port (default: 8765, or env RAG_AGENT_PORT).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()
    # Set env so common.init_model() and process/kernel use them; worker subprocess inherits env.
    if args.local:
        os.environ["LOCAL_LLM"] = "1"
        os.environ["LOCAL_LLM_PORT"] = str(args.endpoint)
    else:
        os.environ.pop("LOCAL_LLM", None)
        os.environ["GROQ_MODEL"] = args.model
    os.environ["RAG_AGENT_PORT"] = str(args.port)

    from client.common import init_all
    init_all()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(_make_connection_reset_safe_policy())

    from client.process import start_process_server
    from client.kernel import main_app
    import uvicorn

    threading.Thread(target=start_process_server, daemon=True).start()
    app, port = main_app()
    uvicorn.run(app, port=port)