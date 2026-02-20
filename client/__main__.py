# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
import asyncio
import sys
from client.kernel import main_app
import uvicorn
from client.process import start_process_server
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


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(_make_connection_reset_safe_policy())

    threading.Thread(target=start_process_server, daemon=True).start()
    app, port = main_app()
    uvicorn.run(app, port=port)