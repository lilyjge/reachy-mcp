import os
import logging
import json
import dotenv
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai import Agent, AgentRunResult

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

all_mcp_servers = {}
_init_done = False
# Model initialization
model = None

def init_model() -> None:
    global model
    try:
        import httpx
        _local_base_url = "https://localhost:6000/v1"
        _models_url = _local_base_url.rstrip("/") + "/models"
        httpx.get(_models_url, timeout=2.0, verify=False).raise_for_status()
        model = OpenAIResponsesModel(
            "openai/gpt-oss-20b",
            provider=OpenAIProvider(base_url=_local_base_url, api_key="foo"),
        )
        print("Using local model")
    except Exception:
        print("Using Groq model")
        model = GroqModel(
            "openai/gpt-oss-120b",
            provider=GroqProvider(
                api_key=os.environ.get("GROQ_API_KEY"),
            ),
        )

def init_mcp_servers() -> None:
    global all_mcp_servers
    print("Initializing MCP servers")
    with open("config/drivers.json", "r") as f:
        config = json.load(f)
        all_mcp_servers = config["mcpServers"]
    print("Getting prompts for MCP servers")
    for server_name, server_config in all_mcp_servers.items():
        prompt_path = os.path.join("config", "prompts", f"{server_name}.txt")
        prompt = ""
        try:
            with open(prompt_path, "r", encoding="utf-8") as pf:
                prompt = pf.read().strip()
        except FileNotFoundError:
            logger.info(
                "No prompt file found for MCP server %s (expected at %s)",
                server_name,
                prompt_path,
            )
        except OSError as exc:
            logger.warning(
                "Failed reading prompt file for MCP server %s at %s: %s",
                server_name,
                prompt_path,
                exc,
            )
        server_config["prompt"] = prompt
        all_mcp_servers[server_name] = server_config

def init_all() -> None:
    global _init_done
    if _init_done:
        return
    print("Initializing")
    init_model()
    init_mcp_servers()
    _init_done = True

def agent_worker(cur_agent: Agent, message: str, message_history: list = []) -> tuple[AgentRunResult, bool]:
    """For running the agent in a background thread, we return a tuple of the result and a boolean indicating if the run was successful."""
    import time
    import logging
    logger = logging.getLogger(__name__)
    t0 = time.perf_counter()
    try:
        result = cur_agent.run_sync(message, message_history=message_history)
        elapsed = time.perf_counter() - t0
        logger.info("Agent run_sync finished in %.2fs (success=True)", elapsed)
        return result, True
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.warning("Agent run_sync failed after %.2fs: %s", elapsed, e)
        return AgentRunResult(output=str(e)), False


# Run once at import so model and all_mcp_servers are set before any consumer uses them.
init_all()