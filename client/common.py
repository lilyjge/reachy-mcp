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

# Default Groq tool-use models (see https://console.groq.com/docs/models)
GROQ_TOOL_USE_MODELS = (
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "moonshotai/kimi-k2-instruct-0905",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
)


def init_model() -> None:
    global model
    use_local = os.environ.get("LOCAL_LLM", "").strip().lower() in ("1", "true", "yes")
    if use_local:
        base_url = os.environ.get("LOCAL_LLM_ENDPOINT", "").strip()
        if not base_url:
            port = int(os.environ.get("LOCAL_LLM_PORT", "6000"))
            base_url = f"https://localhost:{port}/v1"
        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        try:
            import httpx
            _models_url = base_url.rstrip("/") + "/models"
            httpx.get(_models_url, timeout=2.0, verify=False).raise_for_status()
            model = OpenAIResponsesModel(
                os.environ.get("LOCAL_LLM_MODEL", "openai/gpt-oss-20b"),
                provider=OpenAIProvider(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "foo")),
            )
            print("Using local model at", base_url)
        except Exception as e:
            logger.warning("Local LLM endpoint not reachable at %s: %s", base_url, e)
            raise
    else:
        provider = os.environ.get("LLM_PROVIDER", "groq").strip().lower()
        if provider == "anthropic":
            anthropic_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
            api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            if not api_key:
                raise SystemExit(
                    "ANTHROPIC_API_KEY is not set. Get an API key from https://console.anthropic.com and set:\n"
                    "  export ANTHROPIC_API_KEY=your_key   # macOS/Linux\n"
                    "  set ANTHROPIC_API_KEY=your_key      # Windows"
                )
            from pydantic_ai.models.anthropic import AnthropicModel
            from pydantic_ai.providers.anthropic import AnthropicProvider

            model = AnthropicModel(
                anthropic_model,
                provider=AnthropicProvider(api_key=api_key),
            )
            print("Using Anthropic model:", anthropic_model)
        else:
            groq_model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
            api_key = os.environ.get("GROQ_API_KEY", "").strip()
            if not api_key:
                raise SystemExit(
                    "GROQ_API_KEY is not set. Get an API key from https://console.groq.com/keys and set:\n"
                    "  export GROQ_API_KEY=your_key   # macOS/Linux\n"
                    "  set GROQ_API_KEY=your_key      # Windows"
                )
            model = GroqModel(
                groq_model,
                provider=GroqProvider(api_key=api_key),
            )
            print("Using Groq model:", groq_model)

def init_mcp_servers() -> None:
    global all_mcp_servers
    config_dir = os.environ.get("ROSAOS_CONFIG_DIR", "config")
    print("Initializing MCP servers")
    drivers_path = os.path.join(config_dir, "drivers.json")
    with open(drivers_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        # Mutate in-place so other modules that imported this dict get updated.
        all_mcp_servers.clear()
        all_mcp_servers.update(config["mcpServers"])
    print("Getting prompts for MCP servers")
    prompts_dir = os.path.join(config_dir, "prompts")
    for server_name, server_config in all_mcp_servers.items():
        prompt_path = os.path.join(prompts_dir, f"{server_name}.txt")
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
            prompt = server_config["description"]
        except OSError as exc:
            logger.warning(
                "Failed reading prompt file for MCP server %s at %s: %s",
                server_name,
                prompt_path,
                exc,
            )
            prompt = server_config["description"]
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


# init_all() is called by client __main__ (and by client.worker) after env is set; do not run at import.