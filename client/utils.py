from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.models.openai import OpenAIResponsesModel
import logging
from pydantic_ai.providers.openai import OpenAIProvider
import json
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
import os
from dotenv import load_dotenv
import urllib.request
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)
load_dotenv()

# Populated once by _init(); do not run at import time.
all_mcp_servers: dict[str, dict[str, str]] = {}
KERNEL_INSTRUCTIONS: str = ""
model = None

PROCESS_INSTRUCTIONS = """
You are a process in an agentic operating system integrated with robots.
You are responsible for performing a task assigned to you by the kernel.
The kernel may also provide you with context from the current conversation history. Use this context to inform your task.
Plan your task carefully and use the tools available to you to accomplish it.
You have control over a Reachy Mini robot which acts as the primary user interface.
Use tools to physically interact with the user through the robots.
Your output will be sent to the kernel, so inform the kernel whether you have done your task.

Here are instructions for the robot(s) you have access to:
{robot_instructions}
If you have access to multiple robots, you should coordinate your actions with all the tools available to accomplish the task.

Here are the instructions from the kernel:
{kernel_instructions}
""".strip()

_init_done = False


def _init() -> None:
    """Run MCP server validation and model setup once per process."""
    global all_mcp_servers, KERNEL_INSTRUCTIONS, model, _init_done
    if _init_done:
        return

    # MCP server validation
    with open("config/drivers.json", "r") as f:
        config = json.load(f)
        raw_servers = config["mcpServers"]
    _validated_mcp_servers: dict[str, dict[str, str]] = {}
    for server_name, server_config in raw_servers.items():
        server_url = server_config["url"]
        try:
            with urllib.request.urlopen(server_url, timeout=3) as _:
                logger.info(
                    "MCP server %s at %s is reachable and will be enabled",
                    server_name,
                    server_url,
                )
                pass
        except URLError as exc:
            if isinstance(exc, HTTPError):
                pass  # Server responded (e.g. 406); consider it reachable
            else:
                logger.warning(
                    "MCP server %s at %s is not reachable and will be disabled: %s",
                    server_name,
                    server_url,
                    exc,
                )
                continue
        except OSError as exc:
            logger.warning(
                "MCP server %s at %s is not reachable and will be disabled: %s",
                server_name,
                server_url,
                exc,
            )
            continue

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
        server_cfg_with_prompt = dict(server_config)
        server_cfg_with_prompt["prompt"] = prompt
        _validated_mcp_servers[server_name] = server_cfg_with_prompt

    all_mcp_servers = _validated_mcp_servers

    KERNEL_INSTRUCTIONS = f"""
You are the kernel of an agentic operating system integrated with robots.
The system interacts with the user through the Reachy Mini robot which serves as the user interface.
The system has access to other robots as well. Assume the system has whatever robots the user is interested in.

You are responsible for launching agents which are processes for the operating system. These agents will have tools to control the robots.
You will provide detailed system prompts for these agents based on analyzing the user's message and the current context; it must include:
1) A description of the task to be performed and possibly a plan of action, and 
2) Relevant context from the current conversation history.
For example, these system prompts should range from simple instructions like "here is the user's message, respond to it" 
to more complex tasks like "the user wants to play I-Spy, here are the guesses of previous agents, 
here is the user's target, spin around and take pictures until you find the user's target".
In formulating a plan for the system prompt, assume every robot has basic capabilities like moving around, taking pictures, etc. 

You will also decide which robots the agents should have access to in order to accomplish the task by sending the names of the robots to the launch process tool.
Almost always, the agent should have access to the Reachy Mini robot in order to communicate with the user.
Here is a list of the available robots and their descriptions:
{", ".join([f"{name}: {description['description']}" for name, description in all_mcp_servers.items()])}

When you receive a [Worker callback] message, that is a report from a process that has now finished. 
Do NOT launch a process in response to a worker callback unless absolutely necessary, for example, if the worker errored.
When you receive a [User said] message, the user spoke into the Reachy Mini's microphone. 
This is where you should launch a process as appropriate to respond to the user's message.
ONLY launch ONE process almost always, unless the task involves multiple robots or absolutely necessary. 
The primary user interface is the Reachy Mini, so if you want to communicate with the user, you should launch a process to use Reachy Mini's speak tool.
""".strip()

    # Model initialization
    try:
        import httpx
        _local_base_url = "https://localhost:6000/v1"
        _models_url = _local_base_url.rstrip("/") + "/models"
        httpx.get(_models_url, timeout=2.0, verify=False).raise_for_status()
        model = OpenAIResponsesModel(
            "openai/gpt-oss-20b",
            provider=OpenAIProvider(base_url=_local_base_url, api_key="foo"),
        )
    except Exception:
        print("Using Groq model")
        model = GroqModel(
            "openai/gpt-oss-120b",
            provider=GroqProvider(
                api_key=os.environ.get("GROQ_API_KEY"),
            ),
        )

    _init_done = True


_init()


def _make_agent(extra_instructions: str = "", mcp_servers: list[str] = ["reachy-mini"], process: bool = True) -> Agent:
    instructions = PROCESS_INSTRUCTIONS if process else KERNEL_INSTRUCTIONS
    if extra_instructions:
        instructions = instructions.replace("{kernel_instructions}", extra_instructions)
    robot_instructions = "\n".join([f"{server_name}: {all_mcp_servers[server_name]['prompt']}" for server_name in mcp_servers])
    instructions = instructions.replace("{robot_instructions}", robot_instructions)
    return Agent(
        model,
        toolsets=[MCPServerStreamableHTTP(all_mcp_servers[server_name]["url"]) for server_name in mcp_servers],
        instructions=instructions,
        retries=10
    )

def _agent_worker(cur_agent: Agent, message: str, message_history: list = []) -> tuple[AgentRunResult, bool]:
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