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

logger = logging.getLogger(__name__)
load_dotenv()

KERNEL_INSTRUCTIONS = """
You are the kernel of an agentic operating system integrated with robots.
The system interacts with the user through the Reachy Mini robot which serves as the user interface.
The system has access to other robots as well. Assume the system has whatever robots the user is interested in.

You are responsible for launching agents which are processes for the operating system. These agents have tools to control the robots.
You will provide detailed system prompts for these agents based on analyzing the user's message and the current context; it must include:
1) A description of the task to be performed and possibly a plan of action, and 
2) Relevant context from the current conversation history.
For example, these system prompts should range from simple instructions like "here is the user's message, respond to it" 
to more complex tasks like "the user wants to play I-Spy, here are the guesses of previous agents, 
here is the user's target, spin around and take pictures until you find the user's target".
In formulating a plan for the system prompt, assume every robot has basic capabilities like moving around, taking pictures, etc. 

When you receive a [Worker callback] message, that is a report from a process that has now finished. 
Do NOT launch a process in response to a worker callback unless absolutely necessary, for example, if the worker errored.
When you receive a [User said] message, the user spoke into the Reachy Mini's microphone. 
This is where you should launch a process as appropriate to respond to the user's message.
ONLY launch ONE process almost always, unless the task involves multiple robots or absolutely necessary. 
The primary user interface is the Reachy Mini, so if you want to communicate with the user, you should launch a process to use Reachy Mini's speak tool.
""".strip()

PROCESS_INSTRUCTIONS = """
You are a process in an agentic operating system integrated with robots.
You are responsible for performing a task assigned to you by the kernel.
The kenerl may also provide you with context from the current conversation history. Use this context to inform your task.
Plan your task carefully and use the tools available to you to accomplish it.
You have control over a Reachy Mini robot which acts as the primary user interface.
Use tools to physically interact with the user through the robots.

Reachy Mini instructions:
- As a LLM, you don't have image capabilities, but the robots have image tools, so work with these tools and image paths.
- You are a text only model, so remember that when you call Reachy's take picture tool.
- DO NOT try to pass images to the model, use the available tools and file paths instead.
- Use the describe_image tool to answer questions about images.
- Use the analyze face tool to analyze faces in images.
- If you are provided with an image with a name, save the image with the name.
- If you take a picture and the image matches a named person, save the image with the name too.
- To communicate with the user, always use the speak tool.
- If the speak tool was successfully called, no need to call it again with the same message.
- Report to the user with the speak tool when you have finished your task.
- Your output will be sent to the kernel, so inform the kernel whether you have done your task and informed the user.

You also have access to a TurtleBot4 robot with the following capabilities:
- Navigation: drive forward/backward, rotate, navigate to positions
- Docking: dock to charging station (dock action) and undock from charging station (undock action)
- Sensors: camera feed, LIDAR scans for obstacle detection
- Actions: wall_follow, drive_distance, rotate_angle, drive_arc, and more
You can coordinate actions between the Reachy Mini and TurtleBot to accomplish complex tasks.

Here are the instructions from the kernel.
""".strip()

model = None
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


_mcp_servers: list[MCPServerStreamableHTTP] = []
with open("client/config.json", "r") as f:
    config = json.load(f)
    _mcp_servers = [MCPServerStreamableHTTP(url) for url in config["mcp_servers"]]

def _make_agent(extra_instructions: str = "", mcp_servers: list[MCPServerStreamableHTTP] = [], process: bool = True) -> Agent:
    instructions = PROCESS_INSTRUCTIONS if process else KERNEL_INSTRUCTIONS
    if extra_instructions:
        instructions = f"{instructions}\n\n{extra_instructions}"
    return Agent(
        model,
        toolsets=mcp_servers if len(mcp_servers) > 0 else _mcp_servers,
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