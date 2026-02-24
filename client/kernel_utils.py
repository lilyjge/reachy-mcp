import os
from dotenv import load_dotenv
import urllib.request
from urllib.error import HTTPError, URLError
from client.common import all_mcp_servers
import logging
logger = logging.getLogger(__name__)

# MCP server validation
validated_mcp_servers: dict[str, dict[str, str]] = {}
_init_done = False

def validate_mcp_servers() -> None:
    global validated_mcp_servers, _init_done
    if _init_done:
        return
    print("Validating MCP servers")
    for server_name, server_config in all_mcp_servers.items():
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

        validated_mcp_servers[server_name] = server_config    
    
    _init_done = True

validate_mcp_servers()

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
{", ".join([f"{name}: {description['description']}" for name, description in validated_mcp_servers.items()])}

When you receive a [Worker callback] message, that is a report from a process that has now finished. 
Do NOT launch a process in response to a worker callback unless absolutely necessary, for example, if the worker errored.
When you receive a [User said] message, the user spoke into the Reachy Mini's microphone. 
This is where you should launch a process as appropriate to respond to the user's message.
ONLY launch ONE process almost always, unless the task involves multiple robots or absolutely necessary. 
The primary user interface is the Reachy Mini, so if you want to communicate with the user, you should launch a process to use Reachy Mini's speak tool.
""".strip()