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

def _load_kernel_instructions() -> str:
    """Load kernel system prompt from config and inject dynamic robot list."""
    config_dir = os.environ.get("ROSAOS_CONFIG_DIR", "config")
    path = os.path.join(config_dir, "kernel.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            template = f.read().strip()
    except FileNotFoundError:
        logger.warning("Kernel prompt config not found at %s; using fallback", path)
        template = "You are the kernel of an agentic operating system integrated with robots.\nHere is a list of the available robots and their descriptions:\n{robot_list}"
    robot_list = ", ".join(
        [f"{name}: {desc['description']}" for name, desc in validated_mcp_servers.items()]
    )
    return template.replace("{robot_list}", robot_list)


KERNEL_INSTRUCTIONS = _load_kernel_instructions()