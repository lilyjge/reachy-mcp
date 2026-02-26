import os
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
import client.common as common


def _load_process_instructions_template() -> str:
    """Load process agent system prompt template from config."""
    config_dir = os.environ.get("ROSAOS_CONFIG_DIR", "config")
    path = os.path.join(config_dir, "process.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return (
            "You are a process in an agentic operating system integrated with robots.\n"
            "Here are instructions for the robot(s) you have access to:\n{robot_instructions}\n"
            "Here are the instructions from the kernel:\n{kernel_instructions}"
        )


def make_agent(extra_instructions: str = "", mcp_servers: list[str] = ["reachy-mini"]) -> Agent:
    instructions = _load_process_instructions_template()
    instructions = instructions.replace("{kernel_instructions}", extra_instructions or "Complete the assigned task.")
    robot_instructions = "\n".join(
        [
            f"{server_name}: {(common.all_mcp_servers.get(server_name) or {}).get('prompt', '')}".rstrip()
            for server_name in mcp_servers
        ]
    ).strip()
    instructions = instructions.replace("{robot_instructions}", robot_instructions)
    return Agent(
        common.model,
        toolsets=[
            MCPServerStreamableHTTP(common.all_mcp_servers[server_name]["url"])
            for server_name in mcp_servers
            if server_name in common.all_mcp_servers
        ],
        instructions=instructions,
        retries=10
    )