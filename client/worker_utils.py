from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from client.common import model, all_mcp_servers

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

def make_agent(extra_instructions: str = "", mcp_servers: list[str] = ["reachy-mini"]) -> Agent:
    instructions = PROCESS_INSTRUCTIONS
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