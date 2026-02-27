# rosaOS architecture (OS-style)

High-level layout with operating-system analogies: kernel, process manager, processes, and device layer.

```mermaid
flowchart LR
    %% Entrypoints / user interface
    subgraph Entrypoints["Entrypoints"]
        ReachyUI["Reachy Mini UI<br>(primary)"]
        WebUI["Browser UI/CLI"]
    end

    %% rosaOS client side (MCP client)
    subgraph RosaClient["rosaOS"]
        Kernel["Kernel"]
        Agents["Agentic processes <br> (MCP clients)"]
    end

    %% MCP servers (device drivers)
    subgraph MCPS["MCP servers <br>(device drivers)"]
        ReachyMCP["Reachy Mini"]
        ROSdog["Dog (ROS client)"]
        Lamp["Lamp"]
    end

    subgraph LampCode["Lamp hardware"]
        LampSDK["Lamp SDK"]
    end

    %% ROS side
    subgraph ROSWorld["ROS dog hardware"]
        ROSserver["ROS server"]
    end

    %% Hardware detail for Reachy
    subgraph ReachyHW["Reachy hardware"]
        ReachyDaemon["Reachy daemon/SDK"]
    end

    %% Entrypoints into rosaOS
    ReachyUI -->|/stt| Kernel
    WebUI -->|/chat| Kernel

    %% MCP client/server relationships
    Kernel --> Agents
    Agents -->Kernel
    Agents -->|"MCP tool calls"| ReachyMCP
    Agents -->|"MCP tool calls"| ROSdog
    Agents -->|"MCP tool calls"| Lamp

    %% Reachy MCP talks to its ROS server
    ReachyMCP -->|"Tool execution"| ReachyDaemon

    %% ROS MCP talks to ROS graph and devices
    ROSdog -->|"Tool execution"| ROSserver

    Lamp -->|"Tool execution"| LampCode
```

## Flow summary

| Step | What happens |
|------|----------------|
| 1 | User speaks → Reachy mic → STT loop transcribes → POST `/stt` → event queue. |
| 2 | Kernel agent consumes event (e.g. `[User said] ...`), calls `launch_process(system_prompt)` on process server. |
| 3 | Process server spawns worker subprocess; worker runs process agent with robot MCP tools. |
| 4 | Worker uses tools (e.g. `speak`, `take_picture`) → Reachy MCP server → ReachyMini → daemon/robot. |
| 5 | Worker finishes → POST `/event` to kernel with `worker_id`, `message`, `done` → kernel gets `[Worker callback]` event. |
| 6 | Kernel may launch another process (e.g. to speak to user). |

## Ports

| Port | Service | Role |
|------|---------|------|
| 8000 | Reachy Mini daemon | Robot control (external) |
| 5001 | Reachy MCP server | Device layer: robot tools + STT loop |
| 7001 | Process server (MCP) | Internal process manager |
| 8765 | Client (FastAPI) | Kernel + HTTP API (event, stt, chat, updates) |
| 6000 | vLLM (optional) | Local LLM endpoint |
