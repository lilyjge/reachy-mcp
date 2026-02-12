# ROS MCP Server Setup Guide

Complete guide to setting up and using the ROS MCP (Model Context Protocol) server to control ROS robots from AI agents.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running the Server](#running-the-server)
- [Client Integration](#client-integration)
- [VS Code MCP Integration](#vs-code-mcp-integration)
- [Troubleshooting](#troubleshooting)

---

## Overview

The ROS MCP server provides a bridge between AI agents (like Claude, GPT, or custom agents) and ROS robots. It exposes ROS functionality through the Model Context Protocol, allowing:

- **Direct robot control**: Send actions (dock, undock, drive, rotate)
- **Sensor data access**: Read topics, get camera feeds, LIDAR data
- **System inspection**: List nodes, topics, services, actions
- **Multi-transport support**: stdio (VS Code), HTTP, streamable-http

**Key Architecture:**
```
AI Agent (rag_agent.py)
    ↓
ROS MCP Server (port 8090, streamable-http)
    ↓
ROS 2 System (TurtleBot4, etc.)
```

---

## Prerequisites for Turtlebot4 instance running in docker

### 1. ROS Robot Setup
You need a ROS robot with rosbridge running:

**Clone Turtlebot4 Repo**
```bash
git clone https://git.uwaterloo.ca/robohub/turtlebot4
cd turtlebot4
```

**Inside Docker Container (TurtleBot4):**
```bash
# Enter docker container after turtlebot repo is downloaded
./start.sh

# Install rosbridge-suite
sudo apt update
sudo apt install -y ros-humble-rosbridge-suite

# Source ROS environment
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=1  # Match your robot's domain

# Launch rosbridge
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

### 2. Network Configuration
- **Robot IP**: Find your robot's IP (e.g., `192.168.186.3`)
- **Port 9090**: rosbridge must be accessible on this port
- **Verify connection**: `nc -z -v <robot_ip> 9090` or check with `lsof -i :9090`

---

## MCP Install and Running

### Clone the Repository
```bash
git clone https://github.com/robotmcp/ros-mcp-server.git
cd ros-mcp-server
```

### Install Dependencies
```bash
pip install -e .
```

**Required packages:**
- `fastmcp>=2.11.3` - MCP server framework
- `websocket-client>=1.8.0` - WebSocket connection to rosbridge
- `opencv-python>=4.11.0.86` - Image processing
- `pillow>=11.3.0` - Image handling
- `jsonschema>=4.25.1` - JSON validation


---

## Quick Start

### 1. Configure Robot IP
Edit `ros_mcp/main.py` to set your robot's IP:

```python
# ROS bridge connection settings
ROSBRIDGE_IP = "192.168.186.3"  # Your robot's IP
ROSBRIDGE_PORT = 9090            # Default rosbridge port
```

### 2. Start the MCP Server in turtlebot docker shell terminal
```bash
./start

# For AI agent integration (recommended)
python -m ros_mcp.main --transport streamable-http --host 0.0.0.0 --port 9090
#making sure the port number matches the port of the docker container running turtlebot code

```

### 3. Test the Connection
```bash
# Check if server is running
curl -s http://localhost:9090/mcp 2>&1 | head -5
```

---

## Client Integration in Reachy MCP

To integrate the newly running ros mcp server for use with Reachy MCP, open the 'rag_agent.py' file, add a new 'mcpserver' variable with the http link to the running ros-mcp server, and edit the '_make_agent' function so it accepts the newly created ros-mcp toolset

## VS Code MCP Integration

### Setup for VS Code

1. **Create MCP configuration** (`.vscode/mcp.json`):

Edit the url to be the same as the mcp server that was started from the commands running above

```json
{
  "servers": {
     "ros_mcp_http": {
      "type": "http",
      "url": "http://0.0.0.0:9090/mcp"
    },
  }
}
```

2. **Reload VS Code** - The MCP server will auto-start when VS Code loads

---









## Port Reference

| Port | Service | Description |
|------|---------|-------------|
| 5001 | Reachy MCP | Reachy Mini robot tools |
| 9090 | ROS MCP | ROS/TurtleBot control (your MCP server) |
| 8765 | rag_agent | Main AI agent FastAPI server |
| 9090 | rosbridge | WebSocket bridge to ROS (inside docker/robot) |

---

## Common Commands

### Start All Services

```bash
# 1. Start rosbridge (in docker)
./start.sh

# 2. Start ROS MCP server
cd /path/to/ros-mcp-server
python -m ros_mcp.main --transport streamable-http --host 0.0.0.0 --port 8090

# 3. Start Reachy MCP (if using Reachy)
cd /path/to/lily-reachy-mcp/reachy-mcp
SKIP_HARDWARE=true python server/server.py

# 4. Start AI agent
cd /path/to/lily-reachy-mcp/reachy-mcp
python rag_agent.py
```

## Additional Resources

- [ROS MCP Server GitHub](https://github.com/robotmcp/ros-mcp-server)
- [Model Context Protocol Spec](https://modelcontextprotocol.io)
- [rosbridge Protocol](http://wiki.ros.org/rosbridge_protocol)
- [TurtleBot4 Documentation](https://turtlebot.github.io/turtlebot4-user-manual/)
