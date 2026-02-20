#!/bin/bash
# Start all Reachy MCP services

# Get the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
source reachy_mini_env/bin/activate

# Create logs directory
mkdir -p logs

# PID file to track all processes
PID_FILE="logs/services.pid"
rm -f "$PID_FILE"

echo "Starting Reachy MCP services..."
echo "Logs directory: $SCRIPT_DIR/logs"
echo ""

# Start Reachy Mini daemon
echo "Starting Reachy Mini daemon..."
uv run reachy-mini-daemon > logs/reachy_daemon.log 2>&1 &
DAEMON_PID=$!
echo $DAEMON_PID >> "$PID_FILE"
echo "  ✓ Reachy daemon started (PID: $DAEMON_PID)"

# Wait a moment for daemon to initialize
sleep 8

# Start MCP server
echo "Starting MCP server..."
python -m server > logs/mcp_server.log 2>&1 &
MCP_PID=$!
echo $MCP_PID >> "$PID_FILE"
echo "  ✓ MCP server started (PID: $MCP_PID)"

# Wait a moment for MCP server to initialize
sleep 5

# Start RAG agent
echo "Starting RAG agent..."
python -m client > logs/client.log 2>&1 &
AGENT_PID=$!
echo $AGENT_PID >> "$PID_FILE"
echo "  ✓ RAG agent started (PID: $AGENT_PID)"

echo ""
echo "All services started successfully!"
echo ""
echo "Access points:"
echo "  - Agent UI: http://localhost:8765"
echo "  - MCP Server: http://localhost:5001/mcp"
echo ""
echo "View logs:"
echo "  tail -f logs/reachy_daemon.log"
echo "  tail -f logs/mcp_server.log"
echo "  tail -f logs/rag_agent.log"
echo ""
echo "To stop all services, run: ./stop_all.sh"