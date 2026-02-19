#!/bin/bash
# Stop all Reachy MCP services

# Get the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PID_FILE="logs/services.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found. Services may not be running."
    exit 1
fi

echo "Stopping Reachy MCP services..."

# Read PIDs and kill processes
while read pid; do
    if ps -p $pid > /dev/null 2>&1; then
        echo "  Stopping process $pid..."
        kill $pid
    else
        echo "  Process $pid already stopped"
    fi
done < "$PID_FILE"

# Wait for processes to terminate
sleep 2

# Force kill any remaining processes
while read pid; do
    if ps -p $pid > /dev/null 2>&1; then
        echo "  Force stopping process $pid..."
        kill -9 $pid
    fi
done < "$PID_FILE"

rm -f "$PID_FILE"
echo "All services stopped."
