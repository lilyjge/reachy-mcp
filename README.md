# Agentic Reachy Mini with MCP

Reachy Mini controlled by an LLM via MCP with a special architecture and client to allow background tasks for very agentic capabilities.

## Requirements
Using Reachy Mini Lite for easy media stream.
Tested on Windows with Python 3.12.

For LLM client, currently supports local endpoint or Groq API.
Defaults to local and if endpoint is not accessible, uses Groq.
Currently using gpt-oss-20b.

### Local LLM
For local setup, I SSH into a GPU server and deploy with [vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart/):

`vllm serve openai/gpt-oss-20b --tool-call-parser openai --enable-auto-tool-choice --port 6000`

This is done in VS Code for the automatic port forwarding. To test this is successfully, run this locally and you should see the model:

`curl http://localhost:6000/v1/models`

This endpoint is currently hardcoded. Change in code if different.

### Groq API
[Groq](https://console.groq.com/keys) is an inference provider with a free tier for personal use but has limits.
To use it, get an API key and set it as an environment variable.

`GROQ_API_KEY=...`

## Installation

```
git clone https://github.com/lilyjge/reachy-mcp.git
cd reachy-mcp
python -m venv reachy_mini_env
.\reachy_mini_env\Scripts\activate.ps1
pip install -r requirements.txt
```


## Usage
Start Reachy Mini's server on the default port 8000:

`uv run reachy-mini-daemon `

Start the MCP server on port 5000:

`python .\server\server.py`

Start the agent's client server on port 8765:

`python rag_agent.py`

Now you can talk to the Reachy Mini directly.

If you want to directly chat with LLM without going through Reachy, you can launch a CLI chat:

`python client_cli.py`

Or, when the agent is running, visit `http://localhost:8765/` in your browser.

## Technical Details

There are the basic robot MCP tools and some more advanced cool ones like facial analysis.
Ask the LLM to learn more.

### Background Workers
There are two MCP tools that make this work.

First is the tool to **launch a background worker**.
This calls the same agent script as the main agent in a subprocess with instructions that will be injected into the system prompt.
The background worker has access to the same MCP server as the main agent and its job is to do that specific task.
Logs can be found in `logs/workers/`. 
We keep track of worker ids and system prompts so if a subprocess dies unexpectedly, we can tell the main agent. 
What does dying unexpectedly mean? That brings us to the second tool.

We introduce server to client communication. 
This comes in the form of the **callback** MCP tool. 
Background workers are instructed to call this tool when they have completed their task.
The tool posts to an endpoint our custom agentic client has exposed.
On the client side endpoint, the callback message will be injected as a special user message for the main agent to process and to inform the user.

### STT
STT uses a similar workflow as the background workers, but it's special enough to warrant a 'hack'.
The STT loop is always started programmatically when the MCP server is launched.
It simulates natural conversation flow by listening until pauses with VAD and then transcribing with Whisper.
After a complete user turn, and the user pauses, this tool posts to the client endpoint similar to the callback tool. 