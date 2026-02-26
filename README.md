# rosaOS

ROS Agentic Operating System: Control robots with LLMs through MCP with Reachy Mini as the interface.

## Repo Cloning

Cloning this repo requires the use of the recursive flag to download all submodules (ros-mcp-server). Further instructions to setup ros-mcp-server are in the rosaOS setup file found in the submodule directory

```bash
git clone --recursive
```

## Requirements
Using Reachy Mini Lite for easy media stream.

The client supports either a **local OpenAI-compatible LLM** (e.g. vLLM) or the **Groq API**. Choose one via CLI or environment variables.

### Local LLM (OpenAI-compatible endpoint)
For local inference, run an OpenAI-compatible server (e.g. [vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart/)) and point the client at it:

```bash
# On the machine with the GPU (or with port forwarding):
vllm serve openai/gpt-oss-120b --tool-call-parser openai --enable-auto-tool-choice --port 6000
```

Start the client with `--local` and optionally `--endpoint` (port, default 6000):

```bash
python -m client --local
python -m client --local --endpoint 6000
```

Or use environment variables (see [Environment variables](#environment-variables)): `LOCAL_LLM=1`, `LOCAL_LLM_PORT=6000`.

To verify the endpoint: `curl http://localhost:6000/v1/models` (or use `https` if your server uses TLS).

### Groq API
[Groq](https://console.groq.com/keys) provides inference with a free tier (with limits). **You must set an API key** to use Groq:

- **macOS/Linux:** `export GROQ_API_KEY=your_key`
- **Windows (PowerShell):** `$env:GROQ_API_KEY="your_key"`

Get a key at [console.groq.com/keys](https://console.groq.com/keys).

Then start the client without `--local` and optionally choose a model with `--model`:

```bash
python -m client
python -m client --model llama-3.3-70b-versatile
```

Supported Groq tool-use models include: `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `moonshotai/kimi-k2-instruct-0905`, `qwen/qwen3-32b`, and `meta-llama/llama-4-scout-17b-16e-instruct`. Default is `openai/gpt-oss-120b`.

Required for image analysis and better TTS experience. 

## Installation

Developed with Python 3.12.

```
git clone https://github.com/lilyjge/reachy-mcp.git
cd reachy-mcp
python -m venv reachy_mini_env
.\reachy_mini_env\Scripts\activate.ps1  # Windows
# source reachy_mini_env/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```
## Usage

### Quick Start (All Services)
Start all services at once:

```bash
./start_all.sh
```

This will start:
- Reachy Mini daemon (port 8000)
- MCP server (port 5001)
- RAG agent (port 8765)

Logs are saved to `logs/` directory. To stop all services:

```bash
./stop_all.sh
```

### Manual Start (Individual Services)
Alternatively, start each service manually:

Start Reachy Mini's robot daemon server on the default port 8000:

`uv run reachy-mini-daemon`

Start the Reachy Mini's MCP server on port 5001:

`python -m server`

Start the operating system's client (default port 8765):

```bash
python -m client                    # Groq (requires GROQ_API_KEY)
python -m client --local             # Local LLM at port 6000
python -m client --local --endpoint 6000 --port 8765
```

Now you can talk to the Reachy Mini directly.

To chat via CLI instead of the robot:

```bash
python -m client.chat.client_cli
# Optional: --base-url http://localhost:8765  (or set RAG_AGENT_PORT)
```

Or, when the agent is running, visit `http://localhost:8765/` in your browser (or the port you set with `--port` / `RAG_AGENT_PORT`).

### Environment variables

All ports and the LLM source can be overridden by environment variables so scripts and deployed setups don't rely on CLI flags.

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | **Required** when not using `--local`. Groq API key from [console.groq.com/keys](https://console.groq.com/keys). |
| `LOCAL_LLM` | — | Set to `1` or `true` to use local OpenAI-compatible endpoint. |
| `LOCAL_LLM_PORT` | `6000` | Port of local LLM when `LOCAL_LLM` is set. |
| `LOCAL_LLM_ENDPOINT` | — | Full base URL (e.g. `https://localhost:6000/v1`) overrides port. |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model when not using local LLM. |
| `RAG_AGENT_PORT` | `8765` | Client app (kernel + chat) port. |
| `RAG_AGENT_URL` | — | Full base URL for chat CLI (e.g. `http://localhost:8765`). |
| `PROCESS_SERVER_PORT` | `7001` | Process manager MCP server port. |
| `PROCESS_SERVER_URL` | — | Full process server URL (e.g. `http://localhost:7001/mcp`). |
| `REACHY_MCP_PORT` | `5001` | Reachy Mini MCP server port (when starting `python -m server`). |
| `STT_CALLBACK_URL` | from `RAG_AGENT_PORT` | Where the server POSTs transcribed speech (default `http://localhost:{RAG_AGENT_PORT}/stt`). |
| `ROSAOS_CONFIG_DIR` | `config` | Directory for `drivers.json`, `kernel.txt`, `process.txt`, and `prompts/`. |

### Configuration

Agent system prompts and robot config live under the **config** directory (or `ROSAOS_CONFIG_DIR`):

- **`config/kernel.txt`** — System prompt for the kernel agent (one placeholder: `{robot_list}`).
- **`config/process.txt`** — System prompt template for process agents (placeholders: `{robot_instructions}`, `{kernel_instructions}`).
- **`config/drivers.json`** — MCP server names, URLs, and descriptions. If you change `REACHY_MCP_PORT`, update the `reachy-mini` URL in this file to match (e.g. `http://localhost:5001/mcp`).
- **`config/prompts/<server_name>.txt`** — Per-robot instructions for the LLM (e.g. `reachy-mini.txt`).

Edit these files to customize behavior without changing code. 

## Technical Details

rosaOS is structured like a minimal **operating system**: a **kernel** schedules and supervises **processes** (LLM workers) that perform tasks, while a **device layer** (MCP server) exposes hardware (Reachy Mini) as callable tools. The LLM is the “CPU” that executes kernel and process logic.

### High-level architecture

| Layer | Component | OS analogy | Role |
|-------|-----------|------------|------|
| **User / shell** | Reachy Mini, or to chat directly, browser UI or CLI | Shell / terminal | Sends prompts and receives responses; polls for event-driven updates. |
| **Kernel** | Client event worker + Pydantic-AI “kernel” agent | OS kernel / scheduler | Single thread consumes an event queue (speech, worker callbacks, chat messages). Decides when to **launch processes** (workers) via the process server; does not drive the robot directly. |
| **Process manager** | Internal MCP server for kernel | Syscall interface / `fork` | Exposes process management tools to kernel. Spawns worker **subprocesses** (`python -m client.worker`) so each agent has its own event loop and does not block the kernel. |
| **Processes** | Agent worker subprocesses | User processes | Each runs a Pydantic-AI agent with **MCP robot tools**. Executes one task from a system prompt generated by kernel, then POSTs a completion **callback** to the client `/event`. |
| **Device layer** | Reachy MCP server, optionally easily connect additional robot MCP servers | Drivers / HAL | FastMCP server with lifespan owning the ReachyMini connection. Registers tools: `goto_target`, `take_picture`, `speak`, `play_emotion`, `describe_image`, etc. Runs a background **STT loop**: mic → VAD → transcribe → POST to client `/stt`, like a system process for the UI. |
| **Hardware** | Reachy Mini + other robot | Physical devices | Robot daemon and hardware; MCP server talks to Reachy via `reachy_mini` SDK and other robots through ROS. |

### Data flow 

1. **User input** → Speech via Reachy mic is transcribed by the server’s STT loop and POSTed to client `/stt`; or text is sent via CLI or the UI.
2. **Kernel** receives an event (`[User said] ...` or `[Worker callback] ...`). It runs the kernel agent (LLM) with tools from the **process server**, typically calling `launch_process(system_prompt)` to start a worker.
3. **Process manager** starts a worker subprocess with `WORKER_ID`, `WORKER_SYSTEM_PROMPT`, and `CALLBACK_URL` (client `/event`).
4. **Worker** runs the process agent (LLM) with tools from the **Reachy MCP server**: move, see, speak, etc. When done, it POSTs `{ worker_id, message, done }` to `/event`.
5. **Kernel** gets a `[Worker callback]` event and can respond to the user (e.g. via another launched process that uses `speak`) or launch further work. Primary communication to the user is through Reachy speaking; outgoing messages are also pushed to `/updates` for the UI/CLI to poll.

So: **kernel** = one agent that only launches processes; **processes** = short-lived agents that use the robot and report back via callbacks.

### Architecture diagram

See [docs/architecture.md](docs/architecture.md) for a diagram (Mermaid) of the same layout.