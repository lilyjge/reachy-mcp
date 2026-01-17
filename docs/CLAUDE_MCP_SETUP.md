# Reachy Mini MCP Photo Sweep

Turn Reachy Mini into a multimodal scout that can collect photos on demand for Claude.

## Prerequisites

- Reachy Mini daemon running on the same machine as this repo.
- This repository's Python environment activated (e.g. `source reachy_mini_env/bin/activate`).
- Extra MPC dependencies:
  ```bash
  pip install fastmcp mcp
  ```
  (OpenCV and `reachy_mini` are already part of the shipped environment.)

## Starting the MCP server

```bash
python reachy_color_scan_mcp.py
```

The server exposes four tools:

1. `scan_surroundings` – Spins Reachy Mini’s torso and head together across evenly spaced yaw angles, captures multiple photos, and returns lightweight preview data URIs (unless you set `inline_data=False`). The metadata includes the yaw angle and file path for each shot.
2. `list_recent_scans` – Returns cached manifests so Claude can recover a `scanId` or re-download files.
3. `get_scan_images` – Re-encodes the images from a given (or the most recent) scan into compressed previews so Claude can inspect them later without rerunning a sweep.
4. `speak_text` – Synthesizes a short message (via `pyttsx3` if installed, otherwise macOS `say`) and plays it through Reachy Mini’s onboard speaker.

Every capture session is stored under `reachy_captures/<scanId>/shot_XX_yaw_±YY.jpg` alongside a `manifest.json` summary.

## Wiring Claude Desktop

Add the server to the `claude_desktop_config.json` file (restart Claude after saving):

```json
{
  "mcpServers": {
    "reachy-mini-panorama": {
      "command": "/Users/james/Desktop/reachy-mcp/reachy_mini_env/bin/python",
      "args": ["reachy_color_scan_mcp.py"],
      "env": {
        "PYTHONPATH": "/Users/james/Desktop/reachy-mcp"
      }
    }
  }
}
```

Claude can then call `scan_surroundings` whenever a user asks “Find something blue” or “Show me the tidiest corner”. Each call returns:

- `scan.scanId` – Use this to reference photos later.
- `scan.shots[].yawDeg` – Helps narrate where in the panorama the object appeared.
- `scan.shots[].path` – Use Claude’s _Add Attachment_ button or the client’s `resources.readResource` call if you need the original JPEG.

### Example workflow inside Claude

1. User: “Find something blue on my desk.”
2. Claude: Calls `scan_surroundings` with `color_prompt="find something blue"`.
3. Claude inspects the returned data URIs to reason about each frame, then answers the user with coordinates such as “At yaw +45° there is a blue notebook.”

Inline previews are auto-resized (max 640px wide) and JPEG-compressed so tool payloads stay below Claude’s 1 MB limit. Set `inline_data=False` if you want to return only file paths—Claude can always call `get_scan_images` or `resources.readResource` later to fetch the compressed previews.

### Giving Reachy Mini a voice

To use `speak_text`, install `pyttsx3` (cross-platform) **or** rely on macOS’s built-in `say` binary. Example install:

```bash
pip install pyttsx3
```

Claude can then respond with natural phrases like “The blue mug is on the right” while the MCP server streams the synthesized audio directly to Reachy Mini’s speaker.
