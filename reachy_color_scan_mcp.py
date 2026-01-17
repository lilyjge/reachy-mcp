"""MCP server exposing Reachy Mini photo sweeps for multimodal LLMs.

Claude (or any MCP-aware client) can call the provided tools to spin Reachy Mini's
head, capture multiple photos around the room, and receive those frames back as
data URIs so it can answer visual scavenger hunt prompts such as "find something blue".
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import cv2
from mcp.server.fastmcp import FastMCP
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

CLIENT = FastMCP("reachy-mini-panorama")
SCRIPT_DIR = Path(__file__).resolve().parent
CAPTURE_ROOT = SCRIPT_DIR / "reachy_captures"
CAPTURE_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_SHOTS = 6
YAW_RANGE_DEG = (-80.0, 80.0)
MOVE_DURATION = 1.6
SETTLE_DELAY = 0.9
FRAME_TIMEOUT = 12.0


@dataclass
class PhotoSlice:
    index: int
    yaw_deg: float
    path: Path
    data_uri: str | None


def _even_angles(count: int) -> list[float]:
    if count <= 1:
        return [0.0]
    start, end = YAW_RANGE_DEG
    step = (end - start) / (count - 1)
    return [start + i * step for i in range(count)]


def _wait_for_frame(media) -> "cv2.Mat":
    deadline = time.time() + FRAME_TIMEOUT
    while time.time() < deadline:
        frame = media.get_frame()
        if frame is not None:
            return frame
        time.sleep(0.3)
    raise RuntimeError("Camera did not return a frame in the allotted time.")


def _data_uri_for(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _move_head(mini: ReachyMini, yaw_deg: float) -> None:
    pose = create_head_pose(yaw=yaw_deg, degrees=True)
    mini.goto_target(head=pose, duration=MOVE_DURATION, method="minjerk")
    time.sleep(SETTLE_DELAY)


def _capture_panorama(
    request: str,
    backend: str,
    shots: int,
    inline_data: bool,
) -> dict[str, object]:
    timestamp = datetime.now(timezone.utc)
    scan_id = timestamp.strftime("%Y%m%d_%H%M%S")
    out_dir = CAPTURE_ROOT / scan_id
    out_dir.mkdir(parents=True, exist_ok=True)

    slices: list[PhotoSlice] = []
    with ReachyMini(media_backend=backend) as mini:
        for idx, yaw in enumerate(_even_angles(shots), start=1):
            _move_head(mini, yaw)
            frame = _wait_for_frame(mini.media)
            file_name = f"shot_{idx:02d}_yaw_{int(round(yaw)):+03d}.jpg"
            path = out_dir / file_name
            ok = cv2.imwrite(str(path), frame)
            if not ok:
                raise RuntimeError(f"Failed to write frame to {path}.")
            data_uri = _data_uri_for(path) if inline_data else None
            slices.append(PhotoSlice(idx, yaw, path, data_uri))

    manifest = {
        "scanId": scan_id,
        "requestedPrompt": request,
        "capturedAt": timestamp.isoformat(),
        "shots": [
            {
                "index": item.index,
                "yawDeg": item.yaw_deg,
                "path": str(item.path.resolve()),
                "dataUri": item.data_uri,
            }
            for item in slices
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


@CLIENT.tool()
async def scan_surroundings(
    color_prompt: Annotated[
        str,
        "Natural language description of what the LLM should look for (e.g. 'find something blue').",
    ],
    shots: Annotated[
        int,
        "How many evenly spaced angles to capture (2-8 recommended).",
    ] = DEFAULT_SHOTS,
    backend: Annotated[
        str,
        "Reachy Mini media backend (default/gstreamer/webrtc).",
    ] = "default",
    inline_data: Annotated[
        bool,
        "Whether to embed base64 data URIs in the response so Claude can see the images immediately.",
    ] = True,
) -> dict[str, object]:
    """Spin Reachy Mini's head, capture multiple frames, and return the photos."""

    if shots < 1:
        raise ValueError("shots must be >= 1")
    if shots > 10:
        raise ValueError("shots above 10 create very slow sweeps; pick 10 or less.")

    loop = asyncio.get_running_loop()
    manifest = await loop.run_in_executor(
        None, _capture_panorama, color_prompt, backend, shots, inline_data
    )

    summary = {
        "message": f"Captured {len(manifest['shots'])} photos for '{color_prompt}'.",
        "scan": manifest,
        "analysisTips": [
            "Use yawDeg to describe where each object sits relative to the robot.",
            "If inline_data is False, request the files listed under 'path' via the MCP client's file API.",
        ],
    }
    return summary


def _load_recent_manifests(limit: int) -> list[dict[str, object]]:
    manifests: list[dict[str, object]] = []
    for manifest_path in sorted(CAPTURE_ROOT.glob("*/manifest.json"), reverse=True):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        manifests.append(manifest)
        if len(manifests) >= limit:
            break
    return manifests


def _load_manifest(scan_id: str) -> dict[str, object] | None:
    manifest_path = CAPTURE_ROOT / scan_id / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _latest_scan_id() -> str | None:
    manifests = _load_recent_manifests(1)
    if not manifests:
        return None
    return manifests[0]["scanId"]


@CLIENT.tool()
async def list_recent_scans(
    limit: Annotated[int, "Number of capture sessions to return"] = 5,
) -> dict[str, object]:
    """Surface previous scan metadata when Claude forgets the scanId."""

    loop = asyncio.get_running_loop()
    manifests = await loop.run_in_executor(None, _load_recent_manifests, limit)
    return {"recentScans": manifests, "message": f"Returned {len(manifests)} manifests."}


@CLIENT.tool()
async def get_scan_images(
    scan_id: Annotated[
        str | None,
        "Optional scan identifier; leave empty to pull the most recent capture.",
    ] = None,
) -> dict[str, object]:
    """Return base64 image data for a given scan so Claude can inspect frames."""

    loop = asyncio.get_running_loop()

    target_scan = scan_id or await loop.run_in_executor(None, _latest_scan_id)
    if not target_scan:
        return {"error": "No capture sessions are available yet."}

    manifest = await loop.run_in_executor(None, _load_manifest, target_scan)
    if not manifest:
        return {"error": f"Scan '{target_scan}' is missing or corrupted."}

    images: list[dict[str, object]] = []
    for shot in manifest.get("shots", []):
        path = Path(shot["path"])
        if not path.exists():
            images.append({
                "index": shot["index"],
                "yawDeg": shot["yawDeg"],
                "path": shot["path"],
                "error": "File missing on disk",
            })
            continue
        data_uri = _data_uri_for(path)
        images.append({
            "index": shot["index"],
            "yawDeg": shot["yawDeg"],
            "path": shot["path"],
            "dataUri": data_uri,
        })

    return {
        "message": f"Loaded {len(images)} frames from scan '{target_scan}'.",
        "scanId": target_scan,
        "images": images,
    }


if __name__ == "__main__":
    CLIENT.run()
