"""MCP server exposing Reachy Mini photo sweeps for multimodal LLMs.

Claude (or any MCP-aware client) can call the provided tools to spin Reachy Mini's
head, capture multiple photos around the room, and receive those frames back as
data URIs so it can answer visual scavenger hunt prompts such as "find something blue".
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import scipy.signal
import soundfile as sf
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
BODY_YAW_LIMIT_DEG = 85.0
STORAGE_MAX_DIM = 1280
INLINE_MAX_DIM = 640
STORAGE_JPEG_QUALITY = 82
INLINE_JPEG_QUALITY = 68
AUDIO_CHUNK_SIZE = 2048

try:
    import pyttsx3  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pyttsx3 = None


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


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _resize_image(frame: "cv2.Mat", max_dim: int) -> "cv2.Mat":
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return frame
    scale = max_dim / float(longest)
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _encode_jpeg(frame: "cv2.Mat", quality: int) -> bytes:
    success, buffer = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(_clamp(quality, 1, 100))]
    )
    if not success:
        raise RuntimeError("Failed to encode JPEG buffer.")
    return bytes(buffer)


def _inline_data_from_frame(frame: "cv2.Mat") -> str:
    preview = _resize_image(frame, INLINE_MAX_DIM)
    payload = base64.b64encode(_encode_jpeg(preview, INLINE_JPEG_QUALITY)).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _store_frame(frame: "cv2.Mat", path: Path) -> "cv2.Mat":
    processed = _resize_image(frame, STORAGE_MAX_DIM)
    ok = cv2.imwrite(
        str(path), processed, [cv2.IMWRITE_JPEG_QUALITY, STORAGE_JPEG_QUALITY]
    )
    if not ok:
        raise RuntimeError(f"Failed to write frame to {path}.")
    return processed


def _data_uri_for(path: Path) -> str:
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"Unable to read frame from {path} for preview encoding.")
    return _inline_data_from_frame(frame)


def _synthesize_speech_audio(
    text: str, voice_hint: str | None = None
) -> tuple[np.ndarray, int]:
    """Create a waveform for the provided text using available TTS backends."""

    temp_path = Path(tempfile.gettempdir()) / f"reachy_tts_{uuid.uuid4().hex}.aiff"
    try:
        if pyttsx3 is not None:
            try:
                engine = pyttsx3.init()
                if voice_hint:
                    voice_hint_lower = voice_hint.lower()
                    for voice in engine.getProperty("voices"):
                        if voice_hint_lower in voice.name.lower():
                            engine.setProperty("voice", voice.id)
                            break
                engine.save_to_file(text, str(temp_path))
                engine.runAndWait()
                data, sample_rate = sf.read(str(temp_path), dtype="float32")
                if len(data) > 0:  # Check if pyttsx3 actually generated audio
                    return data, sample_rate
            except Exception:
                pass  # Fall through to try macOS say command

        say_bin = shutil.which("say")
        if say_bin:
            cmd = [say_bin, "-o", str(temp_path)]
            if voice_hint:
                cmd += ["-v", voice_hint]
            cmd.append(text)
            subprocess.run(cmd, check=True)
            data, sample_rate = sf.read(str(temp_path), dtype="float32")
            return data, sample_rate

        raise RuntimeError(
            "No speech backend installed. Install pyttsx3 or enable macOS 'say'."
        )
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _prepare_audio_payload(
    samples: np.ndarray, sample_rate: int, target_rate: int
) -> np.ndarray:
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)
    if sample_rate != target_rate and len(samples) > 0:
        new_len = max(1, int(round(len(samples) * target_rate / sample_rate)))
        samples = scipy.signal.resample(samples, new_len)
    return np.ascontiguousarray(samples.astype(np.float32))


def _play_audio_on_reachy(
    text: str, backend: str, voice_hint: str | None = None
) -> dict[str, object]:
    normalized = " ".join(text.split())
    if not normalized:
        raise ValueError("Text must contain at least one non-space character.")

    waveform, sample_rate = _synthesize_speech_audio(normalized, voice_hint)
    duration = len(waveform) / sample_rate if len(waveform) else 0.0

    with ReachyMini(media_backend=backend) as mini:
        target_rate = mini.media.get_output_audio_samplerate()
        payload = _prepare_audio_payload(waveform, sample_rate, target_rate)
        if not len(payload):
            raise RuntimeError("Speech synthesis returned an empty waveform.")
        mini.media.start_playing()
        for idx in range(0, len(payload), AUDIO_CHUNK_SIZE):
            chunk = payload[idx : idx + AUDIO_CHUNK_SIZE]
            mini.media.push_audio_sample(chunk)
        play_time = len(payload) / target_rate
        time.sleep(play_time + 0.25)
        mini.media.stop_playing()

    return {
        "message": "Dialogue played through Reachy Mini's speaker.",
        "approxDurationSec": round(duration, 2),
        "backend": backend,
        "voiceHint": voice_hint,
    }


def _move_head(mini: ReachyMini, yaw_deg: float) -> None:
    pose = create_head_pose(yaw=yaw_deg, degrees=True)
    body_yaw = math.radians(_clamp(yaw_deg, -BODY_YAW_LIMIT_DEG, BODY_YAW_LIMIT_DEG))
    mini.goto_target(
        head=pose,
        body_yaw=body_yaw,
        duration=MOVE_DURATION,
        method="minjerk",
    )
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
            stored_frame = _store_frame(frame, path)
            data_uri = _inline_data_from_frame(stored_frame) if inline_data else None
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
            "Call get_scan_images later to fetch compressed previews without re-running the scan.",
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
        try:
            data_uri = _data_uri_for(path)
        except RuntimeError as exc:
            images.append({
                "index": shot["index"],
                "yawDeg": shot["yawDeg"],
                "path": shot["path"],
                "error": str(exc),
            })
            continue
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


@CLIENT.tool()
async def speak_text(
    text: Annotated[str, "What Reachy Mini should say aloud."],
    backend: Annotated[
        str,
        "Reachy Mini media backend (default/gstreamer/webrtc).",
    ] = "default",
    voice_hint: Annotated[
        str | None,
        "Optional substring to pick a specific voice (depends on OS TTS engine).",
    ] = None,
) -> dict[str, object]:
    """Synthesize text-to-speech audio and play it through Reachy Mini."""

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _play_audio_on_reachy, text, backend, voice_hint)
    return result


if __name__ == "__main__":
    CLIENT.run()
