"""FastMCP quickstart example.

Run from the repository root:
    uv run examples/snippets/servers/fastmcp_quickstart.py
"""

import asyncio
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from time import sleep, monotonic
from threading import Thread, current_thread
from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import cv2
from controller import _play_emotion_worker, moves_and_descriptions, _speak_worker
from image_utils import describe_image as _describe_image, detect_faces as _detect_faces

# Session image storage: all take_picture outputs go here; cleared on server shutdown
_IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"


_worker_threads: set[Thread] = set()
# Stay under server graceful shutdown timeout; run join in thread to avoid blocking event loop
_SHUTDOWN_WAIT_SEC = 5


def _join_workers_sync() -> None:
    """Block until workers finish or deadline; run in thread so event loop is not blocked."""
    deadline = monotonic() + _SHUTDOWN_WAIT_SEC
    for t in list(_worker_threads):
        remaining = max(0.0, deadline - monotonic())
        if remaining <= 0:
            break
        t.join(timeout=min(1.0, remaining))

def _next_image_path() -> Path:
    """Return a unique path under _IMAGES_DIR for the next capture."""
    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    base = monotonic()
    path = _IMAGES_DIR / f"reachy_{base:.3f}.jpg"
    while path.exists():
        base = monotonic()
        path = _IMAGES_DIR / f"reachy_{base:.3f}.jpg"
    return path


def _run_worker(worker_fn, *args) -> None:
    """Run worker in current thread; unregister on completion (called from Thread)."""
    try:
        worker_fn(*args)
    finally:
        _worker_threads.discard(current_thread())


@asynccontextmanager
async def lifespan(server):
    global mini
    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    with ReachyMini() as m:
        mini = m
        yield
    # Shutdown: clear session images, then wait for workers
    if _IMAGES_DIR.is_dir():
        try:
            shutil.rmtree(_IMAGES_DIR)
        except OSError:
            pass
    try:
        await asyncio.wait_for(
            asyncio.to_thread(_join_workers_sync),
            timeout=_SHUTDOWN_WAIT_SEC + 1,
        )
    except asyncio.TimeoutError:
        pass

mcp = FastMCP("Reachy Mini Robot", lifespan=lifespan)

@mcp.tool()
def move(
    head_x: float = 0,
    head_y: float = 0,
    head_z: float = 0,
    head_roll: float = 0,
    head_pitch: float = 0,
    head_yaw: float = 0,
    head_mm: bool = False,
    head_degrees: bool = True,
    body_yaw: float | None = 0.0,
    duration: float = 0.5,
    method: str = "minjerk",
):
    """
    Move Reachy Mini to desired position. 
    Args:
        head_x (float): X coordinate of the position.
        head_y (float): Y coordinate of the position.
        head_z (float): Z coordinate of the position.
        head_roll (float): Roll angle
        head_pitch (float): Pitch angle
        head_yaw (float): Yaw angle
        head_mm (bool): If True, convert position from millimeters to meters.
        head_degrees (bool): If True, interpret roll, pitch, and yaw as degrees; otherwise as radians.
        duration (float): Duration of the movement in seconds.
        method (InterpolationTechnique): Interpolation method to use ("linear", "minjerk", "ease", "cartoon"). Default is "minjerk".
        body_yaw (float | None): Body yaw angle in radians. Use None to keep the current yaw.
    """
    mini.goto_target(
        head=create_head_pose(
            x=head_x,
            y=head_y,
            z=head_z,
            roll=head_roll,
            pitch=head_pitch,
            yaw=head_yaw,
            mm=head_mm,
            degrees=head_degrees,
        ),
        body_yaw=body_yaw,
        duration=duration,
        method=method,
    )

@mcp.tool()
def take_picture(for_text_only_model: bool = True) -> Image | str:
    """Take a picture with Reachy Mini's camera.

    Every capture is saved under images/ (cleared on server shutdown).

    Args:
        for_text_only_model: If True, return a text description of the image
            instead of the image itself. Use this when the client model does
            not accept images (e.g. text-only LLM). If False, return the image
            for multimodal models.
    """
    sleep(2)
    frame = mini.media.get_frame()
    path = _next_image_path()
    if not cv2.imwrite(str(path), frame):
        return "Error: failed to save image."
    if for_text_only_model:
        return _describe_image(path)
    return Image(path=str(path))


def _resolve_image_path(image_path: str) -> Path:
    """Resolve path to an image: under images/ or absolute."""
    p = Path(image_path)
    if not p.is_absolute():
        p = _IMAGES_DIR / p
    return p


@mcp.tool()
def describe_image(image_path: str, question: str = "Describe the image in detail.") -> str:
    """Get a short text description of an image (e.g. from take_picture).

    Use when the model does not accept images. image_path can be a filename
    in the session images folder (e.g. reachy_123.456.jpg) or an absolute path.
    Requires OPENAI_API_KEY for description.
    """
    return _describe_image(_resolve_image_path(image_path), question)


@mcp.tool()
def detect_faces(image_path: str) -> list[str]:
    """Detect faces in an image using OpenCV Haar cascade.
    image_path can be a filename in the session images folder or an absolute path.
    """
    return _detect_faces(_resolve_image_path(image_path))

@mcp.tool()
def save_image_person(image_path: str, person_name: str) -> str:
    """If a name is provided for a person in the image, use this to save an image of a person.
    image_path is a filename in the session images folder.
    person_name is the name of the person to save the image of.
    The image is copied to images/people/<person_name>/ with a unique filename.
    """
    src = _resolve_image_path(image_path)
    if not src.is_file():
        return f"Error: image file not found: {image_path}"
    # Safe folder name: strip path separators and limit length
    safe_name = "".join(c for c in person_name if c not in r'\/:*?"<>|').strip() or "unknown"
    dest_dir = _IMAGES_DIR / "people" / safe_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{person_name}{monotonic():.3f}.jpg"
    try:
        shutil.copy2(src, dest)
        return f"Saved to {dest.relative_to(_IMAGES_DIR)}"
    except OSError as e:
        return f"Error copying image: {e}"


@mcp.tool()
def speak(text: str) -> str:
    """Speak words using text to speech with Reachy Mini's speaker.

    Runs in a background thread to avoid blocking the FastMCP event loop.
    """
    t = Thread(target=_run_worker, args=(_speak_worker, mini, text), daemon=True)
    _worker_threads.add(t)
    t.start()
    return "Done"


@mcp.tool()
def list_emotions() -> dict[str, str]:
    """List all emotions available in the emotions library."""
    return moves_and_descriptions

@mcp.tool()
def play_emotion(emotion: str) -> str:
    """Play an emotion.

    Runs in a background thread to avoid AsyncToSync being used
    from the same thread as the FastMCP async event loop.
    """
    if emotion not in moves_and_descriptions:
        return "Emotion not found! Use list_emotions to get the list of available emotions."
    else:
        t = Thread(target=_run_worker, args=(_play_emotion_worker, mini, emotion), daemon=True)
        _worker_threads.add(t)
        t.start()
        return "Done"

# Run with streamable HTTP transport
if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=5000)