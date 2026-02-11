"""CV helpers for images: description, object detection, face detection.

Used when the client model is text-only so image content is converted to text.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from urllib.parse import urlparse
import urllib.request

from dotenv import load_dotenv
from deepface import DeepFace

load_dotenv()
from reachy_mini import ReachyMini
from typing import Any
import torch
from transformers import pipeline
from fastmcp.utilities.types import Image
from time import sleep, monotonic
import cv2
import shutil

pipeline = pipeline(
    task="visual-question-answering",
    model="Salesforce/blip-vqa-base",
    dtype=torch.float16,
)

_IMAGES_DIR = Path(__file__).resolve().parent.parent.parent / "images"
_REACHY_DIR = _IMAGES_DIR / "reachy"
_UPLOAD_DIR = _IMAGES_DIR / "upload"
# Cache mapping URLs to their downloaded local paths to avoid re-downloading
_url_cache: dict[str, Path] = {}

# Groq vision: Scout is faster and cheaper than Maverick; sufficient for image description
_GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
_MAX_BASE64_MB = 4  # Groq limit for base64 image in request


def _next_image_path() -> Path:
    """Return a unique path under _IMAGES_DIR for the next capture."""
    _REACHY_DIR.mkdir(parents=True, exist_ok=True)
    base = monotonic()
    path = _REACHY_DIR / f"reachy_{base:.3f}.jpg"
    while path.exists():
        base = monotonic()
        path = _REACHY_DIR / f"reachy_{base:.3f}.jpg"
    return path


def _download_image_from_url(url: str) -> Path:
    """Download an image from a URL into images/upload and return its local path."""
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    # Try to keep the original extension if present, fallback to .jpg
    suffix = Path(parsed.path).suffix or ".jpg"
    dest = _UPLOAD_DIR / f"upload_{monotonic():.3f}{suffix}"
    urllib.request.urlretrieve(url, dest)
    return dest


def _resolve_image_path(image_path: str | Path) -> Path:
    """Resolve path to an image.

    - If a URL (http/https), download once into images/upload/ and cache the mapping.
      Subsequent uses of the same URL will reuse the cached file.
    - If a relative path, resolve under the images/ directory.
    - If an absolute path, use as-is.
    """
    # Handle URLs so MCP tools can work directly with image URLs.
    if isinstance(image_path, str) and image_path.startswith(("http://", "https://")):
        # Check cache first to avoid re-downloading the same URL
        if image_path in _url_cache:
            cached_path = _url_cache[image_path]
            # Verify the cached file still exists (in case it was deleted)
            if cached_path.exists():
                return cached_path
            # If file was deleted, remove from cache and re-download
            del _url_cache[image_path]
        # Download and cache
        local_path = _download_image_from_url(image_path)
        _url_cache[image_path] = local_path
        return local_path

    p = Path(image_path)
    if not p.is_absolute():
        p = _IMAGES_DIR / p
    return p


def take_picture(mini: ReachyMini, for_text_only_model: bool = True) -> tuple[str, Image | str]:
    """Take a picture with Reachy Mini's camera.

    Every capture is saved under images/ (cleared on server shutdown).

    Args:
        for_text_only_model: If True, return a text description of the image
            instead of the image itself. Use this when the client model does
            not accept images (e.g. text-only LLM). If False, return the image
            for multimodal models.
    """
    sleep(1)
    # Flush one frame to avoid occasionally getting an outdated buffer frame,
    # then grab the next one as the actual snapshot.
    _ = mini.media.get_frame()
    frame = mini.media.get_frame()
    path = _next_image_path()
    if not cv2.imwrite(str(path), frame):
        return "Error: failed to save image."
    if for_text_only_model:
        return (str(path), describe_image(path))
    return (str(path), Image(path=str(path)))


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


def _try_groq_describe_image(resolved: Path, question: str) -> str | None:
    """Use Groq Llama 4 Scout for vision. Return answer string or None to fall back to BLIP."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        import httpx
        raw = resolved.read_bytes()
        if len(raw) > _MAX_BASE64_MB * 1024 * 1024:
            return None
        b64 = base64.b64encode(raw).decode("utf-8")
        # Use jpeg for photos; png for screenshots. Groq accepts both.
        suffix = resolved.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        data_uri = f"data:{mime};base64,{b64}"
        payload = {
            "model": _GROQ_VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.2,
        }
        r = httpx.post(
            _GROQ_CHAT_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30.0,
        )
        if not r.is_success:
            return None
        data = r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
        if content and isinstance(content, str):
            return content.strip()
    except Exception:
        pass
    return None


def describe_image(image_path: str | Path, question: str = "What is in the image?") -> Any:
    """Describe the image: try Groq Llama 4 Scout first, fall back to local BLIP.

    Returns a string (the answer). Uses GROQ_API_KEY when set.
    """
    resolved = _resolve_image_path(image_path)
    if not resolved.is_file():
        return "Error: image file not found."
    groq_answer = _try_groq_describe_image(resolved, question)
    if groq_answer is not None:
        return groq_answer
    return pipeline(question=question, image=str(resolved))


def detect_faces(image_path: str | Path) -> Any:
    """Detect the names of the faces in the image using the DeepFace model.
    """
    resolved = _resolve_image_path(image_path)
    print(DeepFace.find(img_path=str(resolved), db_path="images/people/"))
    return DeepFace.find(img_path=str(resolved), db_path="images/people/")

def analyze_face(image_path: str | Path) -> Any:
    """Analyze the face in the image using the DeepFace model.
    """
    resolved = _resolve_image_path(image_path)
    return DeepFace.analyze(img_path=str(resolved), actions=['age', 'gender', 'race', 'emotion'])