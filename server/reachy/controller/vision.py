"""CV helpers for images: description, object detection, face detection.

Used when the client model is text-only so image content is converted to text.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
import threading
from urllib.parse import urlparse
import urllib.request

from dotenv import load_dotenv

load_dotenv()
from reachy_mini import ReachyMini
from typing import Any
from fastmcp.utilities.types import Image
from time import sleep, monotonic
import cv2
import numpy as np
import shutil

# Lazy-loaded to avoid slow server startup
_blip_pipeline: Any = None


def _get_blip_pipeline() -> Any:
    """Load BLIP VQA pipeline on first use."""
    global _blip_pipeline
    if _blip_pipeline is None:
        import torch
        from transformers import pipeline as _pipeline
        _blip_pipeline = _pipeline(
            task="visual-question-answering",
            model="Salesforce/blip-vqa-base",
            dtype=torch.float16,
        )
    return _blip_pipeline

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
    # Brief delay to avoid stale buffer; reduce for lower latency (env TAKE_PICTURE_DELAY_SEC overrides)
    delay = float(os.environ.get("TAKE_PICTURE_DELAY_SEC", "0.35"))
    if delay > 0:
        sleep(delay)
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
    return _get_blip_pipeline()(question=question, image=str(resolved))


def detect_faces(image_path: str | Path) -> Any:
    """Detect the names of the faces in the image using the DeepFace model.
    """
    from deepface import DeepFace
    resolved = _resolve_image_path(image_path)
    result = DeepFace.find(img_path=str(resolved), db_path="images/people/")
    print(result)
    return result


def analyze_face(image_path: str | Path) -> Any:
    """Analyze the face in the image using the DeepFace model.
    """
    from deepface import DeepFace
    resolved = _resolve_image_path(image_path)
    return DeepFace.analyze(img_path=str(resolved), actions=['age', 'gender', 'race', 'emotion'])

_FACE_CENTER_MARGIN = 0.67   # face center within ±67% of image center (tighter)
_MIN_FACE_SIZE = 80          # ignore small/distant faces (noisy)
_EYE_CONFIRM_CONSECUTIVE = 4  # require this many consecutive positive frames

_frontal_face_cascade = None
_eye_cascade = None

def _get_frontal_face_cascade():
    """Lazy-load OpenCV Haar cascade for frontal face detection."""
    global _frontal_face_cascade
    if _frontal_face_cascade is None:
        _frontal_face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _frontal_face_cascade


def _get_eye_cascade():
    """Lazy-load OpenCV Haar cascade for eyes (used inside face ROI)."""
    global _eye_cascade
    if _eye_cascade is None:
        _eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
    return _eye_cascade


def _face_has_two_eyes(gray: np.ndarray, x: int, y: int, fw: int, fh: int) -> bool:
    """Return True if we detect two distinct eyes in the upper part of the face ROI."""
    # Eyes are in the upper ~60% of the face
    roi_y = max(0, int(fh * 0.15))
    roi_h = int(fh * 0.55)
    if roi_h < 20:
        return False
    roi = gray[y + roi_y : y + roi_y + roi_h, x : x + fw]
    if roi.size == 0:
        return False
    eye_cascade = _get_eye_cascade()
    eyes = eye_cascade.detectMultiScale(
        roi,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(12, 12),
    )
    # We need at least 2 eye-like regions (left and right); allow 2–4 (can get duplicates)
    if len(eyes) < 2:
        return False
    # Sanity: two eyes should be separated horizontally (one left, one right)
    ex_centers = [ex + ew / 2 for (ex, _, ew, _) in eyes]
    ex_centers.sort()
    span = ex_centers[-1] - ex_centers[0] if ex_centers else 0
    if span < fw * 0.2:  # too close together, likely same eye or noise
        return False
    return True


def check_making_eye_contact(image_path: str | Path | np.ndarray) -> bool:
    """Check if a face in the image is making eye contact (looking straight at the camera).

    Uses: one frontal face, tightly centered, of sufficient size, with both eyes
    detected in the face ROI to reduce false positives (face turned but eyes not at camera).
    """
    if isinstance(image_path, np.ndarray):
        frame = image_path
        if frame.ndim != 3:
            return False
    else:
        resolved = _resolve_image_path(image_path)
        if not resolved.is_file():
            return False
        frame = cv2.imread(str(resolved))
        if frame is None:
            return False

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = _get_frontal_face_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=9,
        minSize=(_MIN_FACE_SIZE, _MIN_FACE_SIZE),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) != 1:
        return False

    for face in faces:
        x, y, fw, fh = face
        face_center_x = x + fw / 2
        face_center_y = y + fh / 2
        img_center_x = w / 2
        img_center_y = h / 2
        margin_x = _FACE_CENTER_MARGIN * w
        margin_y = _FACE_CENTER_MARGIN * h
        if abs(face_center_x - img_center_x) > margin_x or abs(face_center_y - img_center_y) > margin_y:
            continue
        if not _face_has_two_eyes(gray, x, y, fw, fh):
            continue
        return True
    return False


def wait_for_eye_contact(
    mini: ReachyMini,
    stop_event: threading.Event,
    poll_interval: float = 0.08,
) -> bool:
    """Continuously capture frames and check for eye contact until seen or stop is set.

    Requires several consecutive positive frames to avoid false positives.
    Returns True if eye contact was detected, False if stop_event was set first.
    """
    _ = mini.media.get_frame()
    consecutive = 0
    while not stop_event.is_set():
        frame = mini.media.get_frame()
        if frame is not None and check_making_eye_contact(frame):
            consecutive += 1
            if consecutive >= _EYE_CONFIRM_CONSECUTIVE:
                return True
        else:
            consecutive = 0
        stop_event.wait(timeout=poll_interval)
    return False
