"""CV helpers for images: description, object detection, face detection.

Used when the client model is text-only so image content is converted to text.
"""

from __future__ import annotations

from pathlib import Path

from deepface import DeepFace
import torch
from transformers import pipeline

pipeline = pipeline(
    task="visual-question-answering",
    model="Salesforce/blip-image-captioning-base",
    dtype=torch.float16,
)


def describe_image(image_path: str | Path, question: str = "Describe the image in detail.") -> str:
    """Describe the image using the BLIP model.

    Args:
        image_path: The path to the image to describe.
        question: The question to ask the model. Defaults to "Describe the image in detail."
    """
    return pipeline(question=question, image=image_path)

def detect_faces(image_path: str | Path) -> list[str]:
    """Detect the names of thefaces in the image using the DeepFace model.
    """
    print(DeepFace.find(img_path = image_path, db_path = "images/people/"))
    return DeepFace.find(img_path = image_path, db_path = "images/people/")