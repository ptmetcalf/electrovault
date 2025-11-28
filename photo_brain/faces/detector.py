from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from photo_brain.core.models import FaceDetection, PhotoFile


def _encode_face(photo: PhotoFile) -> list[float]:
    digest = hashlib.sha256(f"{photo.id}{photo.path}".encode("utf-8")).digest()
    return [(byte / 255.0) * 2 - 1 for byte in digest[:16]]


def _confidence(photo: PhotoFile) -> float:
    """Deterministic pseudo-confidence for variability across photos."""
    digest = hashlib.sha256(photo.id.encode("utf-8")).digest()
    # Map first byte to [0.55, 0.95]
    return 0.55 + (digest[0] / 255.0) * 0.4


def detect_faces(photo: PhotoFile) -> list[FaceDetection]:
    """Deterministic face detector stub that yields a single detection per image."""
    try:
        with Image.open(Path(photo.path)) as img:
            width, height = img.size
    except Exception:
        width = height = 1
    bbox = (
        0.25 * float(width),
        0.25 * float(height),
        0.75 * float(width),
        0.75 * float(height),
    )
    return [
        FaceDetection(
            photo_id=photo.id,
            bbox=bbox,
            confidence=_confidence(photo),
            encoding=_encode_face(photo),
        )
    ]
