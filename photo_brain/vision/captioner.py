from __future__ import annotations

import os
from pathlib import Path

from photo_brain.core.models import ExifData, PhotoFile, VisionDescription
from photo_brain.vision.model_client import LocalModelError, generate_vision


def _fallback_caption(photo: PhotoFile, exif: ExifData | None) -> VisionDescription:
    stem = Path(photo.path).stem.replace("_", " ").replace("-", " ").strip()
    description_parts = [stem or "photo"]
    if exif and exif.datetime_original:
        description_parts.append(exif.datetime_original.strftime("on %Y-%m-%d"))
    description = " ".join(description_parts).strip() or "photo"
    return VisionDescription(
        photo_id=photo.id,
        description=description,
        model="rule-captioner",
        confidence=0.55,
    )


def describe_photo(photo: PhotoFile, exif: ExifData | None = None) -> VisionDescription:
    """Describe a photo using a local vision model if configured, else fallback heuristic."""
    if os.getenv("OLLAMA_VISION_MODEL"):
        try:
            prompt = (
                "Describe this image in 2 concise sentences focusing on objects, people, "
                "setting, and notable details."
            )
            text = generate_vision(prompt, Path(photo.path))
            return VisionDescription(
                photo_id=photo.id,
                description=text,
                model=os.getenv("OLLAMA_VISION_MODEL"),
                confidence=0.75,
            )
        except LocalModelError:
            # Fallback to deterministic caption if local model fails/unavailable.
            return _fallback_caption(photo, exif)
    return _fallback_caption(photo, exif)
