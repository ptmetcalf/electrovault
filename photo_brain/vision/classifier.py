from __future__ import annotations

import os
from pathlib import Path
from typing import List

from photo_brain.core.models import Classification, ExifData, PhotoFile
from photo_brain.vision.model_client import LocalModelError, classify_vision


def _fallback_labels(photo: PhotoFile) -> List[Classification]:
    path_lower = photo.path.lower()
    labels: list[Classification] = []
    if "selfie" in path_lower or "portrait" in path_lower:
        labels.append(
            Classification(
                photo_id=photo.id, label="portrait", score=0.7, source="rule-classifier"
            )
        )
    if any(word in path_lower for word in ("cat", "dog", "pet")):
        labels.append(
            Classification(
                photo_id=photo.id, label="pet", score=0.65, source="rule-classifier"
            )
        )

    if not labels:
        labels.append(
            Classification(
                photo_id=photo.id, label="photo", score=0.5, source="rule-classifier"
            )
        )
    return labels


def classify_photo(
    photo: PhotoFile, exif: ExifData | None = None
) -> list[Classification]:
    """Classify a photo via local vision model if configured, else filename heuristics."""
    if os.getenv("OLLAMA_VISION_MODEL"):
        try:
            tags = classify_vision(
                "Provide 5-10 short tags for this image as a JSON array of strings.",
                Path(photo.path),
            )
            if tags:
                return [
                    Classification(
                        photo_id=photo.id,
                        label=tag,
                        score=0.7,
                        source="ollama-vision",
                    )
                    for tag in tags
                ]
        except LocalModelError:
            pass
    return _fallback_labels(photo)
