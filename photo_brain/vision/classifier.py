from __future__ import annotations

import os
from pathlib import Path
from typing import List

from photo_brain.core.models import Classification, ExifData, PhotoFile
from photo_brain.vision.model_client import LocalModelError, classify_vision


def _norm_score(score: float) -> float:
    """Clamp and round scores to 2 decimal places in [0, 0.99]."""
    clamped = max(0.0, min(score, 0.99))
    return round(clamped, 2)


def _fallback_labels(photo: PhotoFile) -> List[Classification]:
    path_lower = photo.path.lower()
    labels: list[Classification] = []
    if "selfie" in path_lower or "portrait" in path_lower:
        labels.append(
            Classification(
                photo_id=photo.id,
                label="portrait",
                score=_norm_score(0.7),
                source="rule-classifier",
            )
        )
    if any(word in path_lower for word in ("cat", "dog", "pet")):
        labels.append(
            Classification(
                photo_id=photo.id,
                label="pet",
                score=_norm_score(0.65),
                source="rule-classifier",
            )
        )

    if not labels:
        labels.append(
            Classification(
                photo_id=photo.id,
                label="photo",
                score=_norm_score(0.5),
                source="rule-classifier",
            )
        )
    return labels


def classify_photo(
    photo: PhotoFile, exif: ExifData | None = None
) -> list[Classification]:
    """Classify a photo via local vision model if configured, else filename heuristics."""
    if os.getenv("OLLAMA_VISION_MODEL"):
        try:
            tag_results, ocr_texts = classify_vision(
                (
                    "Respond ONLY with a JSON object (no prose, no code fences). Schema: "
                    '{ "labels": [ { "label": "string", "confidence": 0-1 } ], '
                    '"scene": [ "string" ], "objects": [ { "label": "string", "confidence": 0-1 } ], '
                    '"activities": [ "string" ], "events": [ "string" ], "colors": [ "string" ], '
                    '"brands": [ "string" ], "time_of_day": "string", "weather": "string", '
                    '"people": { "count": int, "attributes": [ "string" ], "age_bands": [ "child|teen|adult|senior" ], '
                    '"genders": [ "string" ] }, "counts": { "pets": int }, '
                    '"quality": { "blur": 0-1, "lighting": "string", "composition": [ "string" ] }, '
                    '"ocr_text": [ "string" ] }. Use short tags (1-3 words); confidences in 0-1.'
                ),
                Path(photo.path),
            )
            if tag_results:
                classifications: list[Classification] = []
                for idx, tag_entry in enumerate(tag_results):
                    if isinstance(tag_entry, str):
                        label, confidence = tag_entry, None
                    else:
                        label, confidence = tag_entry
                    fallback_score = max(0.4, 0.65 - idx * 0.03)
                    classifications.append(
                        Classification(
                            photo_id=photo.id,
                            label=label,
                            score=_norm_score(
                                confidence if confidence is not None else fallback_score
                            ),
                            source="ollama-vision",
                        )
                    )
                return classifications
        except LocalModelError:
            pass
    return _fallback_labels(photo)
