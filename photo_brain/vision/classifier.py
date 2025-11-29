from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

from photo_brain.core.models import Classification, ExifData, PhotoFile
from photo_brain.vision.model_client import LocalModelError, classify_vision

logger = logging.getLogger(__name__)


def _norm_score(score: float) -> float:
    clamped = max(0.0, min(score, 0.99))
    return round(clamped, 2)


_CLASSIFIER_PROMPT = """
You are an image tagger for a photo search system. Return structured JSON only.

Goal:
Produce 12–20 short, low-level visual tags covering objects, attributes,
environment, activities, events, colors, brands/logos, time/weather, quality,
and any visible text. Prefer more specific tags when visible.

Rules:
- Use lowercase, 1–3 word tokens; no sentences or speculation.
- Every label should include a confidence 0–1 when possible.
- Keep arrays concise; omit unknown fields by using null/empty arrays.
- Never add markdown, prose, or explanations.

Required JSON shape:
{
  "labels": [ { "label": "primary tag", "confidence": 0.92 }, ... ],
  "objects": [ { "label": "object tag", "confidence": 0.85 }, ... ],
  "scene": [ "indoor", "kitchen" ],
  "activities": [ "cooking", "talking" ],
  "events": [ "birthday party" ],
  "colors": [ "color:blue", "color:red" ],
  "brands": [ "brand:apple", "brand:nike" ],
  "time_of_day": "morning" | "afternoon" | "evening" | "night" | null,
  "weather": "sunny" | "cloudy" | "rainy" | "snowy" | "foggy" | null,
  "ocr_text": [ "visible sign text" ],
  "people": {
    "count": 0,
    "attributes": [ "adult", "smiling" ],
    "age_bands": [ "adult" ],
    "genders": [ "male" ]
  },
  "counts": { "pets": 0 },
  "quality": { "blur": 0.1, "lighting": "natural", "composition": [ "rule-of-thirds" ] }
}
""".strip()


def _build_classifier_prompt(context: str | None) -> str:
    if context:
        return (
            _CLASSIFIER_PROMPT
            + "\n\nOptional context (may help interpret; do not repeat verbatim):\n"
            + context
        )
    return _CLASSIFIER_PROMPT


def classify_photo(
    photo: PhotoFile, exif: ExifData | None = None, context: str | None = None
) -> Optional[List[Classification]]:
    """
    Classify a photo via the local vision model.

    Null fallback:
    - If model unavailable → []
    - If model fails parsing → []
    - No heuristic filename fallbacks.
    """
    model_name = os.getenv("OLLAMA_VISION_MODEL")
    if not model_name:
        return None  # Missing model → skip updates

    prompt = _build_classifier_prompt(context)

    try:
        tag_results, raw = classify_vision(prompt, Path(photo.path))
    except LocalModelError as exc:
        logger.warning("Vision classifier failed for %s: %s", photo.path, exc)
        return None
    except Exception as exc:
        logger.error("Vision classifier runtime error for %s: %s", photo.path, exc)
        return None

    logger.debug(
        "Vision classifier raw output\nImage: %s\nPrompt:\n%s\nOutput:\n%s",
        photo.path,
        prompt,
        raw,
    )

    if not tag_results:
        return []

    classifications: List[Classification] = []

    for idx, entry in enumerate(tag_results):
        if isinstance(entry, str):
            label = entry
            confidence = None
        elif isinstance(entry, (tuple, list)) and len(entry) >= 1:
            label = str(entry[0])
            confidence = entry[1] if len(entry) > 1 else None
        else:
            continue

        label = label.strip()
        if not label:
            continue

        # Ranked fallback score (purely ordering)
        fallback_score = max(0.4, 0.7 - idx * 0.03)
        score = _norm_score(confidence if isinstance(confidence, (float, int)) else fallback_score)

        classifications.append(
            Classification(
                photo_id=photo.id,
                label=label,
                score=score,
                source=model_name,
            )
        )

    return classifications
