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
    """Very simple filename-based heuristics if no local model is configured or it fails."""
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


_CLASSIFIER_BASE_PROMPT = """
You are an image tagger for a photo library search system.

Your goal is to produce a rich, low-level object/scene description as short tags
that can be used for vector and keyword search.

Respond ONLY with a single JSON object, with EXACTLY these keys:

{
  "labels":   [ { "label": "string", "confidence": 0-1 } ],
  "scene":    [ "string" ],
  "objects":  [ { "label": "string", "confidence": 0-1 } ],
  "activities": [ "string" ],
  "events":   [ "string" ],
  "colors":   [ "string" ],
  "brands":   [ "string" ],
  "time_of_day": "string",
  "weather":  "string",
  "people": {
    "count": int,
    "attributes": [ "string" ],
    "age_bands": [ "child" | "teen" | "adult" | "senior" ],
    "genders": [ "string" ]
  },
  "counts": {
    "pets": int
  },
  "quality": {
    "blur": 0-1,
    "lighting": "string",
    "composition": [ "string" ]
  },
  "ocr_text": [ "string" ]
}

Tagging rules:

- Focus primarily on low-level OBJECTS in "objects":
  - things like: person, man, woman, child, crowd, dog, cat, car, bus,
    tree, flower, table, laptop, phone, book, bottle, plate, building, sign, etc.
  - include important attributes as separate labels where useful:
    "red car", "blue jacket", "snowy road", "wooden table".
- Use short tags (1–3 words), all lowercase where reasonable.
- "labels" should contain the most important, high-level tags summarizing the image.
- "scene" should capture location and context: "city street", "living room",
  "beach", "office", "forest trail", etc.
- "activities" and "events" should capture what people are doing or the type of event:
  "running", "reading", "birthday party", "wedding ceremony", etc.
- "colors" should list dominant colors visible in the scene: "blue", "green", "red", etc.
- "brands" should list any clearly visible brands or logos if you can read them.
- "ocr_text" should contain short strings of any clearly readable text in the image.

Constraints:

- Every confidence value must be a float in [0, 1].
- Do not include any keys other than the ones defined above.
- Do not add explanations, comments, or markdown; output JSON only.
""".strip()


def _build_classifier_prompt(context: str | None) -> str:
    """Inject optional user/context info without changing the JSON schema."""
    if context:
        return (
            _CLASSIFIER_BASE_PROMPT
            + "\n\nOptional context (may help disambiguate, do not echo verbatim):\n"
            + context
        )
    return _CLASSIFIER_BASE_PROMPT


def classify_photo(
    photo: PhotoFile, exif: ExifData | None = None, context: str | None = None
) -> list[Classification]:
    """
    Classify a photo via local vision model if configured, else filename heuristics.

    The local model is prompted to emit a rich set of low-level object and scene tags
    that can be used as a "classifier" / object list for search and filtering.
    """
    if os.getenv("OLLAMA_VISION_MODEL"):
        try:
            prompt = _build_classifier_prompt(context)
            tag_results, _ = classify_vision(prompt, Path(photo.path))
            if tag_results:
                classifications: list[Classification] = []
                for idx, tag_entry in enumerate(tag_results):
                    if isinstance(tag_entry, str):
                        label, confidence = tag_entry, None
                    else:
                        label, confidence = tag_entry

                    # Decreasing fallback score as we go down the ranked list
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
            # If the local model errors, fall back to deterministic heuristics.
            pass

    return _fallback_labels(photo)
