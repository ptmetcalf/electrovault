from __future__ import annotations

import logging
import os
from functools import lru_cache
from math import sqrt
from pathlib import Path
from typing import List, Optional

from photo_brain.core.models import Classification, ExifData, PhotoFile
from photo_brain.vision import taxonomy
from photo_brain.vision.captioner import describe_photo
from photo_brain.vision.model_client import LocalModelError, classify_vision

logger = logging.getLogger(__name__)


def embed_description(
    text: str, photo_id: str | None = None, model: str = "hash-embedder", dim: int = 16
):
    """
    Lazy import wrapper to avoid circular imports when classifier is imported early.
    """
    from photo_brain.embedding.text_embedder import embed_description as _embed_description

    return _embed_description(text, photo_id=photo_id, model=model, dim=dim)


def _norm_score(score: float) -> float:
    clamped = max(0.0, min(score, 0.99))
    return round(clamped, 2)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    da = sqrt(sum(x * x for x in a))
    db = sqrt(sum(y * y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (da * db)))


@lru_cache(maxsize=4)
def _label_embeddings(model_id: str, dim: int) -> list[tuple[str, list[float]]]:
    """
    Precompute embeddings for taxonomy labels for the given embedder.
    """
    labels = taxonomy.taxonomy_labels(include_people_and_pets=False)
    vectors: list[tuple[str, list[float]]] = []
    for label in labels:
        emb = embed_description(label, photo_id=f"label:{label}", dim=dim)
        if emb.model == model_id and len(emb.vector) == dim:
            vectors.append((label, emb.vector))
    return vectors


def _embedding_fallback(photo: PhotoFile, context: str | None) -> Optional[List[Classification]]:
    """
    Fallback classification using caption embedding similarity to taxonomy labels.
    """
    desc = describe_photo(photo, context=context)
    if not desc or not desc.description:
        return None

    query_emb = embed_description(desc.description, photo_id=photo.id, dim=32)
    label_vecs = _label_embeddings(query_emb.model, len(query_emb.vector))
    if not label_vecs:
        return None

    scored: list[tuple[str, float]] = []
    bucket_scores: list[tuple[str, float]] = []
    for label, vec in label_vecs:
        if label.startswith("object:") and label.split(":", 1)[1] in _BLOCKED_OBJECTS:
            continue
        sim = _cosine(query_emb.vector, vec)
        score = _norm_score((sim + 1.0) / 2.0)  # map [-1,1] -> [0,1]
        if label.startswith("bucket:"):
            bucket_scores.append((label, score))
        else:
            scored.append((label, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    bucket_scores.sort(key=lambda x: x[1], reverse=True)
    top_bucket = bucket_scores[0] if bucket_scores else ("bucket:misc_other", 0.5)

    top_labels = scored[:6]
    if top_bucket[0] not in [lbl for lbl, _ in top_labels]:
        top_labels.append(top_bucket)

    classifications: List[Classification] = []
    for label, score in top_labels:
        if label.split(":", 1)[0] in _BLOCKED_CATEGORIES:
            continue
        classifications.append(
            Classification(
                photo_id=photo.id,
                label=label,
                score=score,
                source=f"{query_emb.model}-embed-fallback",
            )
        )
    return classifications if classifications else None


_CLASSIFIER_PROMPT = """
You are an image tagger for a photo search system. Return structured JSON only.

Goal:
Produce 12–20 short, low-level visual tags covering objects, attributes,
environment, activities, events, colors, brands/logos, time/weather, quality,
and any visible text. Prefer more specific tags when visible. Do not tag
people or pets/faces; leave any people/pet fields empty/null because face
detection handles them separately. Also select a
single high-level bucket for the entire image (best match only) from EXACTLY one of:
- people
- groups_events
- selfie
- pets_animals
- food_recipe
- documents
- notes_handwriting
- receipts_bills
- screenshots
- diagrams_charts
- maps_navigation
- memes_comics
- art_illustration
- objects_items
- landscapes_outdoors
- vehicles_transport
- home_interiors
- screens_displays
- shopping_products
- misc_other
If unsure, choose misc_other (never leave bucket null).

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
  "people": null,
  "counts": {},
  "quality": { "blur": 0.1, "lighting": "natural", "composition": [ "rule-of-thirds" ] },
  "bucket": "people",
  "bucket_confidence": 0.94
}
""".strip()

_BLOCKED_OBJECTS = {"person", "adult", "child", "baby", "pet", "dog", "cat", "bird"}
_BLOCKED_CATEGORIES = {"people-count", "people-attr", "age-band", "gender", "pets-count"}


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
        logger.debug("Vision classifier: OLLAMA_VISION_MODEL not set; skipping %s", photo.path)
        return None  # Missing model → skip updates

    prompt = _build_classifier_prompt(context)

    logger.debug(
        "Vision classifier: calling model for %s (prompt %d chars)", photo.path, len(prompt)
    )

    try:
        tag_results, raw = classify_vision(prompt, Path(photo.path))
    except LocalModelError as exc:
        logger.warning("Vision classifier failed for %s: %s", photo.path, exc)
        fallback = _embedding_fallback(photo, context)
        if fallback:
            logger.info("Vision classifier used embedding fallback for %s", photo.path)
            return fallback
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
        fallback = _embedding_fallback(photo, context)
        if fallback:
            logger.info("Vision classifier used embedding fallback (empty tags) for %s", photo.path)
            return fallback
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
