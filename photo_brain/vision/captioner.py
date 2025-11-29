from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional, Tuple

from photo_brain.core.models import ExifData, PhotoFile, VisionDescription
from photo_brain.vision.model_client import (
    LocalModelError,
    generate_vision,
    generate_vision_structured,
)

logger = logging.getLogger(__name__)


def _normalize_conf(value: object) -> Optional[float]:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    if not 0 <= conf <= 1:
        return None
    return round(min(conf, 0.99), 2)


_VECTOR_CAPTION_PROMPT = """
You are a vision captioning engine for a photo search system.

Fill two fields in the response:
- "description": ONE long, information-dense sentence that includes:
  • primary subjects (people/animals/objects)
  • their clothing, appearance, and colors
  • their actions or poses
  • spatial relationships between subjects and key objects
  • important secondary objects
  • environment and background (location type, notable scenery)
  • any visible text (or explicitly say "no readable text")
  • lighting and overall mood (time of day, indoor/outdoor, bright/dim, etc.)
- "confidence": a number between 0 and 1 representing how confident you are
  that the description accurately reflects the image.

Rules:
- Use only concrete visible facts; no speculation or story-telling.
- Do NOT mention camera settings or metadata.
- The description should be at least 40 words and preferably 60-100 words.
""".strip()


def _build_caption_prompt(context: str | None) -> str:
    if context:
        return (
            _VECTOR_CAPTION_PROMPT
            + "\n\nOptional context (may help interpret, never repeat verbatim):\n"
            + context
        )
    return _VECTOR_CAPTION_PROMPT


def describe_photo(
    photo: PhotoFile, exif: ExifData | None = None, context: str | None = None
) -> Optional[VisionDescription]:
    """
    Describe a photo using the local vision model.

    Null fallback policy:
    - If the model is missing → LocalModelError
    - If the model output cannot be parsed → LocalModelError
    - No heuristic fallbacks.
    """
    model_name = os.getenv("OLLAMA_VISION_MODEL")
    if not model_name:
        return None

    prompt = _build_caption_prompt(context)

    schema = {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["description"],
        "additionalProperties": True,
    }

    try:
        structured, raw = generate_vision_structured(prompt, Path(photo.path), schema=schema)
    except Exception:
        structured = None
        raw = None

    if structured and isinstance(structured, dict):
        text = str(structured.get("description") or "").strip()
        confidence = _normalize_conf(structured.get("confidence"))
        if text:
            logger.debug(
                "Vision captioner structured output\nImage: %s\nPrompt:\n%s\nOutput:\n%s",
                photo.path,
                prompt,
                structured,
            )
            return VisionDescription(
                photo_id=photo.id,
                description=text,
                model=model_name,
                confidence=confidence,
                user_context=context,
            )

    try:
        raw_text = raw if raw is not None else generate_vision(prompt, Path(photo.path))
    except Exception:
        return None

    logger.debug(
        "Vision captioner raw output\nImage: %s\nPrompt:\n%s\nOutput:\n%s",
        photo.path,
        prompt,
        raw_text,
    )

    text, confidence = _parse_caption(raw_text)

    return VisionDescription(
        photo_id=photo.id,
        description=text,
        model=model_name,
        confidence=confidence,
    )


def _parse_caption(raw: str) -> Tuple[str, float]:
    """
    Parse caption model output.
    Returns (description, confidence)
    Raises LocalModelError on failure.
    """
    text = raw.strip()

    # Strip markdown fences
    if text.startswith("```"):
        parts = text.split("\n", 1)
        text = parts[1] if len(parts) > 1 else ""
    if "```" in text:
        text = text.split("```", 1)[0]

    text = text.strip()

    # Try JSON (future-proof)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            desc = str(parsed.get("description") or parsed.get("caption") or "").strip()
            conf = _normalize_conf(parsed.get("confidence"))
            if desc:
                return desc, conf if conf is not None else 0.8
    except json.JSONDecodeError:
        pass

    # Try extracting confidence tokens
    match = re.search(r"confidence\s*[:=]\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
    confidence = None
    if match:
        confidence = _normalize_conf(match.group(1))
        text = re.sub(
            r"confidence\s*[:=]\s*([0-9]*\.?[0-9]+)", "", text, flags=re.IGNORECASE
        ).strip()

    # Remove leading "description:" junk
    text = re.sub(r"^description\s*[:=\-]\s*", "", text, flags=re.IGNORECASE).strip()

    # Remove wrapping quotes
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()

    if not text:
        raise LocalModelError(f"Caption parse failed. Raw output:\n{raw}")

    return text, confidence if confidence is not None else 0.8
