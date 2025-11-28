from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

from photo_brain.core.models import ExifData, PhotoFile, VisionDescription
from photo_brain.vision.model_client import LocalModelError, generate_vision


def _fallback_caption(photo: PhotoFile, exif: ExifData | None) -> VisionDescription:
    """Simple filename + date heuristic caption if no model is configured or it fails."""
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


def _normalize_conf(value: object) -> Optional[float]:
    try:
        conf = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if conf < 0 or conf > 1:
        return None
    return round(min(conf, 0.99), 2)


_VECTOR_CAPTION_PROMPT = """
You are creating a caption for vector-based image similarity search.

Describe the image in ONE information-dense sentence, optimized for retrieval.

Follow this order in the sentence:
1. Primary subjects
2. Their attributes and appearance (including clothing and colors)
3. Their actions or poses
4. Spatial relationships between subjects and key objects
5. Secondary subjects or important objects
6. Environment and background details
7. Any visible text in the scene
8. Lighting and overall mood

Constraints:
- Avoid speculation and camera metadata.
- Avoid story-like prose.
- Do NOT list items; write a single coherent sentence.
- Output ONLY the final sentence, no bullet points, no JSON, no markdown, no commentary.
""".strip()


def _build_caption_prompt(context: str | None) -> str:
    """Inject optional context while keeping the output format strictly text-only."""
    if context:
        return (
            _VECTOR_CAPTION_PROMPT
            + "\n\nOptional context about the photo "
            "(may help interpret the scene; do not repeat verbatim):\n"
            + context
        )
    return _VECTOR_CAPTION_PROMPT


def describe_photo(
    photo: PhotoFile, exif: ExifData | None = None, context: str | None = None
) -> VisionDescription:
    """
    Describe a photo using a local vision model if configured, else fallback heuristic.

    The caption is optimized for vector search: a single, dense sentence that encodes
    subjects, attributes, relationships, environment, visible text, and mood.
    """

    def _parse_model_caption(raw: str) -> Tuple[str, float]:
        """
        Extract a usable description and an optional confidence from the model output.

        - Strips code fences if the model accidentally returns markdown.
        - Tries JSON first (for forward-compatibility if a JSON prompt is used later).
        - Falls back to treating the cleaned text as the description.
        """
        text = raw.strip()

        # Strip leading and trailing code fences if present
        if text.startswith("```"):
            # Remove the first line (``` or ```json)
            parts = text.split("\n", 1)
            text = parts[1] if len(parts) > 1 else ""
        if "```" in text:
            # Remove everything after the closing fence
            text = text.split("```", 1)[0]

        text = text.strip()

        import json
        import re

        # Try JSON first: {"description": "...", "confidence": 0.87}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                desc = str(
                    parsed.get("description") or parsed.get("caption") or ""
                ).strip()
                confidence_val = _normalize_conf(parsed.get("confidence"))
                if desc:
                    return desc, confidence_val if confidence_val is not None else 0.8
        except json.JSONDecodeError:
            pass

        # Optional: try to pull a "confidence: 0.87" style token out of plain text
        confidence: Optional[float] = None
        match = re.search(
            r"confidence\s*[:=]\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE
        )
        if match:
            confidence = _normalize_conf(match.group(1))
            text = re.sub(
                r"confidence\s*[:=]\s*([0-9]*\.?[0-9]+)",
                "",
                text,
                flags=re.IGNORECASE,
            )

        # Remove leading description labels and surrounding quotes if present
        text = re.sub(
            r"^description\s*[:=\-]\s*", "", text, flags=re.IGNORECASE
        ).strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1].strip()

        desc = text.strip()
        if not desc:
            raise LocalModelError("Vision model response missing description")

        # Default confidence if none could be parsed
        return desc, confidence if confidence is not None else 0.8

    if os.getenv("OLLAMA_VISION_MODEL"):
        try:
            prompt = _build_caption_prompt(context)
            raw = generate_vision(prompt, Path(photo.path))
            text, confidence = _parse_model_caption(raw)
            return VisionDescription(
                photo_id=photo.id,
                description=text,
                model=os.getenv("OLLAMA_VISION_MODEL"),
                confidence=confidence,
            )
        except LocalModelError:
            # Fallback to deterministic caption if local model fails/unavailable.
            return _fallback_caption(photo, exif)

    return _fallback_caption(photo, exif)
