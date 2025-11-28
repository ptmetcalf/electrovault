from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

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


def _normalize_conf(value: object) -> Optional[float]:
    try:
        conf = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if conf < 0 or conf > 1:
        return None
    return round(min(conf, 0.99), 2)


def describe_photo(photo: PhotoFile, exif: ExifData | None = None) -> VisionDescription:
    """Describe a photo using a local vision model if configured, else fallback heuristic."""
    def _parse_model_caption(raw: str) -> Tuple[str, Optional[float]]:
        """Extract description and optional confidence from model output."""
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        import json

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:  # type: ignore[attr-defined]
            raise LocalModelError(f"Invalid JSON from vision model: {exc}") from exc

        if not isinstance(parsed, dict):
            raise LocalModelError("Vision model did not return a JSON object")

        desc = str(parsed.get("description") or parsed.get("caption") or "").strip()
        if not desc:
            raise LocalModelError("Vision model JSON missing description")
        confidence = _normalize_conf(parsed.get("confidence"))
        return desc, confidence

    if os.getenv("OLLAMA_VISION_MODEL"):
        try:
            prompt = (
                "Return JSON with a detailed photo description. Keys: "
                '"description" (3 compact sentences, ~60-90 words, photojournalistic detail: '
                "people/appearance/clothing colors/patterns, actions/poses, expressions, relative "
                "positions, surroundings/background/signage, lighting/mood, visible text), and "
                '"confidence" (0-1 float reflecting your certainty). Avoid speculation, avoid '
                "camera metadata."
            )
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
