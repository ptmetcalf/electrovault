from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator
from photo_brain.vision.taxonomy import map_label


class LocalModelError(RuntimeError):
    """Raised when a local model call fails."""


def _post_json(url: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        raise LocalModelError(f"Model call failed: {exc}") from exc


def _encode_image(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def _base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _vision_model() -> Optional[str]:
    return os.getenv("OLLAMA_VISION_MODEL")


def _embed_model() -> Optional[str]:
    return os.getenv("OLLAMA_EMBED_MODEL")


def generate_vision(prompt: str, image_path: Path) -> str:
    """Call a local vision-capable model (e.g., LLaVA on Ollama) to get a caption."""
    model = _vision_model()
    if not model:
        raise LocalModelError("OLLAMA_VISION_MODEL not set")

    timeout = int(os.getenv("OLLAMA_HTTP_TIMEOUT", "60"))
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "images": [_encode_image(image_path)],
    }
    response = _post_json(f"{_base_url().rstrip('/')}/api/generate", payload, timeout=timeout)
    text = response.get("response") or response.get("content")
    if not text:
        raise LocalModelError("No response text from vision model")
    return str(text).strip()


def _normalize_label(label: str) -> Optional[str]:
    """Normalize labels to short, lower-case tokens; drop sentence-like or noisy entries."""
    raw = (
        label.strip()
        .strip("`")
        .strip("{}[]\"'")
        .replace("\n", " ")
        .strip()
    )
    if not raw:
        return None

    # Preserve prefix (e.g., scene:, object:, quality:) while normalizing the value.
    prefix = None
    value = raw
    if ":" in raw:
        parts = raw.split(":", 1)
        prefix = parts[0].strip().lower()
        value = parts[1].strip()
        # Prefix must be short and alpha-ish.
        if not prefix or len(prefix) > 24 or not re.fullmatch(r"[a-z0-9\-]+", prefix):
            return None

    # Clean value: keep letters/numbers/spaces/hyphens/underscores.
    value = re.sub(r"[{}\\[\\]`]", " ", value)
    value = re.sub(r"[_]", " ", value)
    value = re.sub(r"\s+", " ", value).strip().lower()
    if not value:
        return None
    # Reject tokens that are essentially numbers or boilerplate.
    if value in {"confidence", "label", "labels", "json", "object", "objects", "probability", "score", "example"}:
        return None
    if re.fullmatch(r"\d+(\.\d+)?", value):
        return None

    words = value.split()
    # Enforce concise labels (max 4 words) to avoid sentences.
    if len(words) == 0 or len(words) > 4:
        return None
    if prefix and len(words) > 2:
        return None
    # Require majority alphabetic characters.
    alpha = sum(ch.isalpha() for ch in value)
    if alpha < 2 or alpha < len(value) * 0.5:
        return None
    if len(value) > 40:
        value = value[:40].rstrip()

    if prefix:
        return f"{prefix}:{value}"
    return value


def _normalize_conf(value: Any) -> Optional[float]:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    if conf < 0 or conf > 1:
        return None
    conf = min(conf, 0.99)
    return round(conf, 2)


class _LLMTag(BaseModel):
    label: str
    confidence: Optional[float] = None

    @field_validator("label")
    @classmethod
    def _norm_label(cls, v: str) -> str:
        normalized = _normalize_label(v)
        if not normalized:
            raise ValueError("empty label")
        return normalized

    @field_validator("confidence")
    @classmethod
    def _norm_conf(cls, v: Any) -> Optional[float]:
        conf = _normalize_conf(v)
        if v is not None and conf is None:
            raise ValueError("invalid confidence")
        return conf


class _LLMResponse(BaseModel):
    labels: List[_LLMTag] = Field(default_factory=list)
    scene: List[str] = Field(default_factory=list)
    objects: List[_LLMTag] = Field(default_factory=list)
    activities: List[str] = Field(default_factory=list)
    events: List[str] = Field(default_factory=list)
    colors: List[str] = Field(default_factory=list)
    brands: List[str] = Field(default_factory=list)
    time_of_day: Optional[str] = None
    weather: Optional[str] = None
    ocr_text: List[str] = Field(default_factory=list)
    people: Dict[str, Any] = Field(default_factory=dict)
    counts: Dict[str, Any] = Field(default_factory=dict)
    quality: Dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "scene",
        "activities",
        "events",
        "colors",
        "brands",
        mode="before",
    )
    @classmethod
    def _norm_list(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            return []
        cleaned: List[str] = []
        for item in v:
            if isinstance(item, str):
                norm = _normalize_label(item)
                if norm:
                    cleaned.append(norm)
        return cleaned[:12]

    @field_validator("ocr_text", mode="before")
    @classmethod
    def _norm_ocr(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [v.strip()] if v.strip() else []
        if not isinstance(v, list):
            return []
        out: List[str] = []
        for item in v:
            if isinstance(item, str) and item.strip():
                text = item.replace("\n", " ").strip()
                if len(text) > 80:
                    text = text[:80].rstrip()
                out.append(text)
        return out[:5]

    @field_validator("time_of_day", "weather", mode="before")
    @classmethod
    def _norm_optional_label(cls, v: Any) -> Optional[str]:
        if not isinstance(v, str):
            return None
        return _normalize_label(v)


def classify_vision(prompt: str, image_path: Path) -> Tuple[List[Tuple[str, Optional[float]]], List[str]]:
    """Call the vision model, require schema-conformant JSON, and return normalized labels + OCR."""
    raw = generate_vision(prompt, image_path)

    def _strip_wrappers(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            parts = text.split("\n", 1)
            text = parts[1] if len(parts) > 1 else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        return text.strip()

    text = _strip_wrappers(raw)
    if not text:
        raise LocalModelError("Empty response from vision model")

    parsed: Any = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None

    validated: _LLMResponse | None = None
    if isinstance(parsed, dict):
        try:
            validated = _LLMResponse.model_validate(parsed)
        except ValidationError:
            validated = None

    results: List[Tuple[str, Optional[float]]] = []
    ocr_texts: List[str] = validated.ocr_text if validated else []
    seen: set[str] = set()
    max_labels = 20

    def _add_label(cat: Optional[str], val: str, conf: Any = None) -> None:
        if len(results) >= max_labels:
            return
        mapped = map_label(cat, val)
        if not mapped:
            return
        m_cat, m_val = mapped
        key = f"{m_cat}:{m_val}" if m_cat else m_val
        if key in seen:
            return
        seen.add(key)
        results.append((key, _normalize_conf(conf)))

    if validated:
        for tag in validated.labels:
            _add_label(None, tag.label, tag.confidence)

        for tag in validated.objects:
            _add_label("object", tag.label, tag.confidence)

        for scene in validated.scene:
            _add_label("scene", scene, 0.8)
        for act in validated.activities:
            _add_label("activity", act, 0.8)
        for event in validated.events:
            _add_label("event", event, 0.8)
        for color in validated.colors:
            _add_label("color", color, 0.6)
        for brand in validated.brands:
            _add_label("brand", brand, 0.8)
        if validated.time_of_day:
            _add_label("time", validated.time_of_day, 0.7)
        if validated.weather:
            _add_label("weather", validated.weather, 0.7)

        people = validated.people or {}
        count = people.get("count")
        if isinstance(count, int):
            _add_label("people-count", str(count), 0.9)
        for attr in people.get("attributes", []) or []:
            if isinstance(attr, str):
                _add_label("people-attr", attr, 0.7)
        for age in people.get("age_bands", []) or []:
            if isinstance(age, str):
                _add_label("age-band", age, 0.7)
        for gender in people.get("genders", []) or []:
            if isinstance(gender, str):
                _add_label("gender", gender, 0.7)

        counts = validated.counts or {}
        pets = counts.get("pets")
        if isinstance(pets, int):
            _add_label("pets-count", str(pets), 0.9)

        quality = validated.quality or {}
        blur = quality.get("blur")
        blur_conf = _normalize_conf(blur)
        if blur_conf is not None:
            if blur_conf >= 0.66:
                _add_label("quality", "blur-high", blur_conf)
            elif blur_conf >= 0.33:
                _add_label("quality", "blur-medium", blur_conf)
            else:
                _add_label("quality", "blur-low", blur_conf)
        lighting = quality.get("lighting")
        if isinstance(lighting, str):
            _add_label("quality", f"lighting-{lighting}", 0.7)
        composition = quality.get("composition")
        if isinstance(composition, list):
            for comp in composition:
                if isinstance(comp, str):
                    _add_label("quality", f"composition-{comp}", 0.7)

    # Fallback parsing if structured JSON fails to yield labels.
    if not results:
        tokens = re.split(r"[,\n]", text)
        for tok in tokens:
            tok = tok.strip()
            if not tok:
                continue
            if ":" in tok:
                cat, val = tok.split(":", 1)
                _add_label(cat.strip(), val.strip(), None)
            else:
                _add_label(None, tok, None)

    if not results:
        raise LocalModelError("No usable labels from vision model")

    return results, ocr_texts


def embed_text(text: str) -> List[float]:
    """Call a local embedding model via Ollama."""
    model = _embed_model()
    if not model:
        raise LocalModelError("OLLAMA_EMBED_MODEL not set")
    payload = {"model": model, "input": text}
    response = _post_json(f"{_base_url().rstrip('/')}/api/embed", payload)
    vector = response.get("embedding") or response.get("vector")
    if not isinstance(vector, list):
        raise LocalModelError("No embedding returned from model")
    return [float(v) for v in vector]
