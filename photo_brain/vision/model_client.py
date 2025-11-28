from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
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

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "images": [_encode_image(image_path)],
    }
    response = _post_json(f"{_base_url().rstrip('/')}/api/generate", payload)
    text = response.get("response") or response.get("content")
    if not text:
        raise LocalModelError("No response text from vision model")
    return str(text).strip()


def classify_vision(prompt: str, image_path: Path) -> List[str]:
    """Ask the local vision model for tags; returns a list of labels."""
    raw = generate_vision(prompt, image_path)
    # Try to parse JSON list first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        if isinstance(parsed, dict) and "labels" in parsed and isinstance(parsed["labels"], list):
            return [str(item).strip() for item in parsed["labels"] if str(item).strip()]
    except json.JSONDecodeError:
        pass
    # Fallback: split by commas
    labels = [part.strip() for part in raw.split(",") if part.strip()]
    return labels


def embed_text(text: str) -> List[float]:
    """Call a local embedding model via Ollama."""
    model = _embed_model()
    if not model:
        raise LocalModelError("OLLAMA_EMBED_MODEL not set")
    payload = {"model": model, "input": text}
    # Ollama embeds endpoint
    response = _post_json(f"{_base_url().rstrip('/')}/api/embed", payload)
    vector = response.get("embedding") or response.get("vector")
    if not isinstance(vector, list):
        raise LocalModelError("No embedding returned from model")
    return [float(v) for v in vector]
