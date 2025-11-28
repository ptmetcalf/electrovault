from __future__ import annotations

import hashlib
import os
from typing import Optional

from photo_brain.core.models import TextEmbedding
from photo_brain.vision.model_client import LocalModelError, embed_text


def _hash_to_vector(text: str, dim: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vector = []
    for i in range(dim):
        byte = digest[i % len(digest)]
        # Map byte to range [-1, 1]
        vector.append((byte / 255.0) * 2 - 1)
    return vector


def embed_description(
    text: str, *, photo_id: Optional[str] = None, model: str = "hash-embedder", dim: int = 16
) -> TextEmbedding:
    normalized = text.strip().lower()

    if os.getenv("OLLAMA_EMBED_MODEL"):
        try:
            vector = embed_text(normalized)
            return TextEmbedding(
                photo_id=photo_id or "query",
                model=os.getenv("OLLAMA_EMBED_MODEL", "ollama-embed"),
                vector=vector,
                dim=len(vector),
            )
        except LocalModelError:
            pass

    vector = _hash_to_vector(normalized, dim)
    return TextEmbedding(
        photo_id=photo_id or "query",
        model=model,
        vector=vector,
        dim=len(vector),
    )
