from __future__ import annotations

import math
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from photo_brain.core.models import TextEmbedding, VectorMatch

from .schema import TextEmbeddingRow


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    length = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(length))
    mag_a = math.sqrt(sum(a[i] * a[i] for i in range(length)))
    mag_b = math.sqrt(sum(b[i] * b[i] for i in range(length)))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class PgVectorBackend:
    """Vector backend backed by the text_embeddings table (pgvector friendly)."""

    def upsert_embedding(
        self, session: Session, embedding: TextEmbedding
    ) -> TextEmbeddingRow:
        vector = list(embedding.vector)
        dim = len(vector)
        existing = session.scalar(
            select(TextEmbeddingRow).where(
                TextEmbeddingRow.photo_id == embedding.photo_id,
                TextEmbeddingRow.model == embedding.model,
            )
        )
        if existing:
            existing.embedding = vector
            existing.dim = dim
            row = existing
        else:
            row = TextEmbeddingRow(
                photo_id=embedding.photo_id,
                model=embedding.model,
                dim=dim,
                embedding=vector,
            )
            session.add(row)
        session.flush()
        return row

    def search(
        self,
        session: Session,
        query_vector: Sequence[float],
        *,
        limit: int = 10,
        model: str | None = None,
    ) -> list[VectorMatch]:
        rows = session.scalars(select(TextEmbeddingRow)).all()
        matches: list[VectorMatch] = []
        for row in rows:
            if model and row.model != model:
                continue
            score = _cosine_similarity(query_vector, row.embedding or [])
            matches.append(VectorMatch(photo_id=row.photo_id, score=score))
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:limit]
