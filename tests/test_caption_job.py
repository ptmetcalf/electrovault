from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from photo_brain.core.models import TextEmbedding, VisionDescription
from photo_brain.index import caption_photo, init_db, session_factory
from photo_brain.index import indexer as indexer_mod
from photo_brain.index.schema import (
    FaceDetectionRow,
    FaceIdentityRow,
    FacePersonLinkRow,
    PersonRow,
    PhotoFileRow,
    TextEmbeddingRow,
    VisionDescriptionRow,
)


def test_caption_job_updates_vision_and_embedding(monkeypatch, tmp_path: Path) -> None:
    db = init_db("sqlite+pysqlite:///:memory:")
    SessionLocal = session_factory(db)

    captured_context: dict[str, Any] = {}

    def fake_describe(photo: Any, exif: Any, context: str | None = None) -> VisionDescription:
        captured_context["value"] = context
        return VisionDescription(
            photo_id=photo.id,
            description="rich caption",
            model="mock-vision",
            confidence=0.9,
        )

    def fake_embed(text: str, photo_id: str | None = None, model: str = "mock-embed", dim: int = 4) -> TextEmbedding:
        return TextEmbedding(photo_id=photo_id or "p", model=model, vector=[1.0, 0.0, 0.0, 0.0], dim=4)

    monkeypatch.setattr(indexer_mod, "describe_photo", fake_describe)
    monkeypatch.setattr(indexer_mod, "embed_description", fake_embed)

    with SessionLocal() as session:
        photo = PhotoFileRow(
            id="p1",
            path=str(tmp_path / "p1.jpg"),
            sha256="x" * 64,
            size_bytes=123,
            mtime=datetime.now(timezone.utc),
        )
        session.add(photo)
        session.flush()
        det = FaceDetectionRow(
            photo_id=photo.id,
            bbox_x1=0.0,
            bbox_y1=0.0,
            bbox_x2=10.0,
            bbox_y2=10.0,
            confidence=0.9,
            encoding=[0.1, 0.2, 0.3, 0.4],
        )
        session.add(det)
        session.flush()
        person = PersonRow(id="alice", display_name="Alice")
        session.add(person)
        session.flush()
        session.add(FacePersonLinkRow(detection_id=det.id, person_id=person.id))
        session.add(FaceIdentityRow(detection_id=det.id, person_label=person.display_name))
        session.add(VisionDescriptionRow(photo_id=photo.id, description="", model=None, user_context="user note"))
        session.commit()

        vision = caption_photo(session, photo, context=None)
        session.commit()

        stored = session.get(VisionDescriptionRow, photo.id)
        assert vision is not None
        assert stored is not None
        assert stored.description == "rich caption"
        assert stored.user_context == "user note"
        assert captured_context["value"] and "Alice" in captured_context["value"]

        embedding_count = session.scalar(
            select(func.count()).select_from(TextEmbeddingRow).where(TextEmbeddingRow.photo_id == photo.id)
        )
        assert embedding_count == 1
