from datetime import datetime, timezone
from photo_brain.core.models import (
    Classification,
    PhotoFile,
    PhotoRecord,
    TextEmbedding,
    VisionDescription,
)


def test_photo_file_model() -> None:
    pf = PhotoFile(
        id="1",
        path="/tmp/photo.jpg",
        sha256="x" * 64,
        size_bytes=123,
        mtime=datetime.now(timezone.utc),
    )
    assert pf.path.endswith("photo.jpg")


def test_photo_record_models() -> None:
    vision = VisionDescription(
        photo_id="1", description="A test photo", model="mock", confidence=0.9
    )
    embedding = TextEmbedding(
        photo_id="1", model="mock-embedder", vector=[0.1, 0.2], dim=2
    )
    classification = Classification(
        photo_id="1", label="outdoor", score=0.8, source="unit"
    )
    record = PhotoRecord(
        file=PhotoFile(
            id="1",
            path="/tmp/photo.jpg",
            sha256="x" * 64,
            size_bytes=123,
            mtime=datetime.now(timezone.utc),
        ),
        vision=vision,
        classifications=[classification],
        embedding=embedding,
    )
    assert record.vision.description.startswith("A test")
    assert record.embedding.dim == len(record.embedding.vector)
