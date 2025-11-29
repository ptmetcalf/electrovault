from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from sqlalchemy import select

from photo_brain.index import (
    ClassificationRow,
    PhotoFileRow,
    TextEmbeddingRow,
    VisionDescriptionRow,
    init_db,
    session_factory,
)
from photo_brain.index.vector_backend import PgVectorBackend
from photo_brain.ingest import ingest_and_index


def _create_image(path: Path) -> None:
    img = Image.new("RGB", (5, 5), color="blue")
    img.save(path)


def test_ingest_and_index_populates_metadata(tmp_path: Path) -> None:
    photo_path = tmp_path / "portrait.jpg"
    _create_image(photo_path)
    engine = init_db("sqlite+pysqlite:///:memory:")
    SessionLocal = session_factory(engine)
    backend = PgVectorBackend()

    with SessionLocal() as session:
        ingest_and_index(tmp_path, session, backend=backend)
        photo = session.scalar(select(PhotoFileRow))
        assert photo is not None
        # Vision/classifications are skipped if no model configured.
        vision = session.scalar(select(VisionDescriptionRow))
        classes = session.scalars(select(ClassificationRow)).all()
        embedding = session.scalar(select(TextEmbeddingRow))
        assert vision is None
        assert not classes
        assert embedding is None
