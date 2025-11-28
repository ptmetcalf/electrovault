from datetime import datetime, timezone

from photo_brain.core.models import TextEmbedding
from photo_brain.index import PhotoFileRow, init_db, session_factory
from photo_brain.index.vector_backend import PgVectorBackend


def test_vector_backend_upsert_and_search() -> None:
    engine = init_db("sqlite+pysqlite:///:memory:")
    SessionLocal = session_factory(engine)
    backend = PgVectorBackend()

    with SessionLocal() as session:
        session.add_all(
            [
                PhotoFileRow(
                    id="p1",
                    path="/tmp/p1.jpg",
                    sha256="1" * 64,
                    size_bytes=10,
                    mtime=datetime.now(timezone.utc),
                ),
                PhotoFileRow(
                    id="p2",
                    path="/tmp/p2.jpg",
                    sha256="2" * 64,
                    size_bytes=12,
                    mtime=datetime.now(timezone.utc),
                ),
            ]
        )
        session.commit()

        backend.upsert_embedding(
            session,
            TextEmbedding(photo_id="p1", model="mock", vector=[1.0, 0.0], dim=2),
        )
        backend.upsert_embedding(
            session,
            TextEmbedding(photo_id="p2", model="mock", vector=[0.0, 1.0], dim=2),
        )
        session.commit()

        matches = backend.search(session, [0.9, 0.1], limit=1, model="mock")
        assert matches
        assert matches[0].photo_id == "p1"
        assert matches[0].score > 0
