from datetime import datetime, timezone

from PIL import Image
from sqlalchemy import select

from photo_brain.embedding import embed_description
from photo_brain.events import group_events
from photo_brain.index import (
    ExifDataRow,
    FaceIdentityRow,
    FaceDetectionRow,
    PhotoFileRow,
    event_photos,
    init_db,
    session_factory,
)
from photo_brain.index.vector_backend import PgVectorBackend
from photo_brain.ingest import ingest_and_index
from photo_brain.search import execute_search, plan_search
from photo_brain.index.updates import assign_face_identity


def test_plan_search_trims_text() -> None:
    query = plan_search(" hello person:alice after:2024-01-01", limit=5)
    assert query.text == "hello"
    assert query.limit == 5
    assert query.people == ["alice"]
    assert query.start_date.year == 2024


def test_execute_search_returns_match() -> None:
    engine = init_db("sqlite+pysqlite:///:memory:")
    SessionLocal = session_factory(engine)
    backend = PgVectorBackend()

    with SessionLocal() as session:
        photo = PhotoFileRow(
            id="p1",
            path="/tmp/p1.jpg",
            sha256="1" * 64,
            size_bytes=10,
            mtime=datetime.now(timezone.utc),
        )
        session.add(photo)
        emb = embed_description("a cat", photo_id=photo.id)
        backend.upsert_embedding(session, emb)
        session.commit()

        results = execute_search(session, backend, plan_search("a cat"))
        assert results
        assert results[0].record.file.id == "p1"


def test_execute_search_filters_by_date() -> None:
    engine = init_db("sqlite+pysqlite:///:memory:")
    SessionLocal = session_factory(engine)
    backend = PgVectorBackend()

    with SessionLocal() as session:
        photo = PhotoFileRow(
            id="p1",
            path="/tmp/p1.jpg",
            sha256="1" * 64,
            size_bytes=10,
            mtime=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        session.add(photo)
        session.add(
            ExifDataRow(
                photo_id=photo.id,
                datetime_original=datetime(2020, 1, 1, tzinfo=timezone.utc),
                gps_lat=None,
                gps_lon=None,
            )
        )
        emb = embed_description("winter day", photo_id=photo.id)
        backend.upsert_embedding(session, emb)
        session.commit()

        results = execute_search(session, backend, plan_search("winter day after:2021-01-01"))
        assert results == []


def test_execute_search_filters_by_person(tmp_path) -> None:
    engine = init_db("sqlite+pysqlite:///:memory:")
    SessionLocal = session_factory(engine)
    backend = PgVectorBackend()
    image_path = tmp_path / "portrait.jpg"
    Image.new("RGB", (8, 8), color="yellow").save(image_path)

    with SessionLocal() as session:
        ingest_and_index(tmp_path, session, backend=backend)
        photo_row = session.scalar(select(PhotoFileRow))
        assert photo_row is not None
        detection_id = session.scalar(select(FaceDetectionRow.id))
        if detection_id is None:
            det = FaceDetectionRow(
                photo_id=photo_row.id,
                bbox_x1=0.0,
                bbox_y1=0.0,
                bbox_x2=1.0,
                bbox_y2=1.0,
                confidence=0.9,
                encoding=[0.1, 0.2, 0.3, 0.4],
            )
            session.add(det)
            session.flush()
            detection_id = det.id
        identity = assign_face_identity(session, detection_id=detection_id, person_label="Alice")
        # Provide a fallback embedding so search can return results when vision is unavailable.
        backend.upsert_embedding(session, embed_description("portrait", photo_id=photo_row.id))
        session.commit()

        query = plan_search(f"portrait person:{identity.person_label}")
        results = execute_search(session, backend, query)
        assert any(result.record.file.id == photo_row.id for result in results)


def test_execute_search_filters_by_event() -> None:
    engine = init_db("sqlite+pysqlite:///:memory:")
    SessionLocal = session_factory(engine)
    backend = PgVectorBackend()

    with SessionLocal() as session:
        p1 = PhotoFileRow(
            id="p1",
            path="/tmp/p1.jpg",
            sha256="1" * 64,
            size_bytes=10,
            mtime=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        p2 = PhotoFileRow(
            id="p2",
            path="/tmp/p2.jpg",
            sha256="2" * 64,
            size_bytes=10,
            mtime=datetime(2020, 1, 3, tzinfo=timezone.utc),
        )
        session.add_all([p1, p2])
        backend.upsert_embedding(session, embed_description("alpha", photo_id="p1"))
        backend.upsert_embedding(session, embed_description("beta", photo_id="p2"))
        session.commit()

        group_events(session, gap_hours=12)
        event_id = session.execute(
            select(event_photos.c.event_id).where(event_photos.c.photo_id == "p1")
        ).scalar_one()

        results = execute_search(session, backend, plan_search(f"alpha event:{event_id}"))
        assert results
        assert results[0].record.file.id == "p1"
