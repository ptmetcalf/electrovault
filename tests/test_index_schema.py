from datetime import datetime, timezone

from sqlalchemy import select

from photo_brain.index import (
    ClassificationRow,
    ExifDataRow,
    FaceDetectionRow,
    FaceIdentityRow,
    MemoryEventRow,
    PhotoFileRow,
    TextEmbeddingRow,
    VisionDescriptionRow,
    init_db,
    session_factory,
)


def test_init_db_and_basic_crud() -> None:
    engine = init_db("sqlite+pysqlite:///:memory:")
    SessionLocal = session_factory(engine)

    with SessionLocal() as session:
        photo = PhotoFileRow(
            id="p1",
            path="/tmp/photo.jpg",
            sha256="a" * 64,
            size_bytes=100,
            mtime=datetime.now(timezone.utc),
        )
        session.add(photo)
        session.add(
            ExifDataRow(
                photo_id=photo.id,
                datetime_original=datetime(2020, 1, 1, tzinfo=timezone.utc),
                gps_lat=1.0,
                gps_lon=2.0,
            )
        )
        session.commit()

        stored = session.scalars(select(PhotoFileRow)).all()
        assert len(stored) == 1
        assert stored[0].exif is not None

        session.add(
            VisionDescriptionRow(
                photo_id=photo.id,
                description="a test scene",
                model="mock",
                confidence=0.9,
            )
        )
        session.add(
            TextEmbeddingRow(
                photo_id=photo.id,
                model="mock-embedder",
                dim=3,
                embedding=[0.1, 0.2, 0.3],
            )
        )
        session.add(
            ClassificationRow(
                photo_id=photo.id,
                label="outdoor",
                score=0.8,
                source="unit",
            )
        )
        detection = FaceDetectionRow(
            photo_id=photo.id,
            bbox_x1=0.0,
            bbox_y1=0.0,
            bbox_x2=0.5,
            bbox_y2=0.5,
            confidence=0.7,
            encoding=[0.1, 0.2, 0.3],
        )
        session.add(detection)
        session.flush()
        session.add(
            FaceIdentityRow(
                detection_id=detection.id,
                person_label="person-1",
                confidence=0.6,
            )
        )
        event = MemoryEventRow(
            id="evt1",
            title="Sample Event",
            summary="",
            start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2020, 1, 2, tzinfo=timezone.utc),
        )
        event.photos.append(photo)
        session.add(event)
        session.commit()

        fetched_embedding = session.scalar(
            select(TextEmbeddingRow).where(TextEmbeddingRow.photo_id == photo.id)
        )
        assert fetched_embedding is not None
        assert fetched_embedding.embedding == [0.1, 0.2, 0.3]

        fetched_event = session.scalar(select(MemoryEventRow).where(MemoryEventRow.id == "evt1"))
        assert fetched_event is not None
        assert fetched_event.photos[0].id == "p1"
