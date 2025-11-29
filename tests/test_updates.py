from __future__ import annotations

from datetime import datetime, timezone

from photo_brain.index import (
    FaceDetectionRow,
    PhotoFileRow,
    VisionDescriptionRow,
    init_db,
    session_factory,
)
from photo_brain.index.updates import assign_face_identity, set_photo_user_context


def _session():
    engine = init_db("sqlite+pysqlite:///:memory:")
    return session_factory(engine)


def test_assign_face_identity_creates_and_updates() -> None:
    SessionLocal = _session()
    with SessionLocal() as session:
        photo = PhotoFileRow(
            id="p1",
            path="/tmp/p1.jpg",
            sha256="x" * 64,
            size_bytes=10,
            mtime=datetime.now(timezone.utc),
        )
        det = FaceDetectionRow(
            id=1,
            photo=photo,
            bbox_x1=0.0,
            bbox_y1=0.0,
            bbox_x2=1.0,
            bbox_y2=1.0,
            confidence=0.9,
            encoding=[],
        )
        session.add_all([photo, det])
        session.commit()

        identity = assign_face_identity(session, detection_id=1, person_label="alice")
        session.commit()
        assert identity.person_label == "alice"

        identity = assign_face_identity(session, detection_id=1, person_label="bob")
        session.commit()
        assert identity.person_label == "bob"


def test_set_photo_user_context_creates_vision_row() -> None:
    SessionLocal = _session()
    with SessionLocal() as session:
        photo = PhotoFileRow(
            id="p2",
            path="/tmp/p2.jpg",
            sha256="y" * 64,
            size_bytes=20,
            mtime=datetime.now(timezone.utc),
        )
        session.add(photo)
        session.commit()

        vision = set_photo_user_context(session, photo, "hello world")
        session.commit()
        assert isinstance(vision, VisionDescriptionRow)
        assert vision.user_context == "hello world"
