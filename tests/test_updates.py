from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from photo_brain.index import (
    FaceDetectionRow,
    FacePersonLinkRow,
    FaceIdentityRow,
    PersonRow,
    PhotoFileRow,
    VisionDescriptionRow,
    init_db,
    session_factory,
)
from photo_brain.index.updates import (
    assign_face_identity,
    merge_persons,
    rename_person,
    set_photo_user_context,
)


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
        person = session.scalar(select(PersonRow).where(PersonRow.display_name == "alice"))
        assert person is not None
        link = session.scalar(
            select(FacePersonLinkRow).where(FacePersonLinkRow.detection_id == det.id)
        )
        assert link is not None
        assert link.person_id == person.id

        identity = assign_face_identity(session, detection_id=1, person_label="bob")
        session.commit()
        assert identity.person_label == "bob"
        person_bob = session.scalar(select(PersonRow).where(PersonRow.display_name == "bob"))
        assert person_bob is not None


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


def test_rename_and_merge_persons() -> None:
    SessionLocal = _session()
    with SessionLocal() as session:
        person_a = PersonRow(id="a", display_name="Alice")
        person_b = PersonRow(id="b", display_name="Bob")
        photo = PhotoFileRow(
            id="p3",
            path="/tmp/p3.jpg",
            sha256="z" * 64,
            size_bytes=30,
            mtime=datetime.now(timezone.utc),
        )
        det = FaceDetectionRow(
            id=2,
            photo=photo,
            bbox_x1=0.0,
            bbox_y1=0.0,
            bbox_x2=1.0,
            bbox_y2=1.0,
            confidence=0.9,
            encoding=[],
        )
        session.add_all([person_a, person_b, photo, det])
        session.flush()
        session.add(FacePersonLinkRow(detection_id=det.id, person_id=person_a.id))
        session.add(FaceIdentityRow(detection_id=det.id, person_label=person_a.display_name))
        session.commit()

        renamed = rename_person(session, "a", "Alicia")
        session.commit()
        assert renamed.display_name == "Alicia"
        identity = session.scalar(select(FaceIdentityRow).where(FaceIdentityRow.detection_id == det.id))
        assert identity is not None and identity.person_label == "Alicia"

        merged = merge_persons(session, "a", "b")
        session.commit()
        assert merged.id == "b"
        link = session.scalar(select(FacePersonLinkRow).where(FacePersonLinkRow.detection_id == det.id))
        assert link is not None and link.person_id == "b"
