from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from photo_brain.index.face_grouping import (
    accept_face_group,
    list_face_group_proposals,
    rebuild_face_group_proposals,
)
from photo_brain.index.schema import (
    Base,
    FaceDetectionRow,
    FacePersonLinkRow,
    PersonRow,
    PhotoFileRow,
    create_engine_from_url,
    session_factory,
)


def _make_session():
    engine = create_engine_from_url("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return session_factory(engine)()


def _add_detection(session, photo_id: str, encoding: list[float]) -> FaceDetectionRow:
    photo = session.get(PhotoFileRow, photo_id)
    if not photo:
        photo = PhotoFileRow(
            id=photo_id,
            path=f"{photo_id}.jpg",
            sha256="hash",
            size_bytes=123,
            mtime=datetime.now(timezone.utc),
        )
        session.add(photo)
    detection = FaceDetectionRow(
        photo_id=photo_id,
        bbox_x1=0,
        bbox_y1=0,
        bbox_x2=10,
        bbox_y2=10,
        confidence=0.9,
        encoding=encoding,
    )
    session.add(detection)
    session.flush()
    return detection


def test_rebuild_and_accept_groups():
    session = _make_session()
    det_a = _add_detection(session, "p1", [0.1, 0.2, 0.3, 0.4])
    det_b = _add_detection(session, "p2", [0.11, 0.21, 0.29, 0.41])  # similar to det_a
    _add_detection(session, "p3", [-0.5, 0.0, 0.4, -0.1])  # should not cluster with above
    session.commit()

    proposals = rebuild_face_group_proposals(session, threshold=0.8, unassigned_only=True)
    session.commit()
    assert proposals
    assert proposals[0].size == 2

    listed, total = list_face_group_proposals(session, status="pending")
    assert total == 1
    assert listed[0].size == 2
    member_ids = {m.detection.id for m in listed[0].members}
    assert det_a.id in member_ids and det_b.id in member_ids

    person = accept_face_group(session, proposals[0].id, target_label="Alice")
    session.commit()
    assert person.display_name == "Alice"
    links = session.scalars(select(FacePersonLinkRow)).all()
    assert len(links) == 2
    stored = session.get(PersonRow, person.id)
    assert stored is not None
