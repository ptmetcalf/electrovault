from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from sqlalchemy import select

from photo_brain import faces
from photo_brain.core.models import FaceDetection, PhotoFile
from photo_brain.faces import detect_faces, recognize_faces
from photo_brain.index import (
    FaceDetectionRow,
    PhotoFileRow,
    init_db,
    index_photo,
    load_photo_record,
    session_factory,
)
from photo_brain.index import indexer
from photo_brain.index.updates import assign_face_identity
from photo_brain.index.vector_backend import PgVectorBackend


def _sample_photo(tmp_path: Path) -> PhotoFile:
    img_path = tmp_path / "face.jpg"
    Image.new("RGB", (10, 10), color="green").save(img_path)
    return PhotoFile(
        id="face1",
        path=str(img_path),
        sha256="x" * 64,
        size_bytes=img_path.stat().st_size,
        mtime=datetime.now(timezone.utc),
    )


def test_detect_and_recognize_faces(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FACE_DETECT_ALLOW_FALLBACK", "1")
    photo = _sample_photo(tmp_path)
    detections = detect_faces(photo)
    assert detections, "Expected at least one detection (fallback allowed in tests)"
    assert detections[0].encoding is not None

    identities = recognize_faces(detections)
    assert identities
    assert identities[0].person_id.startswith("person-")


def test_face_detector_second_pass_threshold(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "multi.jpg"
    # Simple blank image is sufficient; detections are faked via the net.
    Image.new("RGB", (100, 100), color="white").save(image_path)

    class FakeNet:
        def __init__(self, detections):
            self._dets = detections
            self.calls = 0

        def setInput(self, _blob):
            return None

        def forward(self):
            self.calls += 1
            return self._dets

    detections = np.zeros((1, 1, 1, 7), dtype=float)
    detections[0, 0, 0, 2] = 0.35  # below default threshold 0.5 but above 0.2
    detections[0, 0, 0, 3] = 0.1
    detections[0, 0, 0, 4] = 0.1
    detections[0, 0, 0, 5] = 0.5
    detections[0, 0, 0, 6] = 0.5

    fake_net = FakeNet(detections)
    monkeypatch.setattr(faces.detector, "_load_net", lambda: fake_net)
    monkeypatch.setattr(faces.detector, "_CONF_THRESHOLD", 0.6)

    photo = PhotoFile(
        id="multi",
        path=str(image_path),
        sha256="x" * 64,
        size_bytes=image_path.stat().st_size,
        mtime=datetime.now(timezone.utc),
    )

    results = detect_faces(photo)
    assert results  # second-pass should rescue this
    assert fake_net.calls >= 2


def test_face_matching_reuses_named_person(tmp_path: Path, monkeypatch) -> None:
    image_path1 = tmp_path / "face1.jpg"
    image_path2 = tmp_path / "face2.jpg"
    Image.new("RGB", (10, 10), color="white").save(image_path1)
    Image.new("RGB", (10, 10), color="gray").save(image_path2)

    engine = init_db("sqlite+pysqlite:///:memory:")
    SessionLocal = session_factory(engine)
    backend = PgVectorBackend()

    encoding = [0.1, 0.2, 0.3, 0.4]

    def fake_detect(photo: PhotoFile) -> list[FaceDetection]:
        return [
            FaceDetection(
                photo_id=photo.id,
                bbox=(0.0, 0.0, 5.0, 5.0),
                confidence=0.9,
                encoding=list(encoding),
            )
        ]

    monkeypatch.setattr(indexer, "detect_faces", fake_detect)
    # Skip heavy model calls
    monkeypatch.setattr(indexer, "describe_photo", lambda *_, **__: None, raising=False)
    monkeypatch.setattr(indexer, "classify_photo", lambda *_, **__: [], raising=False)

    with SessionLocal() as session:
        row1 = PhotoFileRow(
            id="p1",
            path=str(image_path1),
            sha256="a" * 64,
            size_bytes=image_path1.stat().st_size,
            mtime=datetime.now(timezone.utc),
        )
        row2 = PhotoFileRow(
            id="p2",
            path=str(image_path2),
            sha256="b" * 64,
            size_bytes=image_path2.stat().st_size,
            mtime=datetime.now(timezone.utc),
        )
        session.add_all([row1, row2])
        session.commit()

        index_photo(session, row1, backend=backend, skip_if_fresh=False, preserve_faces=False)
        det_id = session.scalar(select(FaceDetectionRow.id).where(FaceDetectionRow.photo_id == "p1"))
        assert det_id is not None
        assign_face_identity(session, detection_id=det_id, person_label="Alice")
        session.commit()

        index_photo(session, row2, backend=backend, skip_if_fresh=False, preserve_faces=False)
        record = load_photo_record(session, "p2")
        assert record is not None
        # New behavior: faces remain unassigned until explicitly labeled/grouped.
        assert record.faces == []
