from __future__ import annotations

import importlib
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image


def _setup_api(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "api.db"
    thumb_cache = tmp_path / "thumbs"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+pysqlite:///{db_path}")
    monkeypatch.setenv("THUMB_CACHE_DIR", str(thumb_cache))

    from photo_brain.api import http_api

    return importlib.reload(http_api)


def test_health_and_events(tmp_path: Path, monkeypatch) -> None:
    http_api = _setup_api(tmp_path, monkeypatch)
    client = TestClient(http_api.app)

    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

    resp = client.get("/events")
    assert resp.status_code == 200
    assert "events" in resp.json()


def test_thumbnail_endpoint_serves_image(tmp_path: Path, monkeypatch) -> None:
    http_api = _setup_api(tmp_path, monkeypatch)
    client = TestClient(http_api.app)

    image_path = tmp_path / "photo.jpg"
    Image.new("RGB", (12, 12), color="purple").save(image_path)

    with http_api.SessionLocal() as session:
        row = http_api.PhotoFileRow(
            id="photo-1",
            path=str(image_path),
            sha256="x" * 64,
            size_bytes=image_path.stat().st_size,
            mtime=datetime.now(timezone.utc),
        )
        session.add(row)
        session.commit()

    resp = client.get("/thumb/photo-1")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"


def test_full_image_endpoint_serves_original_file(tmp_path: Path, monkeypatch) -> None:
    http_api = _setup_api(tmp_path, monkeypatch)
    client = TestClient(http_api.app)

    image_path = tmp_path / "photo.jpg"
    Image.new("RGB", (8, 8), color="orange").save(image_path)

    with http_api.SessionLocal() as session:
        row = http_api.PhotoFileRow(
            id="photo-1",
            path=str(image_path),
            sha256="y" * 64,
            size_bytes=image_path.stat().st_size,
            mtime=datetime.now(timezone.utc),
        )
        session.add(row)
        session.commit()

    resp = client.get("/photos/photo-1/image")
    assert resp.status_code == 200
    assert resp.content == image_path.read_bytes()
    assert resp.headers["content-type"] == "image/jpeg"


def test_faces_listing_and_crop(tmp_path: Path, monkeypatch) -> None:
    http_api = _setup_api(tmp_path, monkeypatch)
    client = TestClient(http_api.app)

    image_path = tmp_path / "photo.jpg"
    Image.new("RGB", (10, 10), color="yellow").save(image_path)

    with http_api.SessionLocal() as session:
        photo = http_api.PhotoFileRow(
            id="photo-1",
            path=str(image_path),
            sha256="z" * 64,
            size_bytes=image_path.stat().st_size,
            mtime=datetime.now(timezone.utc),
        )
        session.add(photo)
        session.flush()
        det1 = http_api.FaceDetectionRow(
            photo_id=photo.id,
            bbox_x1=1,
            bbox_y1=1,
            bbox_x2=8,
            bbox_y2=8,
            confidence=0.9,
        )
        det2 = http_api.FaceDetectionRow(
            photo_id=photo.id,
            bbox_x1=2,
            bbox_y1=2,
            bbox_x2=9,
            bbox_y2=9,
            confidence=0.85,
        )
        session.add_all([det1, det2])
        session.flush()
        person = http_api.PersonRow(id="alice", display_name="Alice")
        session.add(person)
        session.flush()
        session.add(http_api.FacePersonLinkRow(detection_id=det2.id, person_id=person.id))
        session.add(
            http_api.FaceIdentityRow(
                detection_id=det2.id,
                person_label="Alice",
                confidence=0.88,
            )
        )
        session.commit()
        det1_id = det1.id
        det2_id = det2.id

    resp = client.get("/faces")
    payload = resp.json()
    assert resp.status_code == 200
    assert payload["total"] == 2
    assert len(payload["faces"]) == 2

    resp_unassigned = client.get("/faces?unassigned=true")
    unassigned = resp_unassigned.json()
    assert resp_unassigned.status_code == 200
    assert unassigned["total"] == 1
    assert unassigned["faces"][0]["detection"]["id"] == det1_id

    resp_named = client.get("/faces?person=alice")
    named = resp_named.json()
    assert resp_named.status_code == 200
    assert named["total"] == 1
    assert named["faces"][0]["detection"]["id"] == det2_id
    assert named["faces"][0]["identity"]["person_id"] == "alice"
    assert named["faces"][0]["identity"]["label"] == "Alice"

    resp_crop = client.get(f"/faces/{det1_id}/crop?size=64")
    assert resp_crop.status_code == 200
    assert resp_crop.headers["content-type"] == "image/jpeg"


def test_persons_endpoints(tmp_path: Path, monkeypatch) -> None:
    http_api = _setup_api(tmp_path, monkeypatch)
    client = TestClient(http_api.app)

    # Seed one person with a linked detection
    image_path = tmp_path / "photo2.jpg"
    Image.new("RGB", (8, 8), color="pink").save(image_path)
    with http_api.SessionLocal() as session:
        photo = http_api.PhotoFileRow(
            id="p1",
            path=str(image_path),
            sha256="p" * 64,
            size_bytes=image_path.stat().st_size,
            mtime=datetime.now(timezone.utc),
        )
        det = http_api.FaceDetectionRow(
            photo=photo,
            bbox_x1=0.0,
            bbox_y1=0.0,
            bbox_x2=7.0,
            bbox_y2=7.0,
            confidence=0.9,
            encoding=[0.1, 0.2, 0.3],
        )
        person = http_api.PersonRow(id="alice", display_name="Alice")
        session.add_all([photo, det, person])
        session.flush()
        session.add(http_api.FacePersonLinkRow(detection_id=det.id, person_id=person.id))
        session.add(http_api.FaceIdentityRow(detection_id=det.id, person_label="Alice"))
        session.commit()
        person_id = person.id

    resp = client.get("/persons")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["persons"][0]["display_name"] == "Alice"

    resp_rename = client.post(f"/persons/{person_id}/rename", json={"display_name": "Alicia"})
    assert resp_rename.status_code == 200
    assert resp_rename.json()["person"]["display_name"] == "Alicia"

    # Merge a new person into existing
    with http_api.SessionLocal() as session:
        session.add(http_api.PersonRow(id="bob", display_name="Bob"))
        session.commit()
    resp_merge = client.post("/persons/merge", json={"source_id": "bob", "target_id": person_id})
    assert resp_merge.status_code == 200
    assert resp_merge.json()["person"]["id"] == person_id
