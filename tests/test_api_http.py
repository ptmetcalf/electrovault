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
