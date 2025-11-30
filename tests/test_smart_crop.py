from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from photo_brain.core.models import Classification, FaceDetection, PhotoFile
from photo_brain.vision.smart_crop import generate_smart_crop


def _write_image(path: Path, size: tuple[int, int] = (240, 320)) -> Path:
    """Create a simple synthetic image for crop tests."""
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (int(w * 0.25), int(h * 0.25)), (int(w * 0.75), int(h * 0.75)), (255, 255, 255), -1)
    cv2.imwrite(str(path), img)
    return path


def _photo(path: Path) -> PhotoFile:
    return PhotoFile(
        id=path.stem,
        path=str(path),
        sha256="test",
        size_bytes=path.stat().st_size,
        mtime=datetime.now(timezone.utc),
    )


def test_document_like_prefers_contain_mode(tmp_path: Path) -> None:
    img_path = _write_image(tmp_path / "doc.jpg")
    photo = _photo(img_path)
    cls = Classification(photo_id=photo.id, label="bucket:documents", score=0.92, source="test")

    crop = generate_smart_crop(photo, [cls], [])

    assert crop is not None
    assert crop.subject_type == "document_like"
    assert crop.render_mode == "contain"
    assert math.isclose(crop.primary_crop.w, 1.0, rel_tol=1e-3)
    assert math.isclose(crop.primary_crop.h, 1.0, rel_tol=1e-3)
    assert crop.focal_point.x == 0.5
    assert crop.focal_point.y == 0.5


def test_faces_drive_primary_crop(tmp_path: Path) -> None:
    img_path = _write_image(tmp_path / "face.jpg", size=(300, 400))
    photo = _photo(img_path)
    detection = FaceDetection(
        photo_id=photo.id,
        bbox=(50.0, 60.0, 150.0, 200.0),
        confidence=0.84,
        encoding=None,
    )

    crop = generate_smart_crop(photo, [], [detection])

    assert crop is not None
    assert crop.subject_type == "photo_people"
    assert crop.render_mode == "cover"
    assert 0.0 <= crop.primary_crop.x <= 1.0
    assert 0.0 <= crop.primary_crop.y <= 1.0
    assert 0.0 < crop.primary_crop.w <= 1.0
    assert 0.0 < crop.primary_crop.h <= 1.0
    # Focal point should gravitate toward the detection center (~0.25, 0.43 in normalized coords).
    assert 0.15 < crop.focal_point.x < 0.55
    assert 0.2 < crop.focal_point.y < 0.7
