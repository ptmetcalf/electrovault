from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from photo_brain.core.models import PhotoFile
from photo_brain.faces import detect_faces, recognize_faces


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


def test_detect_and_recognize_faces(tmp_path: Path) -> None:
    photo = _sample_photo(tmp_path)
    detections = detect_faces(photo)
    assert detections
    assert detections[0].encoding is not None

    identities = recognize_faces(detections)
    assert identities
    assert identities[0].person_id.startswith("person-")
