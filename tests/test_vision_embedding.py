from datetime import datetime, timezone

from photo_brain.core.models import ExifData, PhotoFile
from photo_brain.embedding import embed_description
from photo_brain.vision import classify_photo, describe_photo


def test_describe_photo_and_classify() -> None:
    photo = PhotoFile(
        id="1",
        path="/tmp/cat_selfie.jpg",
        sha256="x" * 64,
        size_bytes=1,
        mtime=datetime.now(timezone.utc),
    )
    exif = ExifData(datetime_original=datetime(2023, 1, 1, tzinfo=timezone.utc))
    vision = describe_photo(photo, exif)
    assert vision is None  # No model configured should return None

    classes = classify_photo(photo, exif)
    assert classes is None


def test_embed_description_is_deterministic() -> None:
    embedding_a = embed_description("Hello World", photo_id="p1", dim=8)
    embedding_b = embed_description("Hello World", photo_id="p2", dim=8)
    assert embedding_a.vector == embedding_b.vector
    assert embedding_a.dim == 8
    assert all(-1 <= v <= 1 for v in embedding_a.vector)
