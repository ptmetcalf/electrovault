"""Database and vector index layer for Photo Brain."""

from .schema import (
    Base,
    ClassificationRow,
    ExifDataRow,
    FaceDetectionRow,
    FaceIdentityRow,
    MemoryEventRow,
    PhotoFileRow,
    TextEmbeddingRow,
    VisionDescriptionRow,
    create_engine_from_url,
    event_photos,
    init_db,
    session_factory,
)
from .indexer import index_photo
from .vector_backend import PgVectorBackend

__all__ = [
    "Base",
    "ClassificationRow",
    "ExifDataRow",
    "FaceDetectionRow",
    "FaceIdentityRow",
    "MemoryEventRow",
    "PhotoFileRow",
    "TextEmbeddingRow",
    "VisionDescriptionRow",
    "index_photo",
    "PgVectorBackend",
    "create_engine_from_url",
    "event_photos",
    "init_db",
    "session_factory",
]
