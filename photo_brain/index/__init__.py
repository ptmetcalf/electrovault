"""Database and vector index layer for Photo Brain."""

from .indexer import index_photo
from .records import build_photo_record, list_face_identities, load_photo_record
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
from .updates import assign_face_identity, set_photo_user_context
from .vector_backend import PgVectorBackend

__all__ = [
    "Base",
    "ClassificationRow",
    "ExifDataRow",
    "assign_face_identity",
    "FaceDetectionRow",
    "FaceIdentityRow",
    "MemoryEventRow",
    "PhotoFileRow",
    "TextEmbeddingRow",
    "VisionDescriptionRow",
    "build_photo_record",
    "index_photo",
    "list_face_identities",
    "load_photo_record",
    "set_photo_user_context",
    "PgVectorBackend",
    "create_engine_from_url",
    "event_photos",
    "init_db",
    "session_factory",
]
