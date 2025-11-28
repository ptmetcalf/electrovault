from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from photo_brain.core.models import (
    Classification,
    ExifData,
    FaceIdentity,
    PhotoFile,
    PhotoRecord,
    SearchQuery,
    SearchResult,
    TextEmbedding,
    VectorMatch,
    VisionDescription,
)
from photo_brain.embedding import embed_description
from photo_brain.index.schema import (
    ClassificationRow,
    ExifDataRow,
    FaceDetectionRow,
    FaceIdentityRow,
    PhotoFileRow,
    TextEmbeddingRow,
    VisionDescriptionRow,
)
from photo_brain.index.schema import event_photos
from photo_brain.index.vector_backend import PgVectorBackend


def _to_photo_record(
    session: Session, photo_row: PhotoFileRow, *, embedding_model: str | None
) -> PhotoRecord:
    exif = None
    if photo_row.exif:
        exif = ExifData(
            datetime_original=photo_row.exif.datetime_original,
            gps_lat=photo_row.exif.gps_lat,
            gps_lon=photo_row.exif.gps_lon,
        )

    vision_row = session.scalar(
        select(VisionDescriptionRow).where(VisionDescriptionRow.photo_id == photo_row.id)
    )
    vision = None
    if vision_row:
        vision = VisionDescription(
            photo_id=photo_row.id,
            description=vision_row.description,
            model=vision_row.model,
            confidence=vision_row.confidence,
            created_at=vision_row.created_at,
        )

    classifications = [
        Classification(
            photo_id=cls.photo_id,
            label=cls.label,
            score=cls.score,
            source=cls.source,
            created_at=cls.created_at,
        )
        for cls in session.scalars(
            select(ClassificationRow).where(ClassificationRow.photo_id == photo_row.id)
        ).all()
    ]

    embedding_row = session.scalar(
        select(TextEmbeddingRow)
        .where(TextEmbeddingRow.photo_id == photo_row.id)
        .order_by(TextEmbeddingRow.created_at.desc())
    )
    if embedding_model:
        specific_row = session.scalar(
            select(TextEmbeddingRow)
            .where(
                TextEmbeddingRow.photo_id == photo_row.id,
                TextEmbeddingRow.model == embedding_model,
            )
            .order_by(TextEmbeddingRow.created_at.desc())
        )
        if specific_row:
            embedding_row = specific_row

    embedding: TextEmbedding | None = None
    if embedding_row:
        embedding = TextEmbedding(
            photo_id=embedding_row.photo_id,
            model=embedding_row.model,
            vector=embedding_row.embedding,
            dim=embedding_row.dim,
            created_at=embedding_row.created_at,
        )

    face_rows = session.scalars(
        select(FaceIdentityRow)
        .join(FaceDetectionRow, FaceDetectionRow.id == FaceIdentityRow.detection_id)
        .where(FaceDetectionRow.photo_id == photo_row.id)
    ).all()
    faces = [
        FaceIdentity(
            person_id=face.person_label,
            detection_id=face.detection_id,
            label=face.person_label,
            confidence=face.confidence,
            created_at=face.created_at,
        )
        for face in face_rows
    ]

    event_ids = session.scalars(
        select(event_photos.c.event_id).where(event_photos.c.photo_id == photo_row.id)
    ).all()

    return PhotoRecord(
        file=PhotoFile(
            id=photo_row.id,
            path=photo_row.path,
            sha256=photo_row.sha256,
            size_bytes=photo_row.size_bytes,
            mtime=photo_row.mtime,
        ),
        exif=exif,
        vision=vision,
        classifications=classifications,
        embedding=embedding,
        faces=faces,
        event_ids=event_ids,
    )


def _within_date_window(record: PhotoRecord, query: SearchQuery) -> bool:
    timestamp = record.exif.datetime_original if record.exif else record.file.mtime
    if query.start_date and timestamp < query.start_date:
        return False
    if query.end_date and timestamp > query.end_date:
        return False
    return True


def _matches_people(record: PhotoRecord, people: list[str]) -> bool:
    if not people:
        return True
    lower_people = [person.lower() for person in people]
    for face in record.faces:
        identity = (face.person_id or face.label or "").lower()
        if any(person in identity for person in lower_people):
            return True

    haystack = f"{record.file.path} " + " ".join(cls.label for cls in record.classifications)
    haystack_lower = haystack.lower()
    return any(person in haystack_lower for person in lower_people)


def _matches_events(
    session: Session, record: PhotoRecord, event_filters: list[str]
) -> bool:
    if not event_filters:
        return True
    if not record.event_ids:
        return False
    return any(event_id in record.event_ids for event_id in event_filters)


def execute_search(
    session: Session, backend: PgVectorBackend, query: SearchQuery
) -> list[SearchResult]:
    query_embedding = embed_description(query.text, photo_id="query")
    matches: list[VectorMatch] = backend.search(
        session, query_embedding.vector, limit=query.limit, model=query_embedding.model
    )
    results: list[SearchResult] = []
    for match in matches:
        photo_row = session.get(PhotoFileRow, match.photo_id)
        if photo_row is None:
            continue
        record = _to_photo_record(
            session, photo_row, embedding_model=query_embedding.model
        )
        if not _within_date_window(record, query):
            continue
        if not _matches_people(record, query.people):
            continue
        if not _matches_events(session, record, query.event_ids):
            continue
        matched_filters: list[str] = []
        if query.start_date or query.end_date:
            matched_filters.append("date")
        if query.people:
            matched_filters.append("people")
        if query.event_ids:
            matched_filters.append("event")
        results.append(SearchResult(record=record, score=match.score, matched_filters=matched_filters))
    return results
