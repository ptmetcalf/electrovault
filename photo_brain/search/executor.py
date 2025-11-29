from __future__ import annotations

from sqlalchemy.orm import Session

from photo_brain.core.models import (
    PhotoRecord,
    SearchQuery,
    SearchResult,
    VectorMatch,
)
from photo_brain.embedding import embed_description
from photo_brain.index.records import build_photo_record
from photo_brain.index.schema import PhotoFileRow
from photo_brain.index.vector_backend import PgVectorBackend


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


def _matches_events(record: PhotoRecord, event_filters: list[str]) -> bool:
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
        record = build_photo_record(
            session, photo_row, embedding_model=query_embedding.model
        )
        if not _within_date_window(record, query):
            continue
        if not _matches_people(record, query.people):
            continue
        if not _matches_events(record, query.event_ids):
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
