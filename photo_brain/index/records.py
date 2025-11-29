from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from photo_brain.core.models import (
    Classification,
    ExifData,
    FaceDetection,
    FaceIdentity,
    LocationLabel,
    PhotoLocation,
    PhotoFile,
    PhotoRecord,
    TextEmbedding,
    VisionDescription,
)

from .schema import (
    ClassificationRow,
    ExifDataRow,
    FaceDetectionRow,
    FaceIdentityRow,
    LocationLabelRow,
    PhotoFileRow,
    PhotoLocationRow,
    TextEmbeddingRow,
    VisionDescriptionRow,
    event_photos,
)


def _load_exif(row: ExifDataRow | None) -> ExifData | None:
    if row is None:
        return None
    return ExifData(
        datetime_original=row.datetime_original,
        gps_lat=row.gps_lat,
        gps_lon=row.gps_lon,
    )


def _load_location(row: PhotoLocationRow | None) -> PhotoLocation | None:
    if row is None or row.location is None:
        return None
    label_row: LocationLabelRow = row.location
    label = LocationLabel(
        id=label_row.id,
        name=label_row.name,
        latitude=label_row.latitude,
        longitude=label_row.longitude,
        radius_meters=label_row.radius_meters,
        source=label_row.source,
        raw=label_row.raw,
        created_at=label_row.created_at,
    )
    return PhotoLocation(
        photo_id=row.photo_id,
        location=label,
        method=row.method,
        confidence=row.confidence,
        created_at=row.created_at,
    )


def list_face_identities(session: Session, photo_id: str) -> list[FaceIdentity]:
    """Return face identities for a photo, ordered by detection id."""
    face_rows = session.scalars(
        select(FaceIdentityRow)
        .join(FaceDetectionRow, FaceDetectionRow.id == FaceIdentityRow.detection_id)
        .where(FaceDetectionRow.photo_id == photo_id)
        .order_by(FaceIdentityRow.detection_id)
    ).all()
    return [
        FaceIdentity(
            person_id=face.person_label,
            detection_id=face.detection_id,
            label=face.person_label,
            confidence=face.confidence,
            created_at=face.created_at,
        )
        for face in face_rows
    ]


def _load_detections(session: Session, photo_id: str) -> list[FaceDetection]:
    rows = session.scalars(
        select(FaceDetectionRow).where(FaceDetectionRow.photo_id == photo_id)
    ).all()
    return [
        FaceDetection(
            id=row.id,
            photo_id=row.photo_id,
            bbox=(row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2),
            confidence=row.confidence,
            encoding=row.encoding,
            created_at=row.created_at,
        )
        for row in rows
    ]


def build_photo_record(
    session: Session, photo_row: PhotoFileRow, *, embedding_model: str | None = None
) -> PhotoRecord:
    exif = _load_exif(photo_row.exif)
    location_row = session.get(PhotoLocationRow, photo_row.id)
    location = _load_location(location_row)

    vision_row = session.scalar(
        select(VisionDescriptionRow).where(VisionDescriptionRow.photo_id == photo_row.id)
    )
    vision: VisionDescription | None = None
    if vision_row:
        vision = VisionDescription(
            photo_id=photo_row.id,
            description=vision_row.description,
            model=vision_row.model,
            confidence=vision_row.confidence,
            user_context=vision_row.user_context,
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

    embedding_row: Optional[TextEmbeddingRow] = session.scalar(
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

    faces = list_face_identities(session, photo_row.id)
    detections = _load_detections(session, photo_row.id)

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
        detections=detections,
        faces=faces,
        location=location,
        event_ids=event_ids,
    )


def load_photo_record(
    session: Session, photo_id: str, *, embedding_model: str | None = None
) -> PhotoRecord | None:
    row = session.get(PhotoFileRow, photo_id)
    if row is None:
        return None
    return build_photo_record(session, row, embedding_model=embedding_model)
