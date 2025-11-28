from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from photo_brain.core.models import Classification, ExifData, PhotoFile, VisionDescription
from photo_brain.embedding import embed_description
from photo_brain.faces import detect_faces, recognize_faces
from photo_brain.vision import classify_photo, describe_photo

from .schema import (
    ClassificationRow,
    FaceDetectionRow,
    FaceIdentityRow,
    PhotoFileRow,
    VisionDescriptionRow,
)
from .vector_backend import PgVectorBackend

logger = logging.getLogger(__name__)


def _build_photo_model(row: PhotoFileRow) -> PhotoFile:
    return PhotoFile(
        id=row.id,
        path=row.path,
        sha256=row.sha256,
        size_bytes=row.size_bytes,
        mtime=row.mtime,
    )


def _load_exif_model(row: Optional[ExifDataRow]) -> Optional[ExifData]:
    if row is None:
        return None
    return ExifData(
        datetime_original=row.datetime_original,
        gps_lat=row.gps_lat,
        gps_lon=row.gps_lon,
        gps_altitude=row.gps_altitude,
        gps_altitude_ref=row.gps_altitude_ref,
        gps_timestamp=row.gps_timestamp,
        camera_make=row.camera_make,
        camera_model=row.camera_model,
        lens_model=row.lens_model,
        software=row.software,
        orientation=row.orientation,
        exposure_time=row.exposure_time,
        f_number=row.f_number,
        iso=row.iso,
        focal_length=row.focal_length,
    )


def index_photo(
    session: Session,
    photo_row: PhotoFileRow,
    *,
    backend: Optional[PgVectorBackend] = None,
    context: str | None = None,
    skip_if_fresh: bool = True,
) -> None:
    """Generate vision, classifications, and embeddings for a photo."""
    backend = backend or PgVectorBackend()
    exif_model = _load_exif_model(photo_row.exif)
    photo_model = _build_photo_model(photo_row)

    existing_vision = session.get(VisionDescriptionRow, photo_row.id)
    if skip_if_fresh and existing_vision and existing_vision.user_context == context:
        class_count = session.scalar(
            select(func.count())
            .select_from(ClassificationRow)
            .where(ClassificationRow.photo_id == photo_row.id)
        )
        if class_count and existing_vision.created_at:
            mtime = photo_row.mtime
            created = existing_vision.created_at
            # Normalize timezone awareness before comparing
            if mtime.tzinfo and created.tzinfo is None:
                created = created.replace(tzinfo=mtime.tzinfo)
            elif created.tzinfo and mtime.tzinfo is None:
                mtime = mtime.replace(tzinfo=created.tzinfo)
            if mtime <= created:
                logger.info("Index: skipping photo %s (unchanged, context matched)", photo_row.id)
                return

    logger.info("Index: describing photo %s", photo_row.id)
    vision: VisionDescription = describe_photo(photo_model, exif_model, context=context)
    existing_vision = session.get(VisionDescriptionRow, photo_row.id)
    if existing_vision:
        existing_vision.description = vision.description
        existing_vision.model = vision.model
        existing_vision.confidence = vision.confidence
        existing_vision.user_context = context
    else:
        session.add(
            VisionDescriptionRow(
                photo_id=photo_row.id,
                description=vision.description,
                model=vision.model,
                confidence=vision.confidence,
                user_context=context,
            )
        )

    logger.info("Index: classifying photo %s", photo_row.id)
    classifications = classify_photo(photo_model, exif_model, context=context)
    session.execute(
        delete(ClassificationRow).where(ClassificationRow.photo_id == photo_row.id)
    )
    for classification in classifications:
        session.add(
            ClassificationRow(
                photo_id=photo_row.id,
                label=classification.label,
                score=classification.score,
                source=classification.source,
            )
        )

    session.execute(
        delete(FaceDetectionRow).where(FaceDetectionRow.photo_id == photo_row.id)
    )
    logger.info("Index: detecting faces for photo %s", photo_row.id)
    detections = detect_faces(photo_model)
    identities = recognize_faces(detections)
    for idx, detection in enumerate(detections):
        det_row = FaceDetectionRow(
            photo_id=photo_row.id,
            bbox_x1=detection.bbox[0],
            bbox_y1=detection.bbox[1],
            bbox_x2=detection.bbox[2],
            bbox_y2=detection.bbox[3],
            confidence=detection.confidence,
            encoding=detection.encoding,
        )
        session.add(det_row)
        session.flush()
        identity = identities[idx] if idx < len(identities) else None
        if identity:
            session.add(
                FaceIdentityRow(
                    detection_id=det_row.id,
                    person_label=identity.person_id or identity.label or "unknown",
                    confidence=identity.confidence,
                )
            )

    logger.info("Index: embedding description for photo %s", photo_row.id)
    embedding = embed_description(vision.description, photo_id=photo_row.id)
    backend.upsert_embedding(session, embedding)
    session.commit()
    logger.info(
        "Index: completed photo %s (vision model=%s, %d classes, %d faces, embed model=%s)",
        photo_row.id,
        vision.model,
        len(classifications),
        len(detections),
        embedding.model,
    )
