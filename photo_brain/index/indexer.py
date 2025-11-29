from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from photo_brain.core.models import (
    Classification,
    ExifData,
    FaceDetection,
    FaceIdentity,
    PhotoFile,
    VisionDescription,
)
from photo_brain.embedding import embed_description
from photo_brain.faces import detect_faces, recognize_faces
from photo_brain.index.records import list_face_identities
from photo_brain.vision import classify_photo, describe_photo

from .schema import (
    ClassificationRow,
    ExifDataRow,
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


def _merge_prompt_context(
    user_context: str | None, faces: list[FaceIdentity]
) -> str | None:
    """Combine user context with known face assignments for richer prompts."""
    context_parts: list[str] = []
    names = sorted(
        {face.person_id or face.label for face in faces if (face.person_id or face.label)}
    )
    if names:
        context_parts.append(
            "Known people visible in this photo: "
            + ", ".join(names)
            + ". Use their names when relevant."
        )
    if user_context:
        context_parts.append(user_context)
    combined = "\n".join(part for part in context_parts if part).strip()
    return combined or None


def index_photo(
    session: Session,
    photo_row: PhotoFileRow,
    *,
    backend: Optional[PgVectorBackend] = None,
    context: str | None = None,
    skip_if_fresh: bool = True,
    preserve_faces: bool = True,
) -> None:
    """Generate vision, classifications, and embeddings for a photo."""
    backend = backend or PgVectorBackend()
    exif_model = _load_exif_model(photo_row.exif)
    photo_model = _build_photo_model(photo_row)

    existing_vision = session.get(VisionDescriptionRow, photo_row.id)
    applied_context = context if context is not None else (existing_vision.user_context if existing_vision else None)
    if skip_if_fresh and existing_vision and existing_vision.user_context == applied_context:
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

    existing_faces: list[FaceIdentity] = list_face_identities(session, photo_row.id)
    prompt_context = _merge_prompt_context(applied_context, existing_faces)

    logger.info("Index: describing photo %s", photo_row.id)
    vision: VisionDescription = describe_photo(
        photo_model, exif_model, context=prompt_context
    )
    existing_vision = session.get(VisionDescriptionRow, photo_row.id)
    if existing_vision:
        existing_vision.description = vision.description
        existing_vision.model = vision.model
        existing_vision.confidence = vision.confidence
        existing_vision.user_context = applied_context
    else:
        session.add(
            VisionDescriptionRow(
                photo_id=photo_row.id,
                description=vision.description,
                model=vision.model,
                confidence=vision.confidence,
                user_context=applied_context,
            )
        )

    logger.info("Index: classifying photo %s", photo_row.id)
    classifications = classify_photo(
        photo_model, exif_model, context=prompt_context
    )
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

    detections: list[FaceDetection] = []
    detection_rows = session.scalars(
        select(FaceDetectionRow).where(FaceDetectionRow.photo_id == photo_row.id)
    ).all()
    if detection_rows and preserve_faces:
        logger.info(
            "Index: reusing %d existing face detections for photo %s",
            len(detection_rows),
            photo_row.id,
        )
        for det_row in detection_rows:
            detections.append(
                FaceDetection(
                    id=det_row.id,
                    photo_id=det_row.photo_id,
                    bbox=(det_row.bbox_x1, det_row.bbox_y1, det_row.bbox_x2, det_row.bbox_y2),
                    confidence=det_row.confidence,
                    encoding=det_row.encoding,
                    created_at=det_row.created_at,
                )
            )
    else:
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
        session.flush()

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
