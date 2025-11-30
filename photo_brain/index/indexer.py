from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from photo_brain.core.models import (
    ExifData,
    FaceDetection,
    FaceIdentity,
    PhotoFile,
    VisionDescription,
)
from photo_brain.embedding import embed_description
from photo_brain.faces import detect_faces
from photo_brain.index.records import list_face_identities
from photo_brain.vision import classify_photo, describe_photo

from .location import resolve_photo_location
from .schema import (
    ClassificationRow,
    ExifDataRow,
    FaceDetectionRow,
    FaceIdentityRow,
    FacePersonLinkRow,
    PersonRow,
    PhotoFileRow,
    VisionDescriptionRow,
)
from .updates import upsert_person
from .vector_backend import PgVectorBackend

logger = logging.getLogger(__name__)
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.75"))
FACE_ASSIGN_THRESHOLD = float(os.getenv("FACE_ASSIGN_THRESHOLD", "0.9"))
FACE_ASSIGN_MIN_SAMPLES = int(os.getenv("FACE_ASSIGN_MIN_SAMPLES", "2"))


def _build_photo_model(row: PhotoFileRow) -> PhotoFile:
    return PhotoFile(
        id=row.id,
        path=row.path,
        sha256=row.sha256,
        size_bytes=row.size_bytes,
        mtime=row.mtime,
    )


def _normalize_vec(vec: list[float] | None) -> Optional[np.ndarray]:
    if not vec:
        return None
    arr = np.array(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return None
    return arr / norm


def _load_person_centroids(session: Session) -> dict[str, tuple[np.ndarray, str]]:
    """Return normalized centroid vectors per person_id with display names."""
    stmt = (
        select(FaceDetectionRow, FacePersonLinkRow, PersonRow, FaceIdentityRow)
        .outerjoin(FacePersonLinkRow, FacePersonLinkRow.detection_id == FaceDetectionRow.id)
        .outerjoin(PersonRow, PersonRow.id == FacePersonLinkRow.person_id)
        .outerjoin(FaceIdentityRow, FaceIdentityRow.detection_id == FaceDetectionRow.id)
        .where(FaceDetectionRow.encoding.is_not(None))
    )
    per_person: dict[str, list[np.ndarray]] = {}
    labels: dict[str, str] = {}
    for det_row, link_row, person_row, identity_row in session.execute(stmt).all():
        norm = _normalize_vec(det_row.encoding)
        if norm is None:
            continue
        person_id = None
        display_name = None
        if link_row and person_row:
            person_id = link_row.person_id
            display_name = person_row.display_name
        elif identity_row and identity_row.person_label:
            person_id = identity_row.person_label
            display_name = identity_row.person_label
        if not person_id:
            continue
        labels.setdefault(person_id, display_name or person_id)
        per_person.setdefault(person_id, []).append(norm)

    centroids: dict[str, tuple[np.ndarray, str]] = {}
    for person_id, vecs in per_person.items():
        centroid = np.mean(vecs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            continue
        centroids[person_id] = (centroid / norm, labels.get(person_id, person_id))
    return centroids


def _match_detections_to_persons(
    detections: list[FaceDetection], centroids: dict[str, tuple[np.ndarray, str]]
) -> dict[int, FaceIdentity]:
    matches: dict[int, FaceIdentity] = {}
    if not centroids:
        return matches
    centroid_items = list(centroids.items())
    for idx, detection in enumerate(detections):
        det_vec = _normalize_vec(detection.encoding)
        if det_vec is None:
            continue
        best_score = 0.0
        best_person: Optional[str] = None
        best_label: Optional[str] = None
        for person_id, (centroid_vec, display_name) in centroid_items:
            score = float(det_vec.dot(centroid_vec))
            if score > best_score:
                best_score = score
                best_person = person_id
                best_label = display_name
        if best_person and best_score >= FACE_MATCH_THRESHOLD:
            matches[idx] = FaceIdentity(
                person_id=best_person,
                detection_id=None,
                label=best_label or best_person,
                confidence=best_score,
            )
    return matches


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


def _merge_prompt_context(user_context: str | None, faces: list[FaceIdentity]) -> str | None:
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
    resolve_photo_location(session, photo_row, exif_model)

    existing_vision = session.get(VisionDescriptionRow, photo_row.id)
    applied_context = (
        context
        if context is not None
        else (existing_vision.user_context if existing_vision else None)
    )
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
                session.commit()
                return

    existing_faces: list[FaceIdentity] = list_face_identities(session, photo_row.id)
    prompt_context = _merge_prompt_context(applied_context, existing_faces)

    logger.info("Index: describing photo %s", photo_row.id)
    vision: VisionDescription | None = describe_photo(
        photo_model, exif_model, context=prompt_context
    )
    existing_vision = session.get(VisionDescriptionRow, photo_row.id)
    if vision:
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
    else:
        logger.info("Index: skipping vision upsert for %s (no model output)", photo_row.id)

    logger.info("Index: classifying photo %s", photo_row.id)
    classifications = classify_photo(photo_model, exif_model, context=prompt_context)
    if classifications is not None:
        session.execute(delete(ClassificationRow).where(ClassificationRow.photo_id == photo_row.id))
        for classification in classifications:
            session.add(
                ClassificationRow(
                    photo_id=photo_row.id,
                    label=classification.label,
                    score=classification.score,
                    source=classification.source,
                )
            )
    else:
        logger.info("Index: skipping classification upsert for %s (no model output)", photo_row.id)

    detections: list[FaceDetection] = []
    detection_rows = session.scalars(
        select(FaceDetectionRow).where(FaceDetectionRow.photo_id == photo_row.id)
    ).all()
    stale_detections = any(
        (row.encoding is None) or (len(row.encoding) < 128) for row in detection_rows
    )
    if detection_rows and preserve_faces and not stale_detections:
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
        if detection_rows:
            logger.info(
                "Index: refreshing %d face detections for photo %s (missing/low-dim encodings)",
                len(detection_rows),
                photo_row.id,
            )
        session.execute(delete(FaceDetectionRow).where(FaceDetectionRow.photo_id == photo_row.id))
        logger.info("Index: detecting faces for photo %s", photo_row.id)
        detections = detect_faces(photo_model)
        for detection in detections:
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
        session.flush()

    if detections:
        # Auto-assign only to existing named persons with enough samples and high confidence.
        centroids = _load_person_centroids(session)
        eligible = {
            pid: (vec, label)
            for pid, (vec, label) in centroids.items()
            if session.scalar(
                select(func.count()).select_from(FacePersonLinkRow).where(FacePersonLinkRow.person_id == pid)
            )
            >= FACE_ASSIGN_MIN_SAMPLES
        }
        if eligible:
            matches = _match_detections_to_persons(detections, eligible)
            for idx, detection in enumerate(detections):
                match = matches.get(idx)
                if not match or match.confidence is None or match.confidence < FACE_ASSIGN_THRESHOLD:
                    continue
                # Persist assignment to existing person only.
                person = session.get(PersonRow, match.person_id)
                if not person:
                    person = upsert_person(session, display_name=match.label or match.person_id or "person")
                det_id = detection_rows[idx].id if idx < len(detections) and detection.id else detection.id
                if det_id is None and idx < len(detections):
                    # Fetch freshly inserted detection row.
                    det_row = session.scalar(
                        select(FaceDetectionRow.id)
                        .where(FaceDetectionRow.photo_id == photo_row.id)
                        .order_by(FaceDetectionRow.created_at.desc())
                        .offset(idx)
                    )
                    det_id = det_row
                if det_id is not None:
                    session.add(
                        FaceIdentityRow(
                            detection_id=det_id,
                            person_label=person.display_name,
                            confidence=match.confidence,
                            auto_assigned=True,
                        )
                    )
                    session.add(
                        FacePersonLinkRow(
                            detection_id=det_id,
                            person_id=person.id,
                        )
                    )

    embedding = None
    if vision:
        logger.info("Index: embedding description for photo %s", photo_row.id)
        embedding = embed_description(vision.description, photo_id=photo_row.id)
        backend.upsert_embedding(session, embedding)
    else:
        logger.info("Index: skipping embedding for photo %s (no vision description)", photo_row.id)
    session.commit()
    vision_model = vision.model if vision else "none"
    embedding_model = embedding.model if embedding else "none"
    logger.info(
        "Index: completed photo %s (vision model=%s, %d classes, %d faces, embed model=%s)",
        photo_row.id,
        vision_model,
        len(classifications or []),
        len(detections),
        embedding_model,
    )
