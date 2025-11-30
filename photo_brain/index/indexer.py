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
    SmartCrop,
)
from photo_brain.embedding import embed_description
from photo_brain.faces import detect_faces
from photo_brain.index.records import list_face_identities
from photo_brain.vision import classify_photo, describe_photo
from photo_brain.vision.smart_crop import generate_smart_crop

from .location import resolve_photo_location
from .schema import (
    ClassificationRow,
    ExifDataRow,
    FaceDetectionRow,
    FaceIdentityRow,
    FacePersonLinkRow,
    MemoryEventRow,
    PersonRow,
    PhotoFileRow,
    PhotoLocationRow,
    SmartCropRow,
    VisionDescriptionRow,
    event_photos,
)
from .updates import upsert_person, update_person_stats
from .vector_backend import PgVectorBackend

logger = logging.getLogger(__name__)
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.75"))
FACE_ASSIGN_THRESHOLD = float(os.getenv("FACE_ASSIGN_THRESHOLD", "0.93"))
FACE_ASSIGN_MIN_SAMPLES = int(os.getenv("FACE_ASSIGN_MIN_SAMPLES", "2"))
FACE_ASSIGN_CONFLICT_GAP = float(os.getenv("FACE_ASSIGN_CONFLICT_GAP", "0.04"))


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
    """Return normalized centroid vectors per confirmed person_id with display names."""
    from .schema import PersonStatsRow

    centroids: dict[str, tuple[np.ndarray, str]] = {}
    stats_rows = session.execute(
        select(PersonStatsRow, PersonRow)
        .join(PersonRow, PersonRow.id == PersonStatsRow.person_id)
        .where(
            PersonRow.is_user_confirmed.is_(True),
            PersonRow.auto_assign_enabled.is_(True),
            PersonStatsRow.embedding_centroid.is_not(None),
            PersonStatsRow.embedding_count >= FACE_ASSIGN_MIN_SAMPLES,
        )
    ).all()
    for stats_row, person_row in stats_rows:
        norm = _normalize_vec(stats_row.embedding_centroid)
        if norm is None:
            continue
        centroids[person_row.id] = (norm, person_row.display_name or person_row.id)

    if not centroids:
        # Fallback: derive centroids from linked detections when stats are missing.
        per_person: dict[str, list[np.ndarray]] = {}
        labels: dict[str, str] = {}
        rows = session.execute(
            select(PersonRow, FaceDetectionRow, FacePersonLinkRow)
            .join(FacePersonLinkRow, FacePersonLinkRow.person_id == PersonRow.id)
            .join(FaceDetectionRow, FaceDetectionRow.id == FacePersonLinkRow.detection_id)
            .where(
                PersonRow.is_user_confirmed.is_(True),
                PersonRow.auto_assign_enabled.is_(True),
                FaceDetectionRow.encoding.is_not(None),
            )
        ).all()
        for person_row, det_row, _ in rows:
            norm_vec = _normalize_vec(det_row.encoding)
            if norm_vec is None:
                continue
            per_person.setdefault(person_row.id, []).append(norm_vec)
            labels[person_row.id] = person_row.display_name or person_row.id
        for pid, vecs in per_person.items():
            if len(vecs) < FACE_ASSIGN_MIN_SAMPLES:
                continue
            centroid = np.mean(vecs, axis=0)
            norm = np.linalg.norm(centroid)
            if norm:
                centroids[pid] = (centroid / norm, labels.get(pid, pid))
    return centroids


def _match_detections_to_persons(
    detections: list[FaceDetection], centroids: dict[str, tuple[np.ndarray, str]]
) -> dict[int, tuple[FaceIdentity, float]]:
    """
    Return best match + second-best score per detection for conflict checks.
    """
    matches: dict[int, tuple[FaceIdentity, float]] = {}
    if not centroids:
        return matches
    centroid_items = list(centroids.items())
    for idx, detection in enumerate(detections):
        det_vec = _normalize_vec(detection.encoding)
        if det_vec is None:
            continue
        best_score = 0.0
        second_score = 0.0
        best_person: Optional[str] = None
        best_label: Optional[str] = None
        for person_id, (centroid_vec, display_name) in centroid_items:
            score = float(det_vec.dot(centroid_vec))
            if score > best_score:
                second_score = best_score
                best_score = score
                best_person = person_id
                best_label = display_name
            elif score > second_score:
                second_score = score
        if best_person and best_score >= FACE_MATCH_THRESHOLD:
            matches[idx] = (
                FaceIdentity(
                    person_id=best_person,
                    detection_id=None,
                    label=best_label or best_person,
                    confidence=best_score,
                ),
                second_score,
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


def _build_caption_context(
    session: Session,
    photo_row: PhotoFileRow,
    exif: ExifData | None,
    faces: list[FaceIdentity],
    user_context: str | None,
) -> str | None:
    parts: list[str] = []
    if user_context:
        parts.append(user_context)
    names = sorted({face.label or face.person_id for face in faces if (face.person_id or face.label)})
    if names:
        parts.append("People visible: " + ", ".join(names))
    location_row = session.get(PhotoLocationRow, photo_row.id)
    if location_row and location_row.location:
        parts.append(f"Location: {location_row.location.name}")
    event_titles = session.scalars(
        select(MemoryEventRow.title)
        .join(event_photos, event_photos.c.event_id == MemoryEventRow.id)
        .where(event_photos.c.photo_id == photo_row.id)
    ).all()
    if event_titles:
        parts.append("Events: " + ", ".join(event_titles))
    if exif and exif.datetime_original:
        parts.append(f"Captured on {exif.datetime_original.date().isoformat()}")
    context = "\n".join(parts).strip()
    return context or None


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
    has_smart_crop = photo_row.smart_crop is not None
    applied_context = (
        context
        if context is not None
        else (existing_vision.user_context if existing_vision else None)
    )
    if (
        skip_if_fresh
        and existing_vision
        and existing_vision.user_context == applied_context
        and has_smart_crop
    ):
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
            detection.id = det_row.id
        detection_rows = session.scalars(
            select(FaceDetectionRow).where(FaceDetectionRow.photo_id == photo_row.id)
        ).all()
        session.flush()

    capture_dt = (
        exif_model.datetime_original if exif_model and exif_model.datetime_original else photo_row.mtime
    )
    smart_crop: SmartCrop | None = generate_smart_crop(
        photo_model,
        classifications or [],
        detections,
        captured_at=capture_dt,
    )
    if smart_crop:
        existing_crop = session.get(SmartCropRow, photo_row.id)
        if existing_crop:
            existing_crop.subject_type = smart_crop.subject_type
            existing_crop.render_mode = smart_crop.render_mode
            existing_crop.crop_x = smart_crop.primary_crop.x
            existing_crop.crop_y = smart_crop.primary_crop.y
            existing_crop.crop_w = smart_crop.primary_crop.w
            existing_crop.crop_h = smart_crop.primary_crop.h
            existing_crop.focal_x = smart_crop.focal_point.x
            existing_crop.focal_y = smart_crop.focal_point.y
            existing_crop.type_label = smart_crop.type_label
            existing_crop.summary = smart_crop.summary
        else:
            session.add(
                SmartCropRow(
                    photo_id=photo_row.id,
                    subject_type=smart_crop.subject_type,
                    render_mode=smart_crop.render_mode,
                    crop_x=smart_crop.primary_crop.x,
                    crop_y=smart_crop.primary_crop.y,
                    crop_w=smart_crop.primary_crop.w,
                    crop_h=smart_crop.primary_crop.h,
                    focal_x=smart_crop.focal_point.x,
                    focal_y=smart_crop.focal_point.y,
                    type_label=smart_crop.type_label,
                    summary=smart_crop.summary,
                )
            )

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
                match_tuple = matches.get(idx)
                if not match_tuple:
                    continue
                match, second_best = match_tuple
                if match.confidence is None:
                    continue
                if match.confidence < FACE_ASSIGN_THRESHOLD:
                    continue
                if (match.confidence - second_best) < FACE_ASSIGN_CONFLICT_GAP:
                    continue
                # Persist assignment to existing person only.
                person = session.get(PersonRow, match.person_id)
                if not person:
                    continue
                det_id = None
                if detection.id is not None:
                    det_id = detection.id
                elif idx < len(detection_rows):
                    det_id = detection_rows[idx].id
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
                    update_person_stats(session, person.id)

    session.commit()
    logger.info(
        "Index: completed photo %s (vision model=%s, %d classes, %d faces)",
        photo_row.id,
        "deferred",
        len(classifications or []),
        len(detections),
    )


def caption_photo(
    session: Session,
    photo_row: PhotoFileRow,
    *,
    backend: Optional[PgVectorBackend] = None,
    context: str | None = None,
) -> VisionDescription | None:
    """Generate or refresh caption + embedding using rich context (faces, location, events, user notes)."""
    backend = backend or PgVectorBackend()
    exif_model = _load_exif_model(photo_row.exif)
    photo_model = _build_photo_model(photo_row)
    faces = list_face_identities(session, photo_row.id)
    existing_vision = session.get(VisionDescriptionRow, photo_row.id)
    user_context = context if context is not None else (existing_vision.user_context if existing_vision else None)
    caption_context = _build_caption_context(session, photo_row, exif_model, faces, user_context)

    logger.info("Caption: describing photo %s", photo_row.id)
    vision: VisionDescription | None = describe_photo(photo_model, exif_model, context=caption_context)
    if not vision:
        logger.info("Caption: no vision output for %s", photo_row.id)
        return None

    if existing_vision:
        existing_vision.description = vision.description
        existing_vision.model = vision.model
        existing_vision.user_context = user_context
    else:
        session.add(
            VisionDescriptionRow(
                photo_id=photo_row.id,
                description=vision.description,
                model=vision.model,
                user_context=user_context,
            )
        )

    logger.info("Caption: embedding description for photo %s", photo_row.id)
    embedding = embed_description(vision.description, photo_id=photo_row.id)
    backend.upsert_embedding(session, embedding)
    session.commit()
    return VisionDescription(
        photo_id=photo_row.id,
        description=vision.description,
        model=vision.model,
        user_context=user_context,
        created_at=vision.created_at,
    )
