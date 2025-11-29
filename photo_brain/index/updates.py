from __future__ import annotations

import re
import uuid

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from photo_brain.index.schema import (
    FaceDetectionRow,
    FacePersonLinkRow,
    FaceIdentityRow,
    PersonRow,
    PhotoFileRow,
    VisionDescriptionRow,
)


def _slugify_person_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or uuid.uuid4().hex


def upsert_person(session: Session, *, display_name: str, person_id: str | None = None) -> PersonRow:
    """Create or return a person row for the given display name/id."""
    normalized_name = display_name.strip()
    if not normalized_name:
        raise ValueError("Person name is required")

    if person_id:
        existing = session.get(PersonRow, person_id)
        if existing:
            if existing.display_name != normalized_name:
                existing.display_name = normalized_name
            session.flush()
            return existing
    else:
        # Try to reuse an existing person with the same display name.
        existing = session.scalar(
            select(PersonRow).where(PersonRow.display_name.ilike(normalized_name))
        )
        if existing:
            return existing
        person_id = _slugify_person_id(normalized_name)
        suffix = 1
        candidate = person_id
        while session.get(PersonRow, candidate):
            candidate = f"{person_id}-{suffix}"
            suffix += 1
        person_id = candidate

    person = PersonRow(id=person_id, display_name=normalized_name)
    session.add(person)
    session.flush()
    return person


def set_detection_person(
    session: Session, detection_id: int, *, person: PersonRow, confidence: float | None = None
) -> FaceIdentityRow:
    """Attach a detection to a person (identity row + link)."""
    detection = session.get(FaceDetectionRow, detection_id)
    if detection is None:
        raise ValueError("Face detection not found")

    identity = session.scalar(
        select(FaceIdentityRow).where(FaceIdentityRow.detection_id == detection_id)
    )
    if identity:
        identity.person_label = person.display_name
        if confidence is not None:
            identity.confidence = confidence
    else:
        identity = FaceIdentityRow(
            detection_id=detection_id,
            person_label=person.display_name,
            confidence=confidence,
        )
        session.add(identity)

    link = session.scalar(
        select(FacePersonLinkRow).where(FacePersonLinkRow.detection_id == detection_id)
    )
    if link:
        link.person_id = person.id
    else:
        session.add(FacePersonLinkRow(detection_id=detection_id, person_id=person.id))

    session.flush()
    return identity


def assign_face_identity(session: Session, detection_id: int, person_label: str) -> FaceIdentityRow:
    """Set or update the person label for a face detection."""
    person = upsert_person(session, display_name=person_label)
    return set_detection_person(session, detection_id, person=person)


def rename_person(session: Session, person_id: str, new_name: str) -> PersonRow:
    person = session.get(PersonRow, person_id)
    if not person:
        raise ValueError("Person not found")
    person.display_name = new_name.strip()
    session.flush()
    # Keep identity rows in sync for linked detections.
    detection_ids = session.scalars(
        select(FacePersonLinkRow.detection_id).where(FacePersonLinkRow.person_id == person_id)
    ).all()
    if detection_ids:
        session.execute(
            update(FaceIdentityRow)
            .where(FaceIdentityRow.detection_id.in_(detection_ids))
            .values(person_label=person.display_name)
        )
    session.flush()
    return person


def merge_persons(session: Session, source_id: str, target_id: str) -> PersonRow:
    if source_id == target_id:
        raise ValueError("Cannot merge the same person")
    target = session.get(PersonRow, target_id)
    if not target:
        raise ValueError("Target person not found")
    source = session.get(PersonRow, source_id)
    if not source:
        raise ValueError("Source person not found")

    # Repoint links.
    session.execute(
        update(FacePersonLinkRow)
        .where(FacePersonLinkRow.person_id == source_id)
        .values(person_id=target_id)
    )
    # Update identity labels to target's display name for affected detections.
    detection_ids = session.scalars(
        select(FacePersonLinkRow.detection_id).where(FacePersonLinkRow.person_id == target_id)
    ).all()
    if detection_ids:
        session.execute(
            update(FaceIdentityRow)
            .where(FaceIdentityRow.detection_id.in_(detection_ids))
            .values(person_label=target.display_name)
        )
    session.flush()
    session.delete(source)
    session.flush()
    return target


def set_photo_user_context(
    session: Session, photo_row: PhotoFileRow, context: str
) -> VisionDescriptionRow:
    """Persist user-provided context for a photo."""
    vision = session.get(VisionDescriptionRow, photo_row.id)
    if vision:
        vision.user_context = context
    else:
        vision = VisionDescriptionRow(
            photo_id=photo_row.id,
            description="",
            model=None,
            confidence=None,
            user_context=context,
        )
        session.add(vision)
    session.flush()
    return vision
