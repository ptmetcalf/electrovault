from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from photo_brain.index.schema import FaceDetectionRow, FaceIdentityRow, PhotoFileRow, VisionDescriptionRow


def assign_face_identity(session: Session, detection_id: int, person_label: str) -> FaceIdentityRow:
    """Set or update the person label for a face detection."""
    detection = session.get(FaceDetectionRow, detection_id)
    if detection is None:
        raise ValueError("Face detection not found")

    identity = session.scalar(
        select(FaceIdentityRow).where(FaceIdentityRow.detection_id == detection_id)
    )
    if identity:
        identity.person_label = person_label
    else:
        identity = FaceIdentityRow(detection_id=detection_id, person_label=person_label)
        session.add(identity)
    session.flush()
    return identity


def set_photo_user_context(session: Session, photo_row: PhotoFileRow, context: str) -> VisionDescriptionRow:
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
