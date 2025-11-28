from __future__ import annotations

from typing import List

from sqlalchemy import select
from sqlalchemy.orm import Session

from photo_brain.core.models import MemoryEvent
from photo_brain.index import (
    FaceDetectionRow,
    FaceIdentityRow,
    MemoryEventRow,
    event_photos,
)


def summarize_events(session: Session) -> List[MemoryEvent]:
    events = session.scalars(
        select(MemoryEventRow).order_by(MemoryEventRow.start_time.asc())
    ).all()
    summaries: List[MemoryEvent] = []
    for event in events:
        photo_ids = [photo.id for photo in event.photos]
        people = session.scalars(
            select(FaceIdentityRow.person_label)
            .join(FaceDetectionRow, FaceDetectionRow.id == FaceIdentityRow.detection_id)
            .join(event_photos, event_photos.c.photo_id == FaceDetectionRow.photo_id)
            .where(event_photos.c.event_id == event.id)
        ).all()
        unique_people = sorted(set(filter(None, people)))
        summary_text = f"{len(photo_ids)} photos"
        if unique_people:
            summary_text += f" featuring {', '.join(unique_people)}"
        summaries.append(
            MemoryEvent(
                id=event.id,
                title=event.title,
                photo_ids=photo_ids,
                start_time=event.start_time,
                end_time=event.end_time,
                summary=summary_text,
                people=unique_people,
            )
        )
    return summaries
