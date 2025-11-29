from __future__ import annotations

from datetime import timedelta
from typing import List

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from photo_brain.index import MemoryEventRow, PhotoFileRow, event_photos


def group_events(session: Session, gap_hours: int = 12) -> List[MemoryEventRow]:
    """Cluster photos into events based on time gaps."""
    session.execute(delete(event_photos))
    session.execute(delete(MemoryEventRow))

    photos = session.scalars(select(PhotoFileRow).order_by(PhotoFileRow.mtime.asc())).all()
    if not photos:
        session.commit()
        return []

    events: List[list[PhotoFileRow]] = []
    current_group: list[PhotoFileRow] = []
    last_time = None
    gap = timedelta(hours=gap_hours)
    for photo in photos:
        ts = photo.exif.datetime_original if photo.exif else photo.mtime
        if last_time and ts - last_time > gap and current_group:
            events.append(current_group)
            current_group = []
        current_group.append(photo)
        last_time = ts
    if current_group:
        events.append(current_group)

    persisted_events: List[MemoryEventRow] = []
    for idx, group in enumerate(events):
        start = group[0].exif.datetime_original if group[0].exif else group[0].mtime
        end = group[-1].exif.datetime_original if group[-1].exif else group[-1].mtime
        event = MemoryEventRow(
            id=f"evt-{idx}-{int(start.timestamp())}",
            title=start.strftime("Event on %Y-%m-%d"),
            summary="",
            start_time=start,
            end_time=end,
        )
        event.photos.extend(group)
        session.add(event)
        persisted_events.append(event)

    session.commit()
    return persisted_events
