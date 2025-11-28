from __future__ import annotations

from datetime import datetime
from typing import Optional

from photo_brain.core.models import SearchQuery


def _parse_date(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def plan_search(text: str, *, limit: int = 10) -> SearchQuery:
    """Normalize a raw search string into a SearchQuery with filters."""
    tokens = text.strip().split()
    remaining: list[str] = []
    people: list[str] = []
    event_ids: list[str] = []
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    for token in tokens:
        lower = token.lower()
        if lower.startswith("person:"):
            value = token.split(":", 1)[1]
            if value:
                people.append(value)
            continue
        if lower.startswith("event:"):
            value = token.split(":", 1)[1]
            if value:
                event_ids.append(value)
            continue
        if lower.startswith(("after:", "since:", "from:")):
            start_date = _parse_date(token.split(":", 1)[1])
            continue
        if lower.startswith(("before:", "until:", "to:")):
            end_date = _parse_date(token.split(":", 1)[1])
            continue
        remaining.append(token)

    normalized = " ".join(remaining).strip()
    return SearchQuery(
        text=normalized,
        limit=limit or 10,
        people=people,
        start_date=start_date,
        end_date=end_date,
        event_ids=event_ids,
    )
