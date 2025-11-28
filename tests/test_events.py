from datetime import datetime, timedelta, timezone

from photo_brain.events import group_events, summarize_events
from photo_brain.index import PhotoFileRow, init_db, session_factory


def test_group_and_summarize_events() -> None:
    engine = init_db("sqlite+pysqlite:///:memory:")
    SessionLocal = session_factory(engine)

    with SessionLocal() as session:
        base_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        photos = [
            PhotoFileRow(
                id="p1",
                path="/tmp/p1.jpg",
                sha256="1" * 64,
                size_bytes=1,
                mtime=base_time,
            ),
            PhotoFileRow(
                id="p2",
                path="/tmp/p2.jpg",
                sha256="2" * 64,
                size_bytes=1,
                mtime=base_time + timedelta(hours=2),
            ),
            PhotoFileRow(
                id="p3",
                path="/tmp/p3.jpg",
                sha256="3" * 64,
                size_bytes=1,
                mtime=base_time + timedelta(days=1),
            ),
        ]
        session.add_all(photos)
        session.commit()

        events = group_events(session, gap_hours=6)
        assert len(events) == 2
        summaries = summarize_events(session)
        assert len(summaries) == 2
        assert summaries[0].photo_ids
        assert summaries[0].start_time <= summaries[0].end_time
