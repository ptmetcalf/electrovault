from datetime import datetime, timezone
from sqlalchemy.orm import sessionmaker

from photo_brain.core.models import ExifData, LocationLabel
from photo_brain.index import (
    LocationResolverConfig,
    PhotoFileRow,
    PhotoLocationRow,
    init_db,
    resolve_photo_location,
    session_factory,
    upsert_user_location,
)
from photo_brain.index.location import LocationProvider, _pick_location_name


class StubProvider:
    def __init__(self, label: LocationLabel | None):
        self.label = label
        self.calls = 0

    def lookup(self, latitude: float, longitude: float) -> LocationLabel | None:
        self.calls += 1
        return self.label


def _session_and_config() -> tuple[sessionmaker, LocationResolverConfig]:
    engine = init_db("sqlite+pysqlite:///:memory:")
    SessionLocal = session_factory(engine)
    config = LocationResolverConfig(
        enable_remote=True,
        api_key="key",
        base_url="http://example.test",
        timeout=0.1,
        user_radius_meters=500,
        cache_radius_meters=500,
    )
    return SessionLocal, config


def _photo(row_id: str) -> PhotoFileRow:
    return PhotoFileRow(
        id=row_id,
        path=f"/tmp/{row_id}.jpg",
        sha256="0" * 64,
        size_bytes=10,
        mtime=datetime.now(timezone.utc),
    )


def test_user_locations_are_preferred_over_remote() -> None:
    SessionLocal, config = _session_and_config()
    with SessionLocal() as session:
        photo = _photo("p1")
        session.add(photo)
        upsert_user_location(session, "Home", 1.0, 2.0, radius_meters=400)
        provider = StubProvider(
            LocationLabel(name="Remote", latitude=1.0, longitude=2.0, source="api")
        )

        resolve_photo_location(
            session,
            photo,
            ExifData(gps_lat=1.0001, gps_lon=2.0001),
            config=config,
            provider=provider,
        )
        session.commit()

        assigned = session.get(PhotoLocationRow, "p1")
        assert assigned is not None
        assert assigned.location is not None
        assert assigned.location.source == "user"
        assert provider.calls == 0


def test_cache_reused_before_remote_calls() -> None:
    SessionLocal, config = _session_and_config()
    provider = StubProvider(
        LocationLabel(
            name="Park",
            latitude=5.0,
            longitude=6.0,
            source="api",
            raw={"importance": 0.7},
        )
    )
    with SessionLocal() as session:
        p1 = _photo("p1")
        p2 = _photo("p2")
        session.add_all([p1, p2])

        resolve_photo_location(
            session, p1, ExifData(gps_lat=5.0, gps_lon=6.0), config=config, provider=provider
        )
        resolve_photo_location(
            session,
            p2,
            ExifData(gps_lat=5.00001, gps_lon=6.00001),
            config=config,
            provider=provider,
        )
        session.commit()

        loc1 = session.get(PhotoLocationRow, "p1")
        loc2 = session.get(PhotoLocationRow, "p2")
        assert loc1 is not None and loc2 is not None
        assert loc1.location_id == loc2.location_id
        assert provider.calls == 1


def test_pick_location_name_prefers_poi_over_address() -> None:
    data = {
        "address": {
            "road": "123 Main St",
            "city": "Gotham",
            "shop": "Corner Coffee",
        },
        "display_name": "123 Main St, Gotham",
    }
    assert _pick_location_name(data) == "Corner Coffee"

    data_no_poi = {
        "address": {
            "road": "123 Main St",
            "city": "Gotham",
        },
        "display_name": "123 Main St, Gotham",
    }
    assert _pick_location_name(data_no_poi) == "123 Main St"
