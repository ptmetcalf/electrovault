from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from photo_brain.core.models import ExifData, LocationLabel

from .schema import LocationLabelRow, PhotoFileRow, PhotoLocationRow

logger = logging.getLogger(__name__)


def _haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return approximate great-circle distance in meters."""
    radius_earth_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_earth_km * c * 1000


@dataclass
class LocationResolverConfig:
    enable_remote: bool
    api_key: Optional[str]
    base_url: str
    timeout: float
    user_radius_meters: int
    cache_radius_meters: int

    @classmethod
    def from_env(cls) -> "LocationResolverConfig":
        api_key = os.getenv("LOCATIONIQ_API_KEY")
        enable_remote = os.getenv("LOCATION_RESOLUTION_ENABLED", "0") == "1" and bool(api_key)
        user_radius = int(os.getenv("LOCATION_USER_RADIUS_METERS", "100"))
        cache_radius = int(os.getenv("LOCATION_CACHE_RADIUS_METERS", "250"))
        base_url = os.getenv("LOCATIONIQ_BASE_URL", "https://us1.locationiq.com/v1")
        timeout = float(os.getenv("LOCATIONIQ_HTTP_TIMEOUT", "5.0"))
        return cls(
            enable_remote=enable_remote,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            user_radius_meters=user_radius,
            cache_radius_meters=cache_radius,
        )


class LocationProvider(Protocol):
    def lookup(self, latitude: float, longitude: float) -> Optional[LocationLabel]: ...


class LocationIQProvider:
    """Reverse geocoder backed by LocationIQ (OpenStreetMap data)."""

    def __init__(self, config: LocationResolverConfig, client: Optional[httpx.Client] = None):
        self.config = config
        self.client = client or httpx.Client(base_url=config.base_url, timeout=config.timeout)

    def lookup(self, latitude: float, longitude: float) -> Optional[LocationLabel]:
        if not self.config.api_key:
            return None
        try:
            response = self.client.get(
                "reverse",
                params={
                    "key": self.config.api_key,
                    "lat": latitude,
                    "lon": longitude,
                    "format": "json",
                },
            )
            response.raise_for_status()
            data = response.json()
            name = _pick_location_name(data)
            resolved_lat = float(data.get("lat", latitude))
            resolved_lon = float(data.get("lon", longitude))
            return LocationLabel(
                name=name,
                latitude=resolved_lat,
                longitude=resolved_lon,
                source="api",
                raw=data,
            )
        except Exception as exc:  # pragma: no cover - network/environment failures
            logger.warning("LocationIQ lookup failed: %s", exc)
            return None


def _pick_location_name(data: dict[str, Any]) -> str:
    """Choose a human-friendly name favoring POIs over addresses."""
    if isinstance(data.get("name"), str) and data["name"].strip():
        return data["name"].strip()

    address = data.get("address") if isinstance(data.get("address"), dict) else {}
    priority_fields = [
        "shop",
        "amenity",
        "tourism",
        "attraction",
        "leisure",
        "man_made",
        "building",
        "place",
        "road",
    ]
    for field in priority_fields:
        val = address.get(field)
        if isinstance(val, str) and val.strip():
            return val.strip()

    locality_fields = ["neighbourhood", "suburb", "village", "town", "city"]
    for field in locality_fields:
        val = address.get(field)
        if isinstance(val, str) and val.strip():
            return val.strip()

    display = data.get("display_name")
    if isinstance(display, str) and display.strip():
        return display.split(",")[0].strip()

    return "Unknown location"


def _label_matches(
    label: LocationLabelRow, latitude: float, longitude: float, search_radius: int
) -> bool:
    radius = max(search_radius, label.radius_meters or 0)
    distance = _haversine_distance_meters(latitude, longitude, label.latitude, label.longitude)
    return distance <= radius


def _find_nearby_label(
    session: Session,
    latitude: float,
    longitude: float,
    *,
    source: str | None,
    search_radius: int,
) -> Optional[LocationLabelRow]:
    labels = session.scalars(select(LocationLabelRow)).all()
    best: LocationLabelRow | None = None
    best_distance: float | None = None
    for label in labels:
        if source and label.source != source:
            continue
        if _label_matches(label, latitude, longitude, search_radius):
            distance = _haversine_distance_meters(
                latitude, longitude, label.latitude, label.longitude
            )
            if best is None or distance < (best_distance or float("inf")):
                best = label
                best_distance = distance
    return best


def _ensure_label(session: Session, label: LocationLabel) -> LocationLabelRow:
    existing = _find_nearby_label(
        session,
        label.latitude,
        label.longitude,
        source=label.source,
        search_radius=label.radius_meters,
    )
    if existing:
        if label.source == "user" and existing.name != label.name:
            existing.name = label.name
        return existing
    row = LocationLabelRow(
        name=label.name,
        latitude=label.latitude,
        longitude=label.longitude,
        radius_meters=label.radius_meters,
        source=label.source,
        raw=label.raw,
    )
    session.add(row)
    session.flush()
    return row


def _assign_photo_location(
    session: Session,
    photo: PhotoFileRow,
    label_row: LocationLabelRow,
    *,
    method: str,
    confidence: Optional[float],
    respect_existing_user: bool = True,
) -> PhotoLocationRow:
    existing = session.get(PhotoLocationRow, photo.id)
    if (
        respect_existing_user
        and existing
        and existing.location
        and existing.location.source == "user"
    ):
        # Respect explicit user assignment.
        return existing

    if existing:
        existing.location_id = label_row.id
        existing.method = method
        existing.confidence = confidence
        return existing

    location = PhotoLocationRow(
        photo_id=photo.id,
        location_id=label_row.id,
        method=method,
        confidence=confidence,
    )
    session.add(location)
    return location


def resolve_photo_location(
    session: Session,
    photo: PhotoFileRow,
    exif: Optional[ExifData],
    *,
    config: Optional[LocationResolverConfig] = None,
    provider: Optional[LocationProvider] = None,
) -> None:
    """Assign a location to a photo using cached labels, user overrides, or LocationIQ."""
    if exif is None or exif.gps_lat is None or exif.gps_lon is None:
        return
    config = config or LocationResolverConfig.from_env()

    # Respect existing user-defined location if set.
    if photo.location and photo.location.location and photo.location.location.source == "user":
        return

    # User overrides nearest search.
    user_label = _find_nearby_label(
        session,
        exif.gps_lat,
        exif.gps_lon,
        source="user",
        search_radius=config.user_radius_meters,
    )
    if user_label:
        _assign_photo_location(session, photo, user_label, method="user-cache", confidence=1.0)
        return

    cached = _find_nearby_label(
        session,
        exif.gps_lat,
        exif.gps_lon,
        source=None,
        search_radius=config.cache_radius_meters,
    )
    if cached:
        _assign_photo_location(session, photo, cached, method="cache", confidence=0.9)
        return

    if not config.enable_remote:
        return

    provider = provider or LocationIQProvider(config)
    resolved = provider.lookup(exif.gps_lat, exif.gps_lon)
    if resolved is None:
        return

    resolved.radius_meters = config.cache_radius_meters
    label_row = _ensure_label(session, resolved)
    _assign_photo_location(session, photo, label_row, method="api", confidence=resolved.raw.get("importance") if resolved.raw else None)


def upsert_user_location(
    session: Session,
    name: str,
    latitude: float,
    longitude: float,
    *,
    radius_meters: int,
) -> LocationLabelRow:
    label = LocationLabel(
        name=name,
        latitude=latitude,
        longitude=longitude,
        radius_meters=radius_meters,
        source="user",
    )
    return _ensure_label(session, label)


def assign_user_location(
    session: Session, photo: PhotoFileRow, label_row: LocationLabelRow
) -> PhotoLocationRow:
    """Force-assign a user location to a photo."""
    return _assign_photo_location(
        session,
        photo,
        label_row,
        method="user-manual",
        confidence=1.0,
        respect_existing_user=False,
    )
