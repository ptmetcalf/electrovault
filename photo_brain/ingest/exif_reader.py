from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from PIL import Image

from photo_brain.core.models import ExifData

DATETIME_ORIGINAL_TAG = 36867  # EXIF DateTimeOriginal
GPS_INFO_TAG = 34853  # GPSInfo


def _to_float(value: object) -> Optional[float]:
    if isinstance(value, tuple) and len(value) == 2 and value[1]:
        return float(value[0]) / float(value[1])
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _convert_gps_coordinate(values: object, ref: object) -> Optional[float]:
    if not isinstance(values, tuple) or len(values) != 3 or ref is None:
        return None
    parts = [_to_float(v) for v in values]
    if any(p is None for p in parts):
        return None
    degrees, minutes, seconds = parts  # type: ignore[misc]
    coordinate = degrees + minutes / 60.0 + seconds / 3600.0
    if isinstance(ref, str) and ref.upper() in {"S", "W"}:
        coordinate *= -1
    return coordinate


def read_exif(path: str | Path) -> ExifData:
    """Extract EXIF metadata from a photo file."""
    datetime_original: Optional[datetime] = None
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None

    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return ExifData()

            dt_value = exif.get(DATETIME_ORIGINAL_TAG)
            if dt_value:
                try:
                    datetime_original = datetime.strptime(
                        str(dt_value), "%Y:%m:%d %H:%M:%S"
                    ).replace(tzinfo=timezone.utc)
                except ValueError:
                    datetime_original = None

            gps_info = exif.get(GPS_INFO_TAG)
            if isinstance(gps_info, dict):
                gps_lat = _convert_gps_coordinate(gps_info.get(2), gps_info.get(1))
                gps_lon = _convert_gps_coordinate(gps_info.get(4), gps_info.get(3))
    except Exception:
        # Ingest should never fail because of malformed EXIF.
        return ExifData()

    return ExifData(
        datetime_original=datetime_original,
        gps_lat=gps_lat,
        gps_lon=gps_lon,
    )
