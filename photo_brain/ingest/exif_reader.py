from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import numbers

from PIL import Image

from photo_brain.core.models import ExifData

DATETIME_ORIGINAL_TAG = 36867  # EXIF DateTimeOriginal
DATETIME_TAG = 306  # EXIF DateTime fallback
GPS_INFO_TAG = 34853  # GPSInfo
ORIENTATION_TAG = 274
MAKE_TAG = 271
MODEL_TAG = 272
SOFTWARE_TAG = 305
LENS_MODEL_TAG = 42036
EXPOSURE_TIME_TAG = 33434
FNUMBER_TAG = 33437
ISO_TAG = 34855  # PhotographicSensitivity
FOCAL_LENGTH_TAG = 37386


def _to_float(value: object) -> Optional[float]:
    if isinstance(value, tuple) and len(value) == 2 and value[1]:
        return float(value[0]) / float(value[1])
    if isinstance(value, (int, float)) or isinstance(value, numbers.Real):
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
    gps_altitude: Optional[float] = None
    gps_altitude_ref: Optional[int] = None
    gps_timestamp: Optional[datetime] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    lens_model: Optional[str] = None
    software: Optional[str] = None
    orientation: Optional[int] = None
    exposure_time: Optional[float] = None
    f_number: Optional[float] = None
    iso: Optional[int] = None
    focal_length: Optional[float] = None

    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return ExifData()

            dt_value = exif.get(DATETIME_ORIGINAL_TAG) or exif.get(DATETIME_TAG)
            if dt_value:
                try:
                    datetime_original = datetime.strptime(
                        str(dt_value), "%Y:%m:%d %H:%M:%S"
                    ).replace(tzinfo=timezone.utc)
                except ValueError:
                    datetime_original = None

            orientation_val = exif.get(ORIENTATION_TAG)
            if isinstance(orientation_val, int):
                orientation = orientation_val

            camera_make = str(exif.get(MAKE_TAG) or "").strip() or None
            camera_model = str(exif.get(MODEL_TAG) or "").strip() or None
            software = str(exif.get(SOFTWARE_TAG) or "").strip() or None
            lens_raw = str(exif.get(LENS_MODEL_TAG) or "").strip() or None
            lens_model = lens_raw.lower() if lens_raw else None

            exposure_time_val = _to_float(exif.get(EXPOSURE_TIME_TAG))
            if exposure_time_val is not None:
                exposure_time = exposure_time_val
            f_number_val = _to_float(exif.get(FNUMBER_TAG))
            if f_number_val is not None:
                f_number = f_number_val
            iso_val = exif.get(ISO_TAG)
            if isinstance(iso_val, int):
                iso = iso_val
            focal_length_val = _to_float(exif.get(FOCAL_LENGTH_TAG))
            if focal_length_val is not None:
                focal_length = focal_length_val

            gps_info = exif.get(GPS_INFO_TAG)
            if isinstance(gps_info, int):
                # Pillow stores the GPS IFD offset as an int; need to dereference.
                try:
                    gps_info = exif.get_ifd(GPS_INFO_TAG)
                except Exception:
                    gps_info = None
            if isinstance(gps_info, dict):
                gps_lat = _convert_gps_coordinate(gps_info.get(2), gps_info.get(1))
                gps_lon = _convert_gps_coordinate(gps_info.get(4), gps_info.get(3))
                gps_altitude = _to_float(gps_info.get(6))
                alt_ref = gps_info.get(5)
                if isinstance(alt_ref, (int, float, bytes)):
                    # bytes are common for altitude ref; treat non-zero as below sea level
                    if isinstance(alt_ref, bytes) and len(alt_ref) > 0:
                        gps_altitude_ref = 1 if alt_ref[0] != 0 else 0
                    else:
                        gps_altitude_ref = int(alt_ref)
                timestamp = gps_info.get(7)
                datestamp = gps_info.get(29)
                if (
                    isinstance(timestamp, tuple)
                    and len(timestamp) == 3
                    and isinstance(datestamp, str)
                ):
                    parts = []
                    for v in timestamp:
                        fv = _to_float(v)
                        if fv is None:
                            parts = []
                            break
                        parts.append(int(fv))
                    if parts:
                        try:
                            gps_timestamp = datetime(
                                year=int(datestamp.split(":")[0]),
                                month=int(datestamp.split(":")[1]),
                                day=int(datestamp.split(":")[2]),
                                hour=parts[0],
                                minute=parts[1],
                                second=parts[2],
                                tzinfo=timezone.utc,
                            )
                        except Exception:
                            gps_timestamp = None
    except Exception:
        # Ingest should never fail because of malformed EXIF.
        return ExifData()

    return ExifData(
        datetime_original=datetime_original,
        gps_lat=gps_lat,
        gps_lon=gps_lon,
        gps_altitude=gps_altitude,
        gps_altitude_ref=gps_altitude_ref,
        gps_timestamp=gps_timestamp,
        camera_make=camera_make,
        camera_model=camera_model,
        lens_model=lens_model,
        software=software,
        orientation=orientation,
        exposure_time=exposure_time,
        f_number=f_number,
        iso=iso,
        focal_length=focal_length,
    )
