from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.orm import Session

from photo_brain.core.models import ExifData, PhotoFile
from photo_brain.index import ExifDataRow, PhotoFileRow
from photo_brain.index.indexer import index_photo
from photo_brain.index.vector_backend import PgVectorBackend

from .exif_reader import read_exif
from .scanner import scan_photos

logger = logging.getLogger(__name__)


def upsert_photo(session: Session, photo: PhotoFile, exif: ExifData | None) -> PhotoFileRow:
    """Insert or update a photo record and attach EXIF data if present."""
    row = session.get(PhotoFileRow, photo.id)
    if row is None:
        row = PhotoFileRow(
            id=photo.id,
            path=photo.path,
            sha256=photo.sha256,
            size_bytes=photo.size_bytes,
            mtime=photo.mtime,
        )
        session.add(row)
    else:
        row.path = photo.path
        row.sha256 = photo.sha256
        row.size_bytes = photo.size_bytes
        row.mtime = photo.mtime

    if exif:
        has_exif = any(
            value is not None
            for value in (
                exif.datetime_original,
                exif.gps_lat,
                exif.gps_lon,
                exif.gps_altitude,
                exif.gps_timestamp,
                exif.camera_make,
                exif.camera_model,
                exif.lens_model,
                exif.software,
                exif.orientation,
                exif.exposure_time,
                exif.f_number,
                exif.iso,
                exif.focal_length,
            )
        )
        if has_exif:
            if row.exif is None:
                row.exif = ExifDataRow(photo_id=photo.id)
            row.exif.datetime_original = exif.datetime_original
            row.exif.gps_lat = exif.gps_lat
            row.exif.gps_lon = exif.gps_lon
            row.exif.gps_altitude = exif.gps_altitude
            row.exif.gps_altitude_ref = exif.gps_altitude_ref
            row.exif.gps_timestamp = exif.gps_timestamp
            row.exif.camera_make = exif.camera_make
            row.exif.camera_model = exif.camera_model
            row.exif.lens_model = exif.lens_model
            row.exif.software = exif.software
            row.exif.orientation = exif.orientation
            row.exif.exposure_time = exif.exposure_time
            row.exif.f_number = exif.f_number
            row.exif.iso = exif.iso
            row.exif.focal_length = exif.focal_length

    return row


def ingest_directory(root: str | Path, session: Session) -> list[PhotoFileRow]:
    """Scan a directory for photos, extract EXIF, and upsert into the DB."""
    logger.info("Ingest: scanning %s", root)
    stored_rows: list[PhotoFileRow] = []
    for photo in scan_photos(root):
        exif = read_exif(photo.path)
        stored_rows.append(upsert_photo(session, photo, exif))
    logger.info("Ingest: scanned %d photos (new or updated)", len(stored_rows))
    session.commit()
    return stored_rows


def ingest_and_index(
    root: str | Path, session: Session, *, backend: PgVectorBackend | None = None
) -> list[PhotoFileRow]:
    """Full ingest pipeline: scan, persist, and generate metadata/embeddings."""
    backend = backend or PgVectorBackend()
    rows = ingest_directory(root, session)
    for row in rows:
        index_photo(session, row, backend=backend)
    return rows
