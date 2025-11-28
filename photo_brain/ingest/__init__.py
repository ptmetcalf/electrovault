"""Ingest pipeline for scanning photos and reading EXIF metadata."""

from .exif_reader import read_exif
from .pipeline import ingest_and_index, ingest_directory, upsert_photo
from .scanner import SUPPORTED_EXTENSIONS, scan_photos

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "ingest_and_index",
    "ingest_directory",
    "read_exif",
    "scan_photos",
    "upsert_photo",
]
