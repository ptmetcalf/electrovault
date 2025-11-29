#!/usr/bin/env python
# ruff: noqa: E402
"""
Test LocationIQ reverse geocoding for a single image.

Examples:
  python scripts/test_locationiq.py /absolute/path/to/photo.jpg
  python scripts/test_locationiq.py phototest/IMG_0001.JPG --db sqlite+pysqlite:///./photo_brain.db
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the repo root is on the import path when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from photo_brain.core.env import load_dotenv_if_present
from photo_brain.core.models import PhotoFile
from photo_brain.index.location import LocationResolverConfig, resolve_photo_location
from photo_brain.index.schema import PhotoLocationRow, init_db, session_factory
from photo_brain.ingest.exif_reader import read_exif
from photo_brain.ingest.pipeline import upsert_photo


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_photo_model(path: Path) -> PhotoFile:
    stat = path.stat()
    sha256 = _hash_file(path)
    return PhotoFile(
        id=sha256,
        path=str(path.resolve()),
        sha256=sha256,
        size_bytes=stat.st_size,
        mtime=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Send a single image to the LocationIQ resolver and print the assigned location "
            "(cached/user labels respected)."
        )
    )
    parser.add_argument("image", type=Path, help="Path to the image file to resolve.")
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Database URL for caching labels (default: DATABASE_URL env or in-memory SQLite).",
    )
    args = parser.parse_args()

    load_dotenv_if_present()
    config = LocationResolverConfig.from_env()
    if not config.enable_remote:
        print(
            "Warning: LOCATION_RESOLUTION_ENABLED=0 or LOCATIONIQ_API_KEY missing; "
            "only cached/user labels will be used."
        )

    image_path = args.image.expanduser()
    if not image_path.exists():
        sys.exit(f"File not found: {image_path}")
    if not image_path.is_file():
        sys.exit(f"Not a file: {image_path}")

    exif = read_exif(str(image_path))
    if not exif or exif.gps_lat is None or exif.gps_lon is None:
        sys.exit("Image has no GPS coordinates; nothing to resolve.")

    db_url = args.db or os.getenv("DATABASE_URL") or "sqlite+pysqlite:///:memory:"
    engine = init_db(db_url)
    SessionLocal = session_factory(engine)

    with SessionLocal() as session:
        photo_model = _build_photo_model(image_path)
        row = upsert_photo(session, photo_model, exif)
        session.flush()

        resolve_photo_location(session, row, exif, config=config)
        session.commit()

        stored = session.get(PhotoLocationRow, row.id)
        if stored is None or stored.location is None:
            print("No location resolved.")
            return

        label = stored.location
        print("Resolved location:")
        print(f"- Name: {label.name}")
        print(f"- Source: {label.source}")
        print(f"- Method: {stored.method}")
        print(f"- Coordinates: {label.latitude:.6f}, {label.longitude:.6f}")
        if label.radius_meters:
            print(f"- Radius meters: {label.radius_meters}")
        if stored.confidence is not None:
            print(f"- Confidence: {stored.confidence:.2f}")
        print(f"- Label ID: {label.id}")
        if label.raw:
            print("- Raw snippet:", {k: label.raw.get(k) for k in ("type", "display_name", "name")})
            try:
                print("Full raw response:")
                print(json.dumps(label.raw, indent=2))
            except Exception:
                pass


if __name__ == "__main__":
    main()
