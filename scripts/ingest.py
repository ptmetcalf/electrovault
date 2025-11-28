#!/usr/bin/env python
"""
Run the ingest + index pipeline over a target photo directory.

Usage:
  python scripts/ingest.py /absolute/path/to/photos
  DATABASE_URL=sqlite+pysqlite:///./photo_brain.db python scripts/ingest.py ~/Pictures
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from photo_brain.core.env import load_dotenv_if_present
from photo_brain.index import init_db, session_factory
from photo_brain.ingest import ingest_and_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest and index a photo directory.")
    parser.add_argument("directory", type=Path, help="Directory containing photos (recursed)")
    args = parser.parse_args()

    load_dotenv_if_present()
    database_url = os.getenv("DATABASE_URL", "sqlite+pysqlite:///./photo_brain.db")
    engine = init_db(database_url)
    SessionLocal = session_factory(engine)
    target = args.directory
    if not target.exists() or not target.is_dir():
        raise FileNotFoundError(f"Directory not found or not a folder: {target}")

    with SessionLocal() as session:
        rows = ingest_and_index(target, session)
    count = len(rows)
    if count == 0:
        print(f"No supported photos found under {target}")
    else:
        print(f"Ingest complete: {count} files processed from {target}")


if __name__ == "__main__":
    main()
