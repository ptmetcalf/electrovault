from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from photo_brain.core.models import PhotoFile

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_photos(root: str | Path) -> list[PhotoFile]:
    """Scan a directory tree for photo files and return PhotoFile models."""
    root_path = Path(root)
    files: list[PhotoFile] = []
    for path in root_path.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        stat = path.stat()
        sha256 = _hash_file(path)
        files.append(
            PhotoFile(
                id=sha256,
                path=str(path.resolve()),
                sha256=sha256,
                size_bytes=stat.st_size,
                mtime=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            )
        )
    return files
