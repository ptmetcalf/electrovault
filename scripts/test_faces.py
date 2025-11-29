#!/usr/bin/env python
# ruff: noqa: E402
"""
Quickly run face detection on a single image and print detections.

Example:
  python scripts/test_faces.py /absolute/path/to/photo.jpg --save /tmp/faces.jpg
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the repo root is on the import path when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from photo_brain.core.env import load_dotenv_if_present
from photo_brain.core.models import PhotoFile
from photo_brain.faces import detect_faces


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


def _save_debug(image_path: Path, detections, dest: Path) -> None:
    try:
        import cv2
    except ImportError:
        print("OpenCV not available; cannot save debug image.", file=sys.stderr)
        return

    img = cv2.imread(str(image_path))
    if img is None:
        print("Could not load image for debug save.", file=sys.stderr)
        return

    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.confidence:.2f}"
        cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), img)
    print(f"Saved debug image to {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run face detection on one image and print results."
    )
    parser.add_argument("image", type=Path, help="Path to the image file.")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the image with bounding boxes drawn.",
    )
    args = parser.parse_args()

    load_dotenv_if_present()

    image_path = args.image.expanduser()
    if not image_path.exists():
        sys.exit(f"File not found: {image_path}")
    if not image_path.is_file():
        sys.exit(f"Not a file: {image_path}")

    photo = _build_photo_model(image_path)
    detections = detect_faces(photo)

    print(f"Image: {photo.path}")
    print(f"Detections: {len(detections)}")
    for det in detections:
        print(f"- bbox={det.bbox} conf={det.confidence:.2f} encoding_len={len(det.encoding or [])}")

    if args.save and detections:
        _save_debug(image_path, detections, args.save)


if __name__ == "__main__":
    main()
