#!/usr/bin/env python
"""
Test the classifier prompt against a single image using the local vision model.

Example:
  python scripts/test_classifier_prompt.py /absolute/path/to/photo.jpg --show-prompt
"""
from __future__ import annotations

import argparse
import hashlib
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
from photo_brain.vision.classifier import _build_classifier_prompt, classify_photo


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
        description="Send a single image to the classifier prompt and print the returned tags."
    )
    parser.add_argument("image", type=Path, help="Path to the image file to classify.")
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Optional context string forwarded to the classifier prompt.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the full prompt sent to the vision model.",
    )
    args = parser.parse_args()

    load_dotenv_if_present()
    model_name = os.getenv("OLLAMA_VISION_MODEL")
    if not model_name:
        sys.exit("OLLAMA_VISION_MODEL is not set; configure your local vision model first.")

    image_path = args.image.expanduser()
    if not image_path.exists():
        sys.exit(f"File not found: {image_path}")
    if not image_path.is_file():
        sys.exit(f"Not a file: {image_path}")

    prompt = _build_classifier_prompt(args.context)
    if args.show_prompt:
        print("=== Classifier prompt ===")
        print(prompt)
        print()

    photo = _build_photo_model(image_path)
    classifications = classify_photo(photo, context=args.context)
    if not classifications:
        sys.exit(
            "No classifications returned. Ensure the local vision model is running and produces parseable output."
        )

    print(f"Image: {photo.path}")
    print(f"Model: {model_name}")
    print("Tags:")
    for cls in classifications:
        print(f"- {cls.label} ({cls.score:.2f})")


if __name__ == "__main__":
    main()
