from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image


def build_thumbnail(
    photo_path: Path,
    photo_id: str,
    cache_dir: Path,
    max_size: int = 320,
) -> Optional[Path]:
    """Create a cached JPEG thumbnail for a photo. Returns the thumb path or None on failure."""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        thumb_path = cache_dir / f"{photo_id}.jpg"
        with Image.open(photo_path) as img:
            img = img.convert("RGB")
            img.thumbnail((max_size, max_size))
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)
            thumb_path.write_bytes(buf.getvalue())
        return thumb_path
    except Exception:
        return None
