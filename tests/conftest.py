from __future__ import annotations

import pytest

from photo_brain.faces import detector


@pytest.fixture(autouse=True)
def force_detector_fallback(monkeypatch):
    """Avoid downloading face models during tests by forcing the fallback path."""

    def _fail():
        raise RuntimeError("skip net")

    monkeypatch.setattr(detector, "_load_net", _fail)
    monkeypatch.setenv("FACE_DETECT_ALLOW_FALLBACK", "1")
    yield
