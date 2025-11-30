from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pytest

from photo_brain import vision as vision_pkg
from photo_brain.core.models import PhotoFile, TextEmbedding, VisionDescription
from photo_brain.faces import detector
from photo_brain.vision import captioner, classifier, model_client


def _photo(tmp_path: Path) -> PhotoFile:
    img = tmp_path / "img.jpg"
    img.write_bytes(b"\x00\x01")  # stub content; detectors fall back gracefully
    return PhotoFile(
        id="p",
        path=str(img),
        sha256="x" * 64,
        size_bytes=img.stat().st_size,
        mtime=datetime.now(timezone.utc),
    )


def test_captioner_uses_structured(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OLLAMA_VISION_MODEL", "mock-vision")

    def fake_generate(prompt: str, path: Path, schema: dict | None = None) -> Tuple[Any, Any]:
        return {"description": "a scene  ", "confidence": 0.7}, {"description": "a scene"}

    monkeypatch.setattr(captioner, "generate_vision_structured", fake_generate)
    photo = _photo(tmp_path)
    desc = captioner.describe_photo(photo, context="ctx")
    assert desc is not None
    assert desc.description == "a scene."
    assert desc.model == "mock-vision"
    assert not hasattr(desc, "confidence")


def test_captioner_falls_back_to_raw(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OLLAMA_VISION_MODEL", "mock-vision")
    monkeypatch.setattr(
        captioner, "generate_vision_structured", lambda *args, **kwargs: (None, None)
    )
    monkeypatch.setattr(
        captioner, "generate_vision", lambda *args, **kwargs: "raw text with whitespace "
    )
    photo = _photo(tmp_path)
    desc = captioner.describe_photo(photo, context=None)
    assert desc is not None
    assert desc.description == "raw text with whitespace."


def test_classifier_parses_labels(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OLLAMA_VISION_MODEL", "mock-vision")

    def fake_classify(prompt: str, path: Path):
        return [("cat", 0.9), "dog"], "raw"

    monkeypatch.setattr(classifier, "classify_vision", fake_classify)
    photo = _photo(tmp_path)
    classes = classifier.classify_photo(photo, context=None)
    assert classes is not None
    labels = {c.label for c in classes}
    assert {"cat", "dog"} <= labels


def test_classifier_embedding_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OLLAMA_VISION_MODEL", "mock-vision")

    photo = _photo(tmp_path)

    def fake_classify(prompt: str, path: Path):
        raise model_client.LocalModelError("fail")

    def fake_describe(photo: PhotoFile, exif=None, context=None):
        return VisionDescription(
            photo_id=photo.id,
            description="a playful dog in a park",
            model="mock-vision",
            confidence=None,
        )

    def fake_embed(
        text: str, photo_id: str | None = None, model: str = "hash-embedder", dim: int = 16
    ):
        # Simple deterministic vectors: dog/pets -> [1,0], misc -> [0,1]
        vec = [1.0, 0.0] if "dog" in text or "pets_animals" in text else [0.0, 1.0]
        return TextEmbedding(photo_id=photo_id or "q", model="fake-embed", vector=vec, dim=2)

    monkeypatch.setattr(classifier, "classify_vision", fake_classify)
    monkeypatch.setattr(classifier, "describe_photo", fake_describe)
    monkeypatch.setattr(classifier, "embed_description", fake_embed)
    monkeypatch.setattr(
        vision_pkg.taxonomy,
        "taxonomy_labels",
        lambda include_people_and_pets=True: [
            "bucket:pets_animals",
            "bucket:misc_other",
            "object:tree",
        ],
    )

    classes = classifier.classify_photo(photo, context=None)
    assert classes is not None
    labels = {c.label for c in classes}
    assert "bucket:pets_animals" in labels


def test_detector_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FACE_DETECT_THRESHOLD", "0.9")
    monkeypatch.setenv("FACE_DETECT_ALLOW_FALLBACK", "1")

    def _fail():
        raise RuntimeError("fail")

    monkeypatch.setattr(detector, "_load_net", _fail)
    photo = _photo(tmp_path)
    detections = detector.detect_faces(photo)
    assert len(detections) == 1
    assert detections[0].encoding is not None


def test_detector_returns_network_detections(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "face.jpg"
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - pillow is installed in deps
        pytest.skip("Pillow not available")
    Image.new("RGB", (20, 20), color="white").save(image_path)

    class FakeNet:
        def __init__(self, detections):
            self._dets = detections

        def setInput(self, _blob):
            return None

        def forward(self):
            return self._dets

    detections = np.zeros((1, 1, 1, 7), dtype=float)
    detections[0, 0, 0, 2] = 0.95  # confidence
    detections[0, 0, 0, 3] = 0.1
    detections[0, 0, 0, 4] = 0.1
    detections[0, 0, 0, 5] = 0.9
    detections[0, 0, 0, 6] = 0.9

    monkeypatch.setattr(detector, "_load_net", lambda: FakeNet(detections))
    photo = PhotoFile(
        id="net",
        path=str(image_path),
        sha256="x" * 64,
        size_bytes=image_path.stat().st_size,
        mtime=datetime.now(timezone.utc),
    )
    faces = detector.detect_faces(photo)
    assert faces
    assert faces[0].confidence >= 0.9


def test_model_client_normalize_and_temperature(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_TEMPERATURE", raising=False)
    assert model_client._temperature() == 0.0
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "2.0")
    assert model_client._temperature() == 1.0
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "-1")
    assert model_client._temperature() == 0.0
    assert model_client._normalize_label("Hello World!") == "hello world!"
    assert model_client._normalize_label("json") is None
