from __future__ import annotations

import json
from pathlib import Path

from photo_brain.vision import model_client


def test_call_vision_api_parses_structured(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OLLAMA_VISION_MODEL", "mock-vision")
    image_path = tmp_path / "img.bin"
    image_path.write_bytes(b"data")

    def fake_post(url: str, payload: dict, timeout: int = 60):
        return {"response": json.dumps({"foo": "bar"})}

    monkeypatch.setattr(model_client, "_post_json", fake_post)
    parsed, raw = model_client._call_vision_api("p", image_path, schema={"type": "object"})
    assert parsed == {"foo": "bar"}
    assert json.loads(raw) == {"foo": "bar"}


def test_call_vision_api_uses_message_content(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OLLAMA_VISION_MODEL", "mock-vision")
    image_path = tmp_path / "img.bin"
    image_path.write_bytes(b"data")

    def fake_post(url: str, payload: dict, timeout: int = 60):
        return {"message": {"content": "hello"}}

    monkeypatch.setattr(model_client, "_post_json", fake_post)
    parsed, raw = model_client._call_vision_api("p", image_path, schema=None)
    assert parsed == "hello"
    assert raw == "hello"


def test_classify_vision_handles_bucket(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "img.bin"
    image_path.write_bytes(b"data")

    def fake_generate(prompt: str, image_path: Path, schema: dict | None = None):
        return (
            {
                "labels": [{"label": "cat", "confidence": 0.8}],
                "bucket": "pets_animals",
                "bucket_confidence": 0.91,
            },
            {"bucket": "pets_animals"},
        )

    monkeypatch.setattr(model_client, "generate_vision_structured", fake_generate)
    labels, raw = model_client.classify_vision("p", image_path)
    assert ("bucket:pets_animals", 0.91) in labels


def test_classify_vision_adds_default_bucket_when_missing(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "img.bin"
    image_path.write_bytes(b"data")

    def fake_generate(prompt: str, image_path: Path, schema: dict | None = None):
        return (
            {"labels": [{"label": "person", "confidence": 0.9}]},
            {"labels": ["person"]},
        )

    monkeypatch.setattr(model_client, "generate_vision_structured", fake_generate)
    labels, _ = model_client.classify_vision("p", image_path)
    assert any(lbl == "bucket:misc_other" for lbl, _ in labels)


def test_classify_vision_filters_people_and_pets(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "img.bin"
    image_path.write_bytes(b"data")

    def fake_generate(prompt: str, image_path: Path, schema: dict | None = None):
        return (
            {
                "labels": [
                    {"label": "dog", "confidence": 0.8},
                    {"label": "cat", "confidence": 0.7},
                    {"label": "tree", "confidence": 0.6},
                ],
                "bucket": "landscapes_outdoors",
            },
            {"labels": ["dog", "cat", "tree"]},
        )

    monkeypatch.setattr(model_client, "generate_vision_structured", fake_generate)
    labels, _ = model_client.classify_vision("p", image_path)
    label_keys = {lbl for lbl, _ in labels}
    assert "object:tree" in label_keys
    assert "object:dog" not in label_keys
    assert "object:cat" not in label_keys
