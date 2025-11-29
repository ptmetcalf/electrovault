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
