from photo_brain.vision import model_client


def test_normalize_label_filters_noise_and_cleans_text() -> None:
    normalize = getattr(model_client, "_normalize_label")
    assert normalize("confidence") is None
    assert normalize("0.58") is None
    assert normalize("{ label") is None
    assert normalize("```json") is None
    assert normalize("family_tree") == "family tree"
    assert normalize("Festive lights in background") == "festive lights in background"
    assert normalize("scene:indoor") == "scene:indoor"
    assert normalize("object:girl in birthday party hat") is None  # too many words
    assert normalize("scene:girl is outdoors") is None


def test_normalize_confidence_bounds() -> None:
    norm_conf = getattr(model_client, "_normalize_conf")
    assert norm_conf(0.8) == 0.8
    assert norm_conf(0.9876) == 0.99
    assert norm_conf(1.0) == 0.99
    assert norm_conf(1.2) is None
    assert norm_conf("not-a-number") is None


def test_norm_score_rounds_and_clamps() -> None:
    from photo_brain.vision.classifier import _norm_score

    assert _norm_score(0.1234) == 0.12
    assert _norm_score(0.999) == 0.99
    assert _norm_score(-0.5) == 0.0
    assert _norm_score(1.5) == 0.99


def test_taxonomy_mapping() -> None:
    from photo_brain.vision.taxonomy import map_label

    assert map_label("scene", "indoor") == ("scene", "indoor")
    assert map_label("scene", "indoors") == ("scene", "indoor")
    assert map_label("object", "cat") == ("object", "cat")
    assert map_label("event", "birthday party") == ("event", "birthday")
    assert map_label(None, "portrait") == (None, "portrait")
    assert map_label("brand", "unknownbrand") is None
    assert map_label("quality", "blur-high") == ("quality", "blur-high")
