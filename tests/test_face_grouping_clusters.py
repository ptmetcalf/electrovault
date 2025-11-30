from __future__ import annotations

import numpy as np

from photo_brain.index import face_grouping


def test_dbscan_clusters_basic() -> None:
    # Two close embeddings and one far away; expect a single cluster of the first two.
    candidates = [
        (1, np.array([1.0, 0.0, 0.0]), None),
        (2, np.array([0.98, 0.05, 0.0]), None),
        (3, np.array([0.0, 1.0, 0.0]), None),
    ]
    clusters = face_grouping._dbscan_clusters(candidates, eps=0.8, min_samples=2)
    assert len(clusters) == 1
    ids, sims, _ = clusters[0]
    assert set(ids) == {1, 2}
    # Similarities to centroid should be high and finite
    assert all(0.5 < s <= 1.0 for s in sims)


def test_dbscan_clusters_respects_min_samples() -> None:
    candidates = [
        (1, np.array([1.0, 0.0]), None),
        (2, np.array([0.9, 0.1]), None),
    ]
    # With min_samples=3, no clusters should form.
    clusters = face_grouping._dbscan_clusters(candidates, eps=0.8, min_samples=3)
    assert clusters == []
