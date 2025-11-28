from __future__ import annotations

import hashlib
from typing import List

from photo_brain.core.models import FaceDetection, FaceIdentity


def recognize_faces(detections: List[FaceDetection]) -> List[FaceIdentity]:
    """Assign stable pseudo-identities based on face encodings."""
    identities: List[FaceIdentity] = []
    for detection in detections:
        if detection.encoding:
            encoding_bytes = bytes(int(abs(v) * 100) % 255 for v in detection.encoding)
        else:
            encoding_bytes = str(detection.bbox).encode("utf-8")
        label = hashlib.sha256(encoding_bytes).hexdigest()[:8]
        person_label = f"person-{label}"
        identities.append(
            FaceIdentity(
                person_id=person_label,
                detection_id=None,
                label=person_label,
                confidence=0.5,
            )
        )
    return identities
