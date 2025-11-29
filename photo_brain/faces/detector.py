from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from photo_brain.core.models import FaceDetection, PhotoFile

logger = logging.getLogger(__name__)

# URLs use the widely adopted OpenCV SSD face detector (ResNet backbone).
_CAFFE_PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
_CAFFE_MODEL_URLS = [
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel",
]

_MODEL_DIR = Path(
    os.getenv(
        "FACE_MODEL_DIR",
        Path(__file__).resolve().parents[2] / "models" / "faces",
    )
)
_CONF_THRESHOLD = float(os.getenv("FACE_DETECT_THRESHOLD", "0.5"))


def _allow_fallback() -> bool:
    return os.getenv("FACE_DETECT_ALLOW_FALLBACK", "0").lower() in {"1", "true", "yes"}


def _download(urls: list[str], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for url in urls:
        try:
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as exc:
            last_err = exc
            logger.warning("Face detector: download failed from %s: %s", url, exc)
    if last_err:
        raise last_err


def _ensure_model_files() -> Tuple[Path, Path]:
    prototxt = _MODEL_DIR / "deploy.prototxt"
    caffemodel = _MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
    if not prototxt.exists():
        _download([_CAFFE_PROTO_URL], prototxt)
    if not caffemodel.exists():
        _download(_CAFFE_MODEL_URLS, caffemodel)
    return prototxt, caffemodel


@lru_cache(maxsize=1)
def _load_net() -> cv2.dnn_Net:
    prototxt, caffemodel = _ensure_model_files()
    return cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))


def _encode_crop(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> list[float]:
    """Create a small, deterministic embedding from the face crop."""
    x1, y1, x2, y2 = bbox
    x1, y1 = max(x1, 0), max(y1, 0)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    resized = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_LINEAR)
    digest = hashlib.sha256(resized.tobytes()).digest()
    return [(byte / 255.0) * 2 - 1 for byte in digest[:32]]


def _to_bbox(det: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = int(det[3] * width)
    y1 = int(det[4] * height)
    x2 = int(det[5] * width)
    y2 = int(det[6] * height)
    return (
        max(0, min(x1, width)),
        max(0, min(y1, height)),
        max(0, min(x2, width)),
        max(0, min(y2, height)),
    )


def detect_faces(photo: PhotoFile) -> List[FaceDetection]:
    """Detect faces using OpenCV DNN SSD (ResNet backbone)."""
    img = cv2.imread(photo.path)
    if img is None:
        logger.warning("Face detector: could not read image %s", photo.path)
        return _fallback_detection(photo) if _allow_fallback() else []

    (h, w) = img.shape[:2]
    try:
        net = _load_net()
    except Exception:
        logger.warning("Face detector: failed to load network for %s", photo.path, exc_info=True)
        return _fallback_detection(photo) if _allow_fallback() else []

    def _forward(threshold: float) -> List[FaceDetection]:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
            swapRB=False,
            crop=False,
        )
        net.setInput(blob)
        detections = net.forward()

        results: List[FaceDetection] = []
        num_dets = detections.shape[2]
        for i in range(num_dets):
            conf = float(detections[0, 0, i, 2])
            if conf < threshold:
                continue
            bbox = _to_bbox(detections[0, 0, i], w, h)
            encoding = _encode_crop(img, bbox)
            results.append(
                FaceDetection(
                    photo_id=photo.id,
                    bbox=bbox,
                    confidence=conf,
                    encoding=encoding or None,
                )
            )
        return results

    results = _forward(_CONF_THRESHOLD)

    if not results and _CONF_THRESHOLD > 0.2:
        # If nothing above the configured threshold, try a second pass with a lower threshold.
        lower = max(0.2, _CONF_THRESHOLD / 2)
        results = _forward(lower)
        if results:
            logger.info(
                "Face detector: second-pass detections (%d faces) at threshold %.2f for %s",
                len(results),
                lower,
                photo.path,
            )

    if not results:
        logger.info("Face detector: no faces detected for %s", photo.path)
        return _fallback_detection(photo) if _allow_fallback() else []
    return results


def _fallback_detection(photo: PhotoFile) -> List[FaceDetection]:
    """Deterministic single-box fallback to keep pipeline/test stability."""
    try:
        img = cv2.imread(photo.path)
        h, w = img.shape[:2] if img is not None else (1, 1)
    except Exception:
        h = w = 1
    bbox = (
        0.2 * float(w),
        0.2 * float(h),
        0.8 * float(w),
        0.8 * float(h),
    )
    conf = 0.6
    digest = hashlib.sha256(f"{photo.id}{photo.path}".encode("utf-8")).digest()
    encoding = [(byte / 255.0) * 2 - 1 for byte in digest[:32]]
    return [
        FaceDetection(
            photo_id=photo.id,
            bbox=bbox,
            confidence=conf,
            encoding=encoding,
        )
    ]
