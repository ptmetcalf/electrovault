from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import cv2
import numpy as np

from photo_brain.core.models import (
    Classification,
    CropBox,
    FaceDetection,
    FocalPoint,
    PhotoFile,
    SmartCrop,
)

logger = logging.getLogger(__name__)

DEFAULT_ASPECT = float(os.getenv("GALLERY_THUMB_ASPECT", "1.3333"))  # ~4:3 grid


@dataclass
class _Region:
    x1: float
    y1: float
    x2: float
    y2: float
    kind: str
    confidence: float

    def clamp(self, width: float, height: float) -> "_Region":
        return _Region(
            x1=max(0.0, min(self.x1, width)),
            y1=max(0.0, min(self.y1, height)),
            x2=max(0.0, min(self.x2, width)),
            y2=max(0.0, min(self.y2, height)),
            kind=self.kind,
            confidence=self.confidence,
        )


def _clamp(val: float, low: float, high: float) -> float:
    return max(low, min(val, high))


def _image_size(photo: PhotoFile) -> Optional[tuple[int, int, np.ndarray | None]]:
    """Return (width, height, image) if readable; otherwise None."""
    image = cv2.imread(photo.path)
    if image is None:
        return None
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return None
    return w, h, image


def _infer_subject_type(classifications: Iterable[Classification], face_count: int) -> tuple[str, str]:
    """Map classifier buckets + faces into a coarse subject type and label."""
    bucket_scores: dict[str, float] = {}
    for cls in classifications:
        if cls.label.startswith("bucket:"):
            bucket = cls.label.split(":", 1)[1]
            bucket_scores[bucket] = max(bucket_scores.get(bucket, 0.0), cls.score)

    def _bucket_top(options: set[str]) -> Optional[str]:
        scored = [(b, bucket_scores.get(b, 0.0)) for b in options]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0] if scored and scored[0][1] > 0 else None

    doc_bucket = _bucket_top({"documents", "notes_handwriting", "receipts_bills"})
    if doc_bucket:
        return "document_like", "Document"
    screenshot_bucket = _bucket_top({"screenshots", "diagrams_charts", "maps_navigation"})
    if screenshot_bucket:
        return "screenshot", "Screenshot"

    if face_count > 0 or _bucket_top({"people", "groups_events", "selfie", "pets_animals"}):
        return "photo_people", "People"

    if _bucket_top({"landscapes_outdoors", "home_interiors"}):
        return "photo_scene", "Scene"
    if _bucket_top({"objects_items", "shopping_products", "art_illustration"}):
        return "photo_object", "Object"

    return "photo_scene", "Photo"


def _saliency_region(image: np.ndarray, width: int, height: int) -> Optional[_Region]:
    """Lightweight saliency proxy using Laplacian energy."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None
    target_w = min(360, width)
    target_h = min(360, height)
    if target_w <= 0 or target_h <= 0:
        return None
    small = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(small, cv2.CV_32F)
    mag = np.abs(lap)
    if mag.size == 0:
        return None
    max_val = float(mag.max())
    if max_val <= 0:
        return None
    norm = mag / max_val
    mask = norm >= 0.6
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    scale_x = width / float(target_w)
    scale_y = height / float(target_h)
    return _Region(
        x1=float(x1) * scale_x,
        y1=float(y1) * scale_y,
        x2=float(x2 + 1) * scale_x,
        y2=float(y2 + 1) * scale_y,
        kind="saliency",
        confidence=min(1.0, max_val),
    )


def _score_region(region: _Region, width: float, height: float) -> float:
    """Combine type priority, size, centrality, and confidence into one score."""
    frame_area = max(width * height, 1e-6)
    region_area = max((region.x2 - region.x1) * (region.y2 - region.y1), 1e-6)
    area_ratio = region_area / frame_area
    cx = (region.x1 + region.x2) / 2
    cy = (region.y1 + region.y2) / 2
    dist = math.hypot(cx - width / 2, cy - height / 2)
    max_dist = math.hypot(width / 2, height / 2)
    center_score = 1.0 - min(1.0, dist / max_dist)
    size_score = min(1.0, area_ratio / 0.35)
    priority = {"face": 1.0, "saliency": 0.7, "full": 0.5}
    type_score = priority.get(region.kind, 0.4)
    conf_score = _clamp(region.confidence if region.confidence == region.confidence else 0.5, 0.0, 1.0)
    return type_score * 0.45 + center_score * 0.25 + size_score * 0.15 + conf_score * 0.15


def _fit_crop(region: _Region, width: float, height: float, aspect: float) -> tuple[CropBox, FocalPoint]:
    cx = (region.x1 + region.x2) / 2
    cy = (region.y1 + region.y2) / 2
    w = max(region.x2 - region.x1, 1.0)
    h = max(region.y2 - region.y1, 1.0)
    region_aspect = w / h
    if region_aspect < aspect:
        w = h * aspect
    else:
        h = w / aspect
    padding = 0.08 * max(w, h)
    w = min(width, w + padding)
    h = min(height, h + padding)
    x1 = _clamp(cx - w / 2, 0.0, max(0.0, width - w))
    y1 = _clamp(cy - h / 2, 0.0, max(0.0, height - h))
    crop = CropBox(
        x=x1 / width,
        y=y1 / height,
        w=w / width,
        h=h / height,
    )
    focal = FocalPoint(x=_clamp(cx / width, 0.0, 1.0), y=_clamp(cy / height, 0.0, 1.0))
    return crop, focal


def _summary(type_label: Optional[str], captured_at: Optional[datetime]) -> Optional[str]:
    date_text = captured_at.date().isoformat() if captured_at else None
    if type_label and date_text:
        return f"{type_label} • {date_text}"
    if type_label:
        return type_label
    return date_text


def generate_smart_crop(
    photo: PhotoFile,
    classifications: list[Classification],
    detections: list[FaceDetection],
    *,
    captured_at: datetime | None = None,
    target_aspect: float | None = None,
) -> Optional[SmartCrop]:
    """
    Compute a focal point and crop box that keep the primary subject visible.
    """
    aspect = target_aspect or DEFAULT_ASPECT
    size = _image_size(photo)
    if size is None:
        logger.debug("Smart crop: missing image for %s", photo.path)
        # Fallback to a centered crop when the image cannot be read.
        crop = CropBox(x=0.0, y=0.0, w=1.0, h=1.0)
        focal = FocalPoint(x=0.5, y=0.5)
        subject_type, type_label = _infer_subject_type(classifications, len(detections))
        return SmartCrop(
            photo_id=photo.id,
            subject_type=subject_type,
            render_mode="cover",
            primary_crop=crop,
            focal_point=focal,
            type_label=type_label,
            summary=_summary(type_label, captured_at),
        )

    width, height, image = size
    subject_type, type_label = _infer_subject_type(classifications, len(detections))

    if subject_type in {"document_like", "screenshot"}:
        return SmartCrop(
            photo_id=photo.id,
            subject_type=subject_type,
            render_mode="contain",
            primary_crop=CropBox(x=0.0, y=0.0, w=1.0, h=1.0),
            focal_point=FocalPoint(x=0.5, y=0.5),
            type_label=type_label,
            summary=_summary(type_label, captured_at),
        )

    regions: list[_Region] = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        regions.append(
            _Region(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                kind="face",
                confidence=float(det.confidence),
            ).clamp(width, height)
        )

    saliency = _saliency_region(image, width, height)
    if saliency:
        regions.append(saliency.clamp(width, height))

    if not regions:
        regions.append(_Region(0.0, 0.0, float(width), float(height), "full", 0.4))

    best = max(regions, key=lambda r: _score_region(r, width, height))
    crop, focal = _fit_crop(best, float(width), float(height), aspect)

    return SmartCrop(
        photo_id=photo.id,
        subject_type=subject_type,
        render_mode="cover",
        primary_crop=crop,
        focal_point=focal,
        type_label=type_label,
        summary=_summary(type_label, captured_at),
    )
