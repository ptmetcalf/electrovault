from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PhotoFile(BaseModel):
    id: str
    path: str
    sha256: str
    size_bytes: int
    mtime: datetime


class ExifData(BaseModel):
    datetime_original: Optional[datetime] = None
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    gps_altitude: Optional[float] = None
    gps_altitude_ref: Optional[int] = None
    gps_timestamp: Optional[datetime] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    lens_model: Optional[str] = None
    software: Optional[str] = None
    orientation: Optional[int] = None
    exposure_time: Optional[float] = None
    f_number: Optional[float] = None
    iso: Optional[int] = None
    focal_length: Optional[float] = None


class LocationLabel(BaseModel):
    id: Optional[int] = None
    name: str
    latitude: float
    longitude: float
    radius_meters: int = 100
    source: str = "api"  # api | cache | user
    raw: Optional[dict] = None
    created_at: Optional[datetime] = None


class PhotoLocation(BaseModel):
    photo_id: str
    location: LocationLabel
    method: str = "cache"
    confidence: Optional[float] = None
    created_at: Optional[datetime] = None


class VisionDescription(BaseModel):
    photo_id: str
    description: str
    model: Optional[str] = None
    user_context: Optional[str] = None
    created_at: Optional[datetime] = None


class Classification(BaseModel):
    photo_id: str
    label: str
    score: float
    source: str = "classifier"
    created_at: Optional[datetime] = None


class CropBox(BaseModel):
    """Normalized crop rectangle anchored to the original image frame."""

    x: float
    y: float
    w: float
    h: float


class FocalPoint(BaseModel):
    """Normalized focal point within the original image frame."""

    x: float
    y: float


class SmartCrop(BaseModel):
    """Vision-driven crop + focal metadata shared across views."""

    photo_id: str
    subject_type: str = "unknown"  # photo_people | photo_object | photo_scene | document_like | screenshot | unknown
    render_mode: str = "cover"  # cover | contain
    primary_crop: CropBox
    focal_point: FocalPoint
    type_label: Optional[str] = None
    summary: Optional[str] = None
    created_at: Optional[datetime] = None


class FaceDetection(BaseModel):
    id: Optional[int] = None
    photo_id: str
    bbox: tuple[float, float, float, float]
    confidence: float
    encoding: Optional[list[float]] = None
    created_at: Optional[datetime] = None


class FaceIdentity(BaseModel):
    person_id: str
    detection_id: Optional[int] = None
    label: Optional[str] = None
    confidence: Optional[float] = None
    auto_assigned: bool = False
    created_at: Optional[datetime] = None


class Person(BaseModel):
    id: str
    display_name: str
    face_count: int = 0
    is_user_confirmed: bool = False
    auto_assign_enabled: bool = True
    status: str = "active"
    sample_photo_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class PersonStats(BaseModel):
    person_id: str
    embedding_centroid: Optional[list[float]] = None
    embedding_count: int = 0
    cluster_spread: Optional[float] = None
    last_seen_at: Optional[datetime] = None


class FacePreview(BaseModel):
    """Face detection plus optional identity and photo file."""

    detection: FaceDetection
    identity: Optional[FaceIdentity] = None
    photo: PhotoFile


class FaceGroupProposal(BaseModel):
    id: str
    status: str = "pending"
    suggested_label: Optional[str] = None
    suggested_person_id: Optional[str] = None
    score_min: Optional[float] = None
    score_max: Optional[float] = None
    score_mean: Optional[float] = None
    size: int
    members: list[FacePreview] = Field(default_factory=list)
    created_at: Optional[datetime] = None


class TextEmbedding(BaseModel):
    photo_id: str
    model: str
    vector: list[float]
    dim: int
    created_at: Optional[datetime] = None


class PhotoRecord(BaseModel):
    file: PhotoFile
    exif: Optional[ExifData] = None
    vision: Optional[VisionDescription] = None
    classifications: list[Classification] = Field(default_factory=list)
    smart_crop: Optional[SmartCrop] = None
    embedding: Optional[TextEmbedding] = None
    detections: list[FaceDetection] = Field(default_factory=list)
    faces: list[FaceIdentity] = Field(default_factory=list)
    location: Optional[PhotoLocation] = None
    event_ids: list[str] = Field(default_factory=list)


class MemoryEvent(BaseModel):
    id: str
    title: str
    photo_ids: list[str]
    start_time: datetime
    end_time: datetime
    summary: Optional[str] = None
    people: list[str] = Field(default_factory=list)


class SearchQuery(BaseModel):
    text: str
    limit: int = 10
    people: list[str] = Field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    event_ids: list[str] = Field(default_factory=list)


class VectorMatch(BaseModel):
    photo_id: str
    score: float


class SearchResult(BaseModel):
    record: PhotoRecord
    score: float
    matched_filters: list[str] = Field(default_factory=list)
