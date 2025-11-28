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


class VisionDescription(BaseModel):
    photo_id: str
    description: str
    model: Optional[str] = None
    confidence: Optional[float] = None
    created_at: Optional[datetime] = None


class Classification(BaseModel):
    photo_id: str
    label: str
    score: float
    source: str = "classifier"
    created_at: Optional[datetime] = None


class FaceDetection(BaseModel):
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
    embedding: Optional[TextEmbedding] = None
    faces: list[FaceIdentity] = Field(default_factory=list)
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
