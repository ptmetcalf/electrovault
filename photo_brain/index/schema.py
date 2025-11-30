from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    func,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=convention)


class PhotoFileRow(Base):
    __tablename__ = "photo_files"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    path: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    sha256: Mapped[str] = mapped_column(String, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    mtime: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    exif: Mapped["ExifDataRow"] = relationship(
        back_populates="photo", cascade="all, delete-orphan", uselist=False
    )
    vision: Mapped[Optional["VisionDescriptionRow"]] = relationship(
        back_populates="photo", cascade="all, delete-orphan", uselist=False
    )
    classifications: Mapped[list["ClassificationRow"]] = relationship(
        back_populates="photo", cascade="all, delete-orphan"
    )
    embeddings: Mapped[list["TextEmbeddingRow"]] = relationship(
        back_populates="photo", cascade="all, delete-orphan"
    )
    detections: Mapped[list["FaceDetectionRow"]] = relationship(
        back_populates="photo", cascade="all, delete-orphan"
    )
    events: Mapped[list["MemoryEventRow"]] = relationship(
        secondary="event_photos", back_populates="photos"
    )
    location: Mapped[Optional["PhotoLocationRow"]] = relationship(
        back_populates="photo", cascade="all, delete-orphan", uselist=False
    )
    smart_crop: Mapped[Optional["SmartCropRow"]] = relationship(
        back_populates="photo", cascade="all, delete-orphan", uselist=False
    )


class ExifDataRow(Base):
    __tablename__ = "exif_data"

    photo_id: Mapped[str] = mapped_column(
        ForeignKey("photo_files.id", ondelete="CASCADE"), primary_key=True
    )
    datetime_original: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    gps_lat: Mapped[Optional[float]] = mapped_column(Float)
    gps_lon: Mapped[Optional[float]] = mapped_column(Float)
    gps_altitude: Mapped[Optional[float]] = mapped_column(Float)
    gps_altitude_ref: Mapped[Optional[int]] = mapped_column(Integer)
    gps_timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    camera_make: Mapped[Optional[str]] = mapped_column(String)
    camera_model: Mapped[Optional[str]] = mapped_column(String)
    lens_model: Mapped[Optional[str]] = mapped_column(String)
    software: Mapped[Optional[str]] = mapped_column(String)
    orientation: Mapped[Optional[int]] = mapped_column(Integer)
    exposure_time: Mapped[Optional[float]] = mapped_column(Float)
    f_number: Mapped[Optional[float]] = mapped_column(Float)
    iso: Mapped[Optional[int]] = mapped_column(Integer)
    focal_length: Mapped[Optional[float]] = mapped_column(Float)

    photo: Mapped[PhotoFileRow] = relationship(back_populates="exif")


class VisionDescriptionRow(Base):
    __tablename__ = "vision_descriptions"

    photo_id: Mapped[str] = mapped_column(
        ForeignKey("photo_files.id", ondelete="CASCADE"), primary_key=True
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[Optional[str]] = mapped_column(String)
    user_context: Mapped[Optional[str]] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    photo: Mapped[PhotoFileRow] = relationship(back_populates="vision")


class SmartCropRow(Base):
    __tablename__ = "smart_crops"

    photo_id: Mapped[str] = mapped_column(
        ForeignKey("photo_files.id", ondelete="CASCADE"), primary_key=True
    )
    subject_type: Mapped[str] = mapped_column(String, nullable=False, default="unknown")
    render_mode: Mapped[str] = mapped_column(String, nullable=False, default="cover")
    crop_x: Mapped[float] = mapped_column(Float, nullable=False)
    crop_y: Mapped[float] = mapped_column(Float, nullable=False)
    crop_w: Mapped[float] = mapped_column(Float, nullable=False)
    crop_h: Mapped[float] = mapped_column(Float, nullable=False)
    focal_x: Mapped[float] = mapped_column(Float, nullable=False)
    focal_y: Mapped[float] = mapped_column(Float, nullable=False)
    type_label: Mapped[Optional[str]] = mapped_column(String)
    summary: Mapped[Optional[str]] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    photo: Mapped[PhotoFileRow] = relationship(back_populates="smart_crop")


class ClassificationRow(Base):
    __tablename__ = "classifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    photo_id: Mapped[str] = mapped_column(
        ForeignKey("photo_files.id", ondelete="CASCADE"), nullable=False, index=True
    )
    label: Mapped[str] = mapped_column(String, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False, default="classifier")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    photo: Mapped[PhotoFileRow] = relationship(back_populates="classifications")


class TextEmbeddingRow(Base):
    __tablename__ = "text_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    photo_id: Mapped[str] = mapped_column(
        ForeignKey("photo_files.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model: Mapped[str] = mapped_column(String, nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    photo: Mapped[PhotoFileRow] = relationship(back_populates="embeddings")


class FaceDetectionRow(Base):
    __tablename__ = "face_detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    photo_id: Mapped[str] = mapped_column(
        ForeignKey("photo_files.id", ondelete="CASCADE"), nullable=False, index=True
    )
    bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x2: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y2: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    encoding: Mapped[Optional[list[float]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    photo: Mapped[PhotoFileRow] = relationship(back_populates="detections")
    identities: Mapped[list["FaceIdentityRow"]] = relationship(
        back_populates="detection", cascade="all, delete-orphan"
    )
    group_memberships: Mapped[list["FaceGroupProposalMemberRow"]] = relationship(
        back_populates="detection", cascade="all, delete-orphan"
    )


class FaceIdentityRow(Base):
    __tablename__ = "face_identities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    detection_id: Mapped[int] = mapped_column(
        ForeignKey("face_detections.id", ondelete="CASCADE"), nullable=False, index=True
    )
    person_label: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    auto_assigned: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("0"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    detection: Mapped[FaceDetectionRow] = relationship(back_populates="identities")


class PersonRow(Base):
    __tablename__ = "persons"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    is_user_confirmed: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("0"))
    auto_assign_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("1"))
    status: Mapped[str] = mapped_column(String, nullable=False, server_default=text("active"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), server_onupdate=func.now()
    )

    links: Mapped[list["FacePersonLinkRow"]] = relationship(
        back_populates="person", cascade="all, delete-orphan"
    )


class FacePersonLinkRow(Base):
    __tablename__ = "face_person_links"
    __table_args__ = (UniqueConstraint("detection_id", name="uq_face_person_detection"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    detection_id: Mapped[int] = mapped_column(
        ForeignKey("face_detections.id", ondelete="CASCADE"), nullable=False, index=True
    )
    person_id: Mapped[str] = mapped_column(
        ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    detection: Mapped[FaceDetectionRow] = relationship()
    person: Mapped[PersonRow] = relationship(back_populates="links")


class FaceGroupProposalRow(Base):
    __tablename__ = "face_group_proposals"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    suggested_label: Mapped[Optional[str]] = mapped_column(String)
    suggested_person_id: Mapped[Optional[str]] = mapped_column(String)
    score_min: Mapped[Optional[float]] = mapped_column(Float)
    score_max: Mapped[Optional[float]] = mapped_column(Float)
    score_mean: Mapped[Optional[float]] = mapped_column(Float)
    size: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    members: Mapped[list["FaceGroupProposalMemberRow"]] = relationship(
        back_populates="proposal", cascade="all, delete-orphan"
    )


class FaceGroupProposalMemberRow(Base):
    __tablename__ = "face_group_proposal_members"
    __table_args__ = (UniqueConstraint("detection_id", name="uq_group_member_detection"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    proposal_id: Mapped[str] = mapped_column(
        ForeignKey("face_group_proposals.id", ondelete="CASCADE"), nullable=False, index=True
    )
    detection_id: Mapped[int] = mapped_column(
        ForeignKey("face_detections.id", ondelete="CASCADE"), nullable=False, index=True
    )
    similarity: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    proposal: Mapped[FaceGroupProposalRow] = relationship(back_populates="members")
    detection: Mapped[FaceDetectionRow] = relationship(back_populates="group_memberships")


class LocationLabelRow(Base):
    __tablename__ = "location_labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    radius_meters: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    source: Mapped[str] = mapped_column(String, nullable=False, default="api")
    raw: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    photos: Mapped[list["PhotoLocationRow"]] = relationship(
        back_populates="location", cascade="all, delete-orphan"
    )


class PersonStatsRow(Base):
    __tablename__ = "person_stats"

    person_id: Mapped[str] = mapped_column(
        ForeignKey("persons.id", ondelete="CASCADE"), primary_key=True
    )
    embedding_centroid: Mapped[Optional[list[float]]] = mapped_column(JSON)
    embedding_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cluster_spread: Mapped[Optional[float]] = mapped_column(Float)
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class PhotoLocationRow(Base):
    __tablename__ = "photo_locations"

    photo_id: Mapped[str] = mapped_column(
        ForeignKey("photo_files.id", ondelete="CASCADE"), primary_key=True
    )
    location_id: Mapped[int] = mapped_column(
        ForeignKey("location_labels.id", ondelete="CASCADE"), nullable=False
    )
    method: Mapped[str] = mapped_column(String, nullable=False, default="cache")
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    photo: Mapped[PhotoFileRow] = relationship(back_populates="location")
    location: Mapped[LocationLabelRow] = relationship(back_populates="photos")


class MemoryEventRow(Base):
    __tablename__ = "memory_events"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text)
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    photos: Mapped[list[PhotoFileRow]] = relationship(
        secondary="event_photos", back_populates="events"
    )


event_photos = Table(
    "event_photos",
    Base.metadata,
    # Table is defined with Column objects because it is an association table.
    # Using Column keeps compatibility across SQLite and Postgres.
    Column(
        "event_id",
        ForeignKey("memory_events.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "photo_id",
        ForeignKey("photo_files.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


def _enable_sqlite_foreign_keys(dbapi_connection: object, _: object) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def create_engine_from_url(database_url: str) -> Engine:
    engine = create_engine(database_url, future=True)
    if engine.dialect.name == "sqlite":
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)
    return engine


def init_db(database_url: str | Engine) -> Engine:
    engine = (
        database_url if isinstance(database_url, Engine) else create_engine_from_url(database_url)
    )

    if engine.dialect.name == "postgresql":
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

    Base.metadata.create_all(engine)

    def _add_column_if_missing(conn: Engine, table: str, name: str, ddl: str) -> None:
        if engine.dialect.name != "sqlite":
            return
        info = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
        if not info:
            return
        cols = {row[1] for row in info}
        if name not in cols:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))

    if engine.dialect.name == "sqlite":
        with engine.begin() as conn:
            _add_column_if_missing(conn, "persons", "is_user_confirmed", "BOOLEAN DEFAULT 0")
            _add_column_if_missing(conn, "persons", "auto_assign_enabled", "BOOLEAN DEFAULT 1")
            _add_column_if_missing(conn, "persons", "status", "TEXT DEFAULT 'active'")
            _add_column_if_missing(conn, "persons", "updated_at", "DATETIME")
            _add_column_if_missing(conn, "face_identities", "auto_assigned", "BOOLEAN DEFAULT 0")
            _add_column_if_missing(conn, "face_group_proposals", "suggested_person_id", "TEXT")

    return engine


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, class_=Session, future=True)
