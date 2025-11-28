from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
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


class ExifDataRow(Base):
    __tablename__ = "exif_data"

    photo_id: Mapped[str] = mapped_column(
        ForeignKey("photo_files.id", ondelete="CASCADE"), primary_key=True
    )
    datetime_original: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
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
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    user_context: Mapped[Optional[str]] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    photo: Mapped[PhotoFileRow] = relationship(back_populates="vision")


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


class FaceIdentityRow(Base):
    __tablename__ = "face_identities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    detection_id: Mapped[int] = mapped_column(
        ForeignKey("face_detections.id", ondelete="CASCADE"), nullable=False, index=True
    )
    person_label: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    detection: Mapped[FaceDetectionRow] = relationship(back_populates="identities")


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
    engine = database_url if isinstance(database_url, Engine) else create_engine_from_url(database_url)

    if engine.dialect.name == "postgresql":
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

    Base.metadata.create_all(engine)
    return engine


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, class_=Session, future=True)
