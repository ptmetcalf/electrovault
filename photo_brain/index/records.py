from __future__ import annotations

from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from photo_brain.core.models import (
    Classification,
    CropBox,
    ExifData,
    FaceDetection,
    FaceIdentity,
    FacePreview,
    FocalPoint,
    LocationLabel,
    Person,
    PhotoFile,
    PhotoLocation,
    PhotoRecord,
    TextEmbedding,
    VisionDescription,
    SmartCrop,
)

from .schema import (
    ClassificationRow,
    ExifDataRow,
    FaceDetectionRow,
    FacePersonLinkRow,
    FaceIdentityRow,
    LocationLabelRow,
    PersonRow,
    PersonStatsRow,
    PhotoFileRow,
    PhotoLocationRow,
    SmartCropRow,
    TextEmbeddingRow,
    VisionDescriptionRow,
    event_photos,
)


def _load_exif(row: ExifDataRow | None) -> ExifData | None:
    if row is None:
        return None
    return ExifData(
        datetime_original=row.datetime_original,
        gps_lat=row.gps_lat,
        gps_lon=row.gps_lon,
    )


def _load_location(row: PhotoLocationRow | None) -> PhotoLocation | None:
    if row is None or row.location is None:
        return None
    label_row: LocationLabelRow = row.location
    label = LocationLabel(
        id=label_row.id,
        name=label_row.name,
        latitude=label_row.latitude,
        longitude=label_row.longitude,
        radius_meters=label_row.radius_meters,
        source=label_row.source,
        raw=label_row.raw,
        created_at=label_row.created_at,
    )
    return PhotoLocation(
        photo_id=row.photo_id,
        location=label,
        method=row.method,
        confidence=row.confidence,
        created_at=row.created_at,
    )


def _load_smart_crop(row: SmartCropRow | None) -> SmartCrop | None:
    if row is None:
        return None
    return SmartCrop(
        photo_id=row.photo_id,
        subject_type=row.subject_type,
        render_mode=row.render_mode,
        primary_crop=CropBox(x=row.crop_x, y=row.crop_y, w=row.crop_w, h=row.crop_h),
        focal_point=FocalPoint(x=row.focal_x, y=row.focal_y),
        type_label=row.type_label,
        summary=row.summary,
        created_at=row.created_at,
    )


def list_face_identities(session: Session, photo_id: str) -> list[FaceIdentity]:
    """Return face identities for a photo, ordered by detection id."""
    rows = session.execute(
        select(FaceIdentityRow, FaceDetectionRow, FacePersonLinkRow, PersonRow)
        .join(FaceDetectionRow, FaceDetectionRow.id == FaceIdentityRow.detection_id)
        .outerjoin(FacePersonLinkRow, FacePersonLinkRow.detection_id == FaceDetectionRow.id)
        .outerjoin(PersonRow, PersonRow.id == FacePersonLinkRow.person_id)
        .where(FaceDetectionRow.photo_id == photo_id)
        .order_by(FaceDetectionRow.id)
    ).all()
    identities: list[FaceIdentity] = []
    for face, detection, link, person in rows:
        person_id = link.person_id if link else face.person_label
        label = person.display_name if person else face.person_label
        identities.append(
            FaceIdentity(
                person_id=person_id,
                detection_id=face.detection_id,
                label=label,
                confidence=face.confidence,
                auto_assigned=bool(getattr(face, "auto_assigned", False)),
                created_at=face.created_at,
            )
        )
    return identities


def _load_detections(session: Session, photo_id: str) -> list[FaceDetection]:
    rows = session.scalars(
        select(FaceDetectionRow).where(FaceDetectionRow.photo_id == photo_id)
    ).all()
    return [
        FaceDetection(
            id=row.id,
            photo_id=row.photo_id,
            bbox=(row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2),
            confidence=row.confidence,
            encoding=row.encoding,
            created_at=row.created_at,
        )
        for row in rows
    ]


def list_face_previews(
    session: Session,
    *,
    unassigned: bool | None = None,
    person: str | None = None,
    limit: int = 24,
    offset: int = 0,
) -> tuple[list[FacePreview], int]:
    """Return face detections with optional identities and the parent photo."""
    det = FaceDetectionRow
    identity = FaceIdentityRow
    link = FacePersonLinkRow
    person_row = PersonRow
    photo = PhotoFileRow

    stmt = (
        select(det, identity, link, person_row, photo)
        .join(photo, det.photo_id == photo.id)
        .outerjoin(identity, identity.detection_id == det.id)
        .outerjoin(link, link.detection_id == det.id)
        .outerjoin(person_row, person_row.id == link.person_id)
    )
    count_stmt = (
        select(func.count())
        .select_from(det)
        .outerjoin(identity, identity.detection_id == det.id)
        .outerjoin(link, link.detection_id == det.id)
        .outerjoin(person_row, person_row.id == link.person_id)
    )

    conditions = []
    if unassigned is True:
        conditions.append(link.id.is_(None))
    elif unassigned is False:
        conditions.append(link.id.is_not(None))
    if person:
        conditions.append(
            person_row.display_name.ilike(f"%{person}%") | identity.person_label.ilike(f"%{person}%")
        )
    if conditions:
        stmt = stmt.where(*conditions)
        count_stmt = count_stmt.where(*conditions)

    stmt = stmt.order_by(det.created_at.desc()).offset(offset).limit(limit)

    rows = session.execute(stmt).all()
    faces: list[FacePreview] = []
    for det_row, identity_row, link_row, person_row_obj, photo_row in rows:
        detection = FaceDetection(
            id=det_row.id,
            photo_id=det_row.photo_id,
            bbox=(det_row.bbox_x1, det_row.bbox_y1, det_row.bbox_x2, det_row.bbox_y2),
            confidence=det_row.confidence,
            encoding=det_row.encoding,
            created_at=det_row.created_at,
        )
        face_identity: FaceIdentity | None = None
        if link_row and person_row_obj:
            face_identity = FaceIdentity(
                person_id=person_row_obj.id,
                detection_id=det_row.id,
                label=person_row_obj.display_name,
                confidence=identity_row.confidence if identity_row else None,
                auto_assigned=bool(identity_row.auto_assigned) if identity_row else False,
                created_at=identity_row.created_at if identity_row else link_row.created_at,
            )
        elif identity_row:
            face_identity = FaceIdentity(
                person_id=identity_row.person_label,
                detection_id=identity_row.detection_id,
                label=identity_row.person_label,
                confidence=identity_row.confidence,
                auto_assigned=bool(identity_row.auto_assigned),
                created_at=identity_row.created_at,
            )
        faces.append(
            FacePreview(
                detection=detection,
                identity=face_identity,
                photo=PhotoFile(
                    id=photo_row.id,
                    path=photo_row.path,
                    sha256=photo_row.sha256,
                    size_bytes=photo_row.size_bytes,
                    mtime=photo_row.mtime,
                ),
            )
        )

    total = session.scalar(count_stmt) or 0
    return faces, int(total)


def build_photo_record(
    session: Session, photo_row: PhotoFileRow, *, embedding_model: str | None = None
) -> PhotoRecord:
    exif = _load_exif(photo_row.exif)
    location_row = session.get(PhotoLocationRow, photo_row.id)
    location = _load_location(location_row)
    smart_crop = _load_smart_crop(photo_row.smart_crop)

    vision_row = session.scalar(
        select(VisionDescriptionRow).where(VisionDescriptionRow.photo_id == photo_row.id)
    )
    vision: VisionDescription | None = None
    if vision_row:
        vision = VisionDescription(
            photo_id=photo_row.id,
            description=vision_row.description,
            model=vision_row.model,
            user_context=vision_row.user_context,
            created_at=vision_row.created_at,
        )

    classifications = [
        Classification(
            photo_id=cls.photo_id,
            label=cls.label,
            score=cls.score,
            source=cls.source,
            created_at=cls.created_at,
        )
        for cls in session.scalars(
            select(ClassificationRow).where(ClassificationRow.photo_id == photo_row.id)
        ).all()
    ]

    embedding_row: Optional[TextEmbeddingRow] = session.scalar(
        select(TextEmbeddingRow)
        .where(TextEmbeddingRow.photo_id == photo_row.id)
        .order_by(TextEmbeddingRow.created_at.desc())
    )
    if embedding_model:
        specific_row = session.scalar(
            select(TextEmbeddingRow)
            .where(
                TextEmbeddingRow.photo_id == photo_row.id,
                TextEmbeddingRow.model == embedding_model,
            )
            .order_by(TextEmbeddingRow.created_at.desc())
        )
        if specific_row:
            embedding_row = specific_row

    embedding: TextEmbedding | None = None
    if embedding_row:
        embedding = TextEmbedding(
            photo_id=embedding_row.photo_id,
            model=embedding_row.model,
            vector=embedding_row.embedding,
            dim=embedding_row.dim,
            created_at=embedding_row.created_at,
        )

    faces = list_face_identities(session, photo_row.id)
    detections = _load_detections(session, photo_row.id)

    event_ids = session.scalars(
        select(event_photos.c.event_id).where(event_photos.c.photo_id == photo_row.id)
    ).all()

    return PhotoRecord(
        file=PhotoFile(
            id=photo_row.id,
            path=photo_row.path,
            sha256=photo_row.sha256,
            size_bytes=photo_row.size_bytes,
            mtime=photo_row.mtime,
        ),
        exif=exif,
        vision=vision,
        classifications=classifications,
        smart_crop=smart_crop,
        embedding=embedding,
        detections=detections,
        faces=faces,
        location=location,
        event_ids=event_ids,
    )


def load_photo_record(
    session: Session, photo_id: str, *, embedding_model: str | None = None
) -> PhotoRecord | None:
    row = session.get(PhotoFileRow, photo_id)
    if row is None:
        return None
    return build_photo_record(session, row, embedding_model=embedding_model)


def list_persons(
    session: Session, *, search: str | None = None, limit: int = 50, offset: int = 0
) -> tuple[list[Person], int]:
    """Return people with basic stats (face count and a sample photo id)."""
    filter_clause = []
    if search:
        filter_clause.append(PersonRow.display_name.ilike(f"%{search}%"))

    count_stmt = select(func.count()).select_from(PersonRow)
    if filter_clause:
        count_stmt = count_stmt.where(*filter_clause)
    total = int(session.scalar(count_stmt) or 0)

    rows = (
        session.execute(
            select(
                PersonRow,
                func.count(FacePersonLinkRow.detection_id).label("face_count"),
                func.min(FaceDetectionRow.photo_id).label("sample_photo_id"),
            )
            .join(FacePersonLinkRow, FacePersonLinkRow.person_id == PersonRow.id, isouter=True)
            .join(FaceDetectionRow, FaceDetectionRow.id == FacePersonLinkRow.detection_id, isouter=True)
            .where(*filter_clause)
            .group_by(PersonRow.id)
            .order_by(PersonRow.created_at.desc())
            .offset(offset)
            .limit(limit)
        ).all()
        if total
        else []
    )

    people: list[Person] = []
    for person_row, face_count, sample_photo_id in rows:
        people.append(
            Person(
                id=person_row.id,
                display_name=person_row.display_name,
                face_count=int(face_count or 0),
                is_user_confirmed=bool(person_row.is_user_confirmed),
                auto_assign_enabled=bool(person_row.auto_assign_enabled),
                status=person_row.status,
                sample_photo_id=sample_photo_id,
                created_at=person_row.created_at,
                updated_at=getattr(person_row, "updated_at", None),
            )
        )
    return people, total
