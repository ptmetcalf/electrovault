import mimetypes
import os
from io import BytesIO
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from photo_brain.core.env import configure_logging, load_dotenv_if_present
from photo_brain.events import group_events, summarize_events
from photo_brain.index import (
    PgVectorBackend,
    PhotoFileRow,
    assign_face_identity,
    assign_user_location,
    index_photo,
    init_db,
    load_photo_record,
    list_face_previews,
    list_persons,
    merge_persons,
    rename_person,
    session_factory,
    set_photo_user_context,
    upsert_user_location,
)
from photo_brain.index.schema import FaceDetectionRow, FaceIdentityRow, FacePersonLinkRow, PersonRow
from photo_brain.ingest import index_existing_photos, ingest_directory
from photo_brain.search import execute_search, plan_search

app = FastAPI(title="Photo Brain API")

load_dotenv_if_present()
configure_logging()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+pysqlite:///./photo_brain.db")
engine = init_db(DATABASE_URL)
SessionLocal = session_factory(engine)
vector_backend = PgVectorBackend()
THUMB_MAX_SIZE = int(os.getenv("THUMB_MAX_SIZE", "320"))
THUMB_CACHE_DIR = Path(
    os.getenv("THUMB_CACHE_DIR", Path(__file__).resolve().parents[2] / "thumbnails")
)

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


class SearchRequest(BaseModel):
    query: str
    limit: int = 10


def get_session() -> Session:
    with SessionLocal() as session:
        yield session


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/search")
def search(request: SearchRequest, session: Session = Depends(get_session)) -> dict:
    search_query = plan_search(request.query, limit=request.limit)
    results = execute_search(session, vector_backend, search_query)
    return {"results": [result.model_dump() for result in results]}


@app.get("/events")
def list_events(session: Session = Depends(get_session)) -> dict:
    group_events(session)
    summaries = summarize_events(session)
    return {"events": [event.model_dump() for event in summaries]}


class ReindexRequest(BaseModel):
    photo_id: str
    context: str | None = None
    preserve_faces: bool = True


class FaceAssignmentRequest(BaseModel):
    detection_id: int
    person_label: str
    reindex: bool = True


class ContextUpdateRequest(BaseModel):
    context: str
    reindex: bool = True


class LocationCreateRequest(BaseModel):
    name: str
    latitude: float
    longitude: float
    radius_meters: int = 100
    photo_id: str | None = None


class PersonRenameRequest(BaseModel):
    display_name: str


class PersonMergeRequest(BaseModel):
    source_id: str
    target_id: str


@app.get("/faces")
def list_faces(
    *,
    unassigned: bool | None = Query(None),
    person: str | None = Query(None),
    limit: int = Query(24, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
) -> dict:
    faces, total = list_face_previews(
        session, unassigned=unassigned, person=person, limit=limit, offset=offset
    )
    return {
        "faces": [face.model_dump() for face in faces],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get("/persons")
def persons(
    *,
    search: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
) -> dict:
    people, total = list_persons(session, search=search, limit=limit, offset=offset)
    return {
        "persons": [person.model_dump() for person in people],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.post("/persons/{person_id}/rename")
def rename_person_endpoint(
    person_id: str, req: PersonRenameRequest, session: Session = Depends(get_session)
) -> dict:
    person = rename_person(session, person_id, req.display_name.strip())
    session.commit()
    return {"person": {"id": person.id, "display_name": person.display_name}}


@app.post("/persons/merge")
def merge_person_endpoint(req: PersonMergeRequest, session: Session = Depends(get_session)) -> dict:
    target = merge_persons(session, req.source_id, req.target_id)
    session.commit()
    return {"person": {"id": target.id, "display_name": target.display_name}}


@app.post("/reindex")
def reindex_photo(req: ReindexRequest, session: Session = Depends(get_session)) -> dict:
    row = session.get(PhotoFileRow, req.photo_id)
    if not row:
        raise HTTPException(status_code=404, detail="Photo not found")
    index_photo(
        session,
        row,
        context=req.context,
        skip_if_fresh=False,
        preserve_faces=req.preserve_faces,
    )
    record = load_photo_record(session, req.photo_id)
    return {"photo": record.model_dump() if record else None}


@app.get("/photos/{photo_id}")
def get_photo(photo_id: str, session: Session = Depends(get_session)) -> dict:
    record = load_photo_record(session, photo_id)
    if not record:
        raise HTTPException(status_code=404, detail="Photo not found")
    return {"photo": record.model_dump()}


@app.post("/photos/{photo_id}/faces")
def assign_face(
    photo_id: str, req: FaceAssignmentRequest, session: Session = Depends(get_session)
) -> dict:
    detection = session.get(FaceDetectionRow, req.detection_id)
    if not detection or detection.photo_id != photo_id:
        raise HTTPException(status_code=404, detail="Face detection not found")
    person_label = req.person_label.strip()
    if not person_label:
        raise HTTPException(status_code=400, detail="Person label is required")

    assign_face_identity(session, detection.id, person_label)
    if req.reindex:
        photo_row = detection.photo or session.get(PhotoFileRow, detection.photo_id)
        if not photo_row:
            raise HTTPException(status_code=404, detail="Photo not found for detection")
        index_photo(
            session,
            photo_row,
            context=None,
            skip_if_fresh=False,
            preserve_faces=True,
        )
    else:
        session.commit()

    record = load_photo_record(session, photo_id)
    return {"photo": record.model_dump() if record else None}


@app.post("/photos/{photo_id}/context")
def update_context(
    photo_id: str, req: ContextUpdateRequest, session: Session = Depends(get_session)
) -> dict:
    row = session.get(PhotoFileRow, photo_id)
    if not row:
        raise HTTPException(status_code=404, detail="Photo not found")
    context_value = req.context.strip()

    if req.reindex:
        index_photo(
            session,
            row,
            context=context_value,
            skip_if_fresh=False,
            preserve_faces=True,
        )
    else:
        set_photo_user_context(session, row, context_value)
        session.commit()

    record = load_photo_record(session, photo_id)
    return {"photo": record.model_dump() if record else None}


@app.post("/locations")
def create_location(req: LocationCreateRequest, session: Session = Depends(get_session)) -> dict:
    if req.radius_meters <= 0:
        raise HTTPException(status_code=400, detail="radius_meters must be positive")
    label_row = upsert_user_location(
        session,
        req.name.strip(),
        req.latitude,
        req.longitude,
        radius_meters=req.radius_meters,
    )
    assigned_record = None
    if req.photo_id:
        photo_row = session.get(PhotoFileRow, req.photo_id)
        if not photo_row:
            raise HTTPException(status_code=404, detail="Photo not found for assignment")
        assign_user_location(session, photo_row, label_row)
        session.flush()
        assigned_record = load_photo_record(session, req.photo_id)
    session.commit()
    payload = {
        "id": label_row.id,
        "name": label_row.name,
        "latitude": label_row.latitude,
        "longitude": label_row.longitude,
        "radius_meters": label_row.radius_meters,
        "source": label_row.source,
    }
    response: dict[str, object] = {"location": payload}
    if assigned_record:
        response["photo"] = assigned_record.model_dump()
    return response


@app.get("/thumb/{photo_id}")
def thumbnail(photo_id: str, session: Session = Depends(get_session)) -> Response:
    """Return a resized JPEG thumbnail for a photo."""
    row = session.get(PhotoFileRow, photo_id)
    if not row:
        raise HTTPException(status_code=404, detail="Photo not found")
    path = Path(row.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Photo file missing")
    cache_path = THUMB_CACHE_DIR / f"{photo_id}.jpg"
    if cache_path.exists():
        return FileResponse(cache_path, media_type="image/jpeg")
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((THUMB_MAX_SIZE, THUMB_MAX_SIZE))
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)
            THUMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(buf.getvalue())
            return Response(content=buf.getvalue(), media_type="image/jpeg")
    except Exception as exc:  # pragma: no cover - fallback for unexpected image errors
        raise HTTPException(status_code=500, detail=f"Error generating thumbnail: {exc}") from exc


@app.get("/photos/{photo_id}/image")
def full_image(photo_id: str, session: Session = Depends(get_session)) -> Response:
    """Return the original image file for a photo."""
    row = session.get(PhotoFileRow, photo_id)
    if not row:
        raise HTTPException(status_code=404, detail="Photo not found")
    path = Path(row.path)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Photo file missing")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type)


@app.get("/faces/{detection_id}/crop")
def face_crop(
    detection_id: int,
    size: int = Query(320, ge=32, le=1024),
    session: Session = Depends(get_session),
) -> Response:
    """Return a cropped JPEG of a detected face."""
    detection = session.get(FaceDetectionRow, detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Face detection not found")
    photo_row = detection.photo or session.get(PhotoFileRow, detection.photo_id)
    if not photo_row:
        raise HTTPException(status_code=404, detail="Photo not found for detection")
    path = Path(photo_row.path)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Photo file missing")

    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            width, height = img.size
            x1 = int(max(0, min(detection.bbox_x1, width)))
            y1 = int(max(0, min(detection.bbox_y1, height)))
            x2 = int(max(x1 + 1, min(detection.bbox_x2, width)))
            y2 = int(max(y1 + 1, min(detection.bbox_y2, height)))
            face_img = img.crop((x1, y1, x2, y2))
            face_img.thumbnail((size, size))
            buf = BytesIO()
            face_img.save(buf, format="JPEG", quality=90)
            return Response(content=buf.getvalue(), media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - unexpected image errors
        raise HTTPException(status_code=500, detail=f"Error generating face crop: {exc}") from exc


class BulkIndexRequest(BaseModel):
    root: str | None = None
    context: str | None = None


def _resolve_ingest_root(root: str | None) -> Path:
    if root:
        path = Path(root).expanduser()
    elif os.getenv("AUTO_INGEST_DIR"):
        path = Path(os.getenv("AUTO_INGEST_DIR", "")).expanduser()
    else:
        path = Path(__file__).resolve().parents[2] / "phototest"
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Ingest path not found: {path}")
    return path


@app.post("/reindex/full")
def reindex_all(req: BulkIndexRequest, session: Session = Depends(get_session)) -> dict:
    ingest_root = _resolve_ingest_root(req.root)
    ingest_directory(ingest_root, session)
    count = index_existing_photos(
        session,
        backend=vector_backend,
        context=req.context,
        only_missing=False,
    )
    return {"processed": count}


@app.post("/reindex/pending")
def reindex_pending(req: BulkIndexRequest, session: Session = Depends(get_session)) -> dict:
    ingest_root = _resolve_ingest_root(req.root)
    ingest_directory(ingest_root, session)
    count = index_existing_photos(
        session,
        backend=vector_backend,
        context=req.context,
        only_missing=True,
    )
    return {"processed": count}


@app.get("/photos")
def list_photos(
    *,
    limit: int = Query(24, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
) -> dict:
    total = session.scalar(select(func.count()).select_from(PhotoFileRow))
    rows = (
        session.scalars(
            select(PhotoFileRow).order_by(PhotoFileRow.mtime.desc()).offset(offset).limit(limit)
        ).all()
        if total
        else []
    )
    records = [load_photo_record(session, row.id) for row in rows]
    return {
        "photos": [rec.model_dump() for rec in records if rec],
        "total": int(total or 0),
        "limit": limit,
        "offset": offset,
    }


@app.get("/", response_class=HTMLResponse)
def ui() -> HTMLResponse:
    """Serve the built React frontend."""
    if not FRONTEND_DIST.exists():
        raise HTTPException(
            status_code=503,
            detail="Frontend not built. Run `cd frontend && npm install && npm run build`.",
        )
    index_file = FRONTEND_DIST / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found in frontend/dist")
    return FileResponse(index_file)
