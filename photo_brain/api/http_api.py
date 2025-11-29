import os
from io import BytesIO
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from sqlalchemy.orm import Session

from photo_brain.core.env import configure_logging, load_dotenv_if_present
from photo_brain.events import group_events, summarize_events
from photo_brain.index import (
    PgVectorBackend,
    PhotoFileRow,
    assign_face_identity,
    index_photo,
    init_db,
    load_photo_record,
    session_factory,
    set_photo_user_context,
)
from photo_brain.index.schema import FaceDetectionRow
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
