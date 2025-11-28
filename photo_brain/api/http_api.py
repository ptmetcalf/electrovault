import os
from io import BytesIO
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from sqlalchemy.orm import Session

from photo_brain.core.env import load_dotenv_if_present
from photo_brain.events import group_events, summarize_events
from photo_brain.index import PgVectorBackend, PhotoFileRow, init_db, session_factory
from photo_brain.ingest import ingest_and_index
from photo_brain.search import execute_search, plan_search

app = FastAPI(title="Photo Brain API")

load_dotenv_if_present()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+pysqlite:///./photo_brain.db")
engine = init_db(DATABASE_URL)
SessionLocal = session_factory(engine)
vector_backend = PgVectorBackend()
THUMB_MAX_SIZE = int(os.getenv("THUMB_MAX_SIZE", "320"))
THUMB_CACHE_DIR = Path(os.getenv("THUMB_CACHE_DIR", Path(__file__).resolve().parents[2] / "thumbnails"))

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


@app.post("/reindex")
def reindex_photo(req: ReindexRequest, session: Session = Depends(get_session)) -> dict:
    row = session.get(PhotoFileRow, req.photo_id)
    if not row:
        raise HTTPException(status_code=404, detail="Photo not found")
    index_photo(session, row, context=req.context)
    return {"status": "ok"}


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


@app.on_event("startup")
def auto_ingest() -> None:
    """Automatically ingest and index a directory on startup."""
    target = os.getenv("AUTO_INGEST_DIR")
    if target:
        ingest_root = Path(target)
    else:
        ingest_root = Path(__file__).resolve().parents[2] / "phototest"

    if not ingest_root.exists() or not ingest_root.is_dir():
        # Skip if missing; keeps startup non-fatal.
        return

    force_reindex = os.getenv("AUTO_INGEST_FORCE", "0") == "1"
    with SessionLocal() as session:
        ingest_and_index(
            ingest_root,
            session,
            context=os.getenv("AUTO_INGEST_CONTEXT"),
            skip_if_fresh=not force_reindex,
        )


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
