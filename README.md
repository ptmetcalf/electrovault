# Photo Brain

Local-first photo understanding and search engine with fully agentic development workflow.

## Features

- Local-only processing
- Modular Python architecture
- PGVector or Qdrant vector search
- Facial recognition pipeline (optional)
- Natural-language photo search
- Memory/Event grouping (trips, holidays, birthdays)

## Repo Structure

See docs/ARCHITECTURE.md for detailed diagrams.

## Local Testing

Run tests with pytest from the repo root:

1) Create a virtualenv with Python 3.12: `python3.12 -m venv .venv && source .venv/bin/activate`
2) Install project + test deps: `pip install -U pip && pip install -e ".[dev]"`
3) Execute the suite: `pytest`

If you previously installed before the dev extra was added, clear the old editable install first: `pip uninstall -y photo-brain && pip install -e ".[dev]"`.

### Common dev commands (Makefile)

- `make install` — create `.venv`, install Python deps, install frontend deps.
- `make lint` — ruff on Python, eslint on frontend.
- `make format` — black + ruff --fix on Python.
- `make test` / `make test-cov` — run pytest (optionally with coverage).
- `make api` — start the FastAPI server (`uvicorn photo_brain.api.http_api:app`).
- `make frontend-dev` / `make frontend-build` — run Vite dev server or build frontend.
- `make ingest-sample` — ingest the `phototest/` sample set into the current DB.

## Running the API

- Default to SQLite by omitting `DATABASE_URL`, or point at Postgres/pgvector:  
  `DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/photo_brain`
- Launch the operator UI + API:  
  `uvicorn photo_brain.api.http_api:app --host 0.0.0.0 --port 8000`
- Open `http://localhost:8000` for the React UI or hit `/events` for event summaries. The backend auto-ingests the `phototest/` directory by default; set `AUTO_INGEST_DIR=/absolute/path` to change it.
  - Auto-ingest skips unchanged photos (same mtime/context); set `AUTO_INGEST_FORCE=1` to force re-index on startup. Use `AUTO_INGEST_CONTEXT` to inject user context during auto-ingest.
  - Thumbnails: `GET /thumb/{photo_id}` returns a JPEG thumbnail (max size controlled by `THUMB_MAX_SIZE`, default 320).

## Using Local Vision + Embedding Models (Ollama/LM Studio)

Out of the box, captions/classifications/embeddings use deterministic fallbacks. To power them with a local model:

1) Install and run Ollama (or LM Studio) locally.  
   - Example: `ollama pull llava:latest` (vision) and `ollama pull nomic-embed-text` (embeddings).
2) Set environment variables when running ingest/API:
   - `OLLAMA_BASE_URL` (optional, default `http://localhost:11434`)
   - `OLLAMA_VISION_MODEL=llava:latest`
   - `OLLAMA_EMBED_MODEL=nomic-embed-text`
3) Ingest photos (see CLI below) and start the API/UI. The vision/tagging pipeline will call your local model; embeddings will use the embed model. If the model call fails, the code falls back to deterministic heuristics so tests/local runs remain stable.

### .env convenience

- Copy the sample: `cp .env.example .env` and fill in values (DB + Ollama models).  
- The API automatically loads `.env` on startup; for scripts, call `photo_brain.core.env.load_dotenv_if_present()` or run commands via `python -m dotenv run -- ...`.

### Location resolution (optional)

- Reverse geocode EXIF GPS coordinates via LocationIQ (OpenStreetMap data) and cache results for reuse.
- Enable with `LOCATION_RESOLUTION_ENABLED=1` and supply `LOCATIONIQ_API_KEY`; tune cache radii with `LOCATION_USER_RADIUS_METERS` and `LOCATION_CACHE_RADIUS_METERS`.
- User-defined locations take priority and can be created/assigned via `POST /locations` (name, latitude, longitude, radius_meters, optional photo_id).

## CLI Ingest (local folder)

```bash
python - <<'PY'
from photo_brain.index import init_db, session_factory
from photo_brain.ingest import ingest_and_index

DB_URL = "sqlite+pysqlite:///./photo_brain.db"  # or your DATABASE_URL
PHOTOS_DIR = "/absolute/path/to/your/photos"

engine = init_db(DB_URL)
SessionLocal = session_factory(engine)
with SessionLocal() as session:
    ingest_and_index(PHOTOS_DIR, session)
print("Ingest complete")
PY
```

After ingest, start the API (`uvicorn photo_brain.api.http_api:app --host 0.0.0.0 --port 8000`) and search via `http://localhost:8000`. The UI is built from the `frontend/` React app (`cd frontend && npm install && npm run build`). On startup, the API will ingest `phototest/` (or `AUTO_INGEST_DIR` if set).

## Docker

- Build (run from repo root): `docker build -t photo-brain .`
  - If you get a buildx error, install the Docker buildx plugin (recommended) or temporarily fall back with `DOCKER_BUILDKIT=0 docker build -t photo-brain .`.
- Run: `docker run --rm -p 8000:8000 photo-brain`

## Docker Compose (with Postgres + optional Qdrant)

`docker-compose up --build`

This starts:
- `db`: Postgres with pgvector
- `app`: Photo Brain API/UI on port 8000
- `qdrant`: Optional vector store on 6333 (not required by defaults)

## Local Development Workflow (fast iteration)

1) **Prep env**  
   - `python -m venv .venv && source .venv/bin/activate`  
   - `pip install -e ".[dev]"`  
   - `cp .env.example .env` and adjust (SQLite is fine for dev).

2) **Rapid ingest loop**  
   - Run the ingest script when you add/change photos:  
     ```bash
     python - <<'PY'
     from photo_brain.index import init_db, session_factory
     from photo_brain.ingest import ingest_and_index
     from photo_brain.core.env import load_dotenv_if_present

     load_dotenv_if_present()
     engine = init_db()
     SessionLocal = session_factory(engine)
     with SessionLocal() as session:
         ingest_and_index("/absolute/path/to/photos", session)
     print("Ingest complete")
     PY
     ```

3) **Run API/UI with reload**  
   - `uvicorn photo_brain.api.http_api:app --host 0.0.0.0 --port 8000 --reload`  
   - Open `http://localhost:8000` to test search/events. Reload picks up code edits.

4) **Use local models when needed**  
   - Set `OLLAMA_VISION_MODEL` / `OLLAMA_EMBED_MODEL` in `.env` (see above).  
   - Keep models running locally (e.g., `ollama serve`) during dev; falls back to deterministic stubs if unavailable.

5) **Run tests quickly**  
   - Full suite: `pytest`  
   - Targeted file: `pytest tests/test_search.py -q`

6) **Hot code editing tips**  
   - Keep `uvicorn --reload` running; re-run ingest if you change ingest/index logic or add new photos.  
   - SQLite (`photo_brain.db`) is convenient for dev; switch to Postgres via `DATABASE_URL` when you need vector-native behavior.

### CLI ingest script

- Run the helper (loads `.env` automatically):  
  `python scripts/ingest.py /absolute/path/to/photos`
- Override DB inline if desired:  
  `DATABASE_URL=sqlite+pysqlite:///./photo_brain.db python scripts/ingest.py ~/Pictures`
- Note: the path must exist on the local filesystem. If you have a network share (SMB/UNC like `//host/share`), mount it first to a local path (e.g., `/mnt/photos`) and point the script there.
