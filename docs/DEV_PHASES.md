# Development Phases

Follow these phases in order. Complete every task in a phase before advancing.

## Phase 0 — Foundation & Guardrails
- Add PRODUCT_BRIEF.md defining capabilities.
- Add DEV_PHASES.md describing the phased build.

## Phase 1 — Core Domain & Storage
- Implement index/schema.py with SQLAlchemy models.
- Add init_db() to create tables and pgvector extension.
- Expand DB tests covering basic CRUD.

## Phase 2 — Ingest Pipeline
- Implement ingest.scan_photos().
- Add tests using temp directories.
- Implement ingest.read_exif() and an ingestion upsert loop.

## Phase 3 — Vector Backend & Basic Search
- Implement PgVectorBackend upsert/search.
- Build minimal search plan and executor wiring to backend.
- Wire /search endpoint; add vector backend tests.

## Phase 4 — Vision & Embeddings Pipeline
- Implement vision.describe_photo().
- Implement vision.classify_photo().
- Implement embedding.embed_description().
- Integrate outputs into indexing workflow.

## Phase 5 — Search Improvements
- Expand query parsing (dates, events, people).
- Apply filters in executor.
- Improve response shape and metadata.

## Phase 6 — Faces & People
- Implement faces.detect_faces() and faces.recognize_faces().
- Add schema tables for faces and identities.
- Integrate people filters into search.

## Phase 7 — Events & Memories
- Implement events.group_events() and events.summarize_events().
- Persist events and expose /events endpoint.

## Phase 8 — UI Layer
- Build operator UI in FastAPI (search box, result grid, event browser).
- Optional: extend to full frontend.

## Phase 9 — Packaging & Deployment
- Add Dockerfile and docker-compose with Postgres (and optional Qdrant).
- Update README with runtime docs and perform cleanup/type passes.
