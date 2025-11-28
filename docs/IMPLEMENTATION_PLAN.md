# Photo Brain – Implementation Plan

A complete, pragmatic roadmap from scaffold to fully functional system.

## 0. Foundation & Guardrails

This project already includes:
- AGENTS.md
- docs/AGENT_RULES.md
- docs/ARCHITECTURE.md
- docs/CODING_TASK_TEMPLATE.md
- Initial scaffold for modules

Before further development:
- [ ] Add docs/PRODUCT_BRIEF.md defining final capabilities
- [ ] Add docs/DEV_PHASES.md defining phased build approach

## 1. Core Domain & Storage

### Objectives
- Solidify Pydantic models
- Stabilize Postgres schema
- Guarantee DB connectivity flow

### Tasks
- [ ] Implement index/schema.py with SQLAlchemy models
- [ ] Add init_db() for table creation & pgvector extension
- [ ] Expand DB tests verifying basic CRUD

## 2. Ingest Pipeline

### Objectives
Convert filesystem photos into PhotoFile + ExifData rows in DB.

### Tasks
- [ ] Implement ingest.scan_photos()
- [ ] Add tests for scanning via temp directories
- [ ] Implement ingest.read_exif() using Pillow or exifread
- [ ] Implement ingestion loop to upsert into DB

## 3. Vector Backend & Basic Search

### Objectives
Enable real vector-backed search via Postgres pgvector.

### Tasks
- [ ] Implement PgVectorBackend upsert/search
- [ ] Build minimal search plan → executor logic
- [ ] Wire /search to backend
- [ ] Add vector backend tests

## 4. Vision & Embeddings Pipeline

### Objectives
Generate captions, classifications, and semantic embeddings.

### Tasks
- [ ] Implement vision.describe_photo()
- [ ] Implement vision.classify_photo()
- [ ] Implement embedding.embed_description()
- [ ] Integrate into indexing workflow

## 5. Search Improvements

### Objectives
Natural language → SearchPlan → real results with filters.

### Tasks
- [ ] Expand query parsing (dates, events, people)
- [ ] Apply filters in executor
- [ ] Improve response shape and metadata

## 6. Faces & People

### Objectives
Detect faces, cluster embeddings, and allow person-based search.

### Tasks
- [ ] Implement faces.detect_faces()
- [ ] Implement faces.recognize_faces()
- [ ] Add schema tables for faces & identities
- [ ] Integrate people filters into search

## 7. Events & Memories

### Objectives
Group photos into trips, celebrations, and meaningful collections.

### Tasks
- [ ] Implement events.group_events()
- [ ] Implement events.summarize_events()
- [ ] Add DB persistence and /events endpoint

## 8. UI Layer

### Layer 1: Operator UI
Simple HTML served via FastAPI:
- Search box
- Result grid
- Event browser

### Layer 2: Optional Full UI
Dedicated frontend (React/Next).

## 9. Packaging & Deployment

### Objectives
Make the system portable, reproducible, and easy to run anywhere.

### Tasks
- [ ] Dockerfile for app
- [ ] docker-compose.yml with Postgres + optional Qdrant
- [ ] Runtime docs in README
- [ ] Cleanup and type/lint passes
