# Agent Coding Rules

These rules govern all coding agents working in this repo.

## 1. General Behavior

- Make small, safe, focused changes.
- Avoid multi-file wide edits unless required.
- NEVER introduce new dependencies without:
  - Updating pyproject.toml
  - Documenting the reason

## 2. Module Boundary Rules

### core/
Shared models, config, and logging.  
Other modules depend on core.  
Core depends on nobody.

### ingest/
Filesystem → PhotoFile extraction.

### vision/
PhotoWithExif → VisionDescription + Classification.

### faces/
PhotoFile → FaceDetections → FaceIdentities.

### embedding/
VisionDescription → TextEmbedding.

### index/
- Postgres / PGVector implementation
- Optional Qdrant implementation
- All DB interactions must happen here
- Other modules NEVER talk to Postgres/Qdrant directly

### events/
Group PhotoRecords into MemoryEvents.

### search/
Natural language → SearchPlan → SearchResults.

### api/
FastAPI only.  
No business logic allowed.

## 3. Data Model Rules

All cross-module data types MUST be defined in core/models.py.

## 4. Storage Rules

- Use vector_backend abstraction.
- Do not import pgvector or qdrant backends directly.
- Do not embed SQL outside index/.

## 5. Code Style

- Use Python typing everywhere
- Write tests for all important functions
- Keep functions short
- Prefer extracting helpers

## 6. If Unsure

Agents must preserve architecture and note conflicts rather than modifying structure.
