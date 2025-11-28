# Photo Brain — Product Brief

## Purpose
Photo Brain is a local-first photo intelligence tool that ingests personal photo libraries, extracts metadata, and makes them searchable through captions, classifications, faces, and vector embeddings. It keeps analysis and storage on the user's stack (Postgres + pgvector) to preserve ownership and privacy.

## Target Users
- Individuals or teams with large personal or shared photo folders.
- Power users who want fast semantic search without uploading assets to third-party services.
- Operators comfortable running a local Postgres instance.

## Core Capabilities
- **Ingest**: Scan directories, register photo files, and capture EXIF metadata.
- **Understand**: Generate vision descriptions, classifications, face detections/identities, and text embeddings using the defined module pipeline.
- **Index**: Persist structured photo records and vectors in Postgres/pgvector via the index layer.
- **Search**: Natural-language queries translate into search plans executed against the vector backend; results return relevant photo records with context.
- **Events** (later phase): Group related photos into meaningful memory events and summaries.

## Out of Scope (Now)
- Cloud storage, sync, or remote inference endpoints.
- Direct database access from non-index modules (must use vector backend abstraction).
- New dependencies beyond the approved stack without explicit approval.
- UI beyond the planned operator FastAPI layer until later phases.

## Technical Guardrails
- Core data contracts live in `photo_brain/core/models.py` and drive all cross-module types.
- Storage is Postgres with pgvector; all DB I/O flows through the `index` module and `vector_backend`.
- Architecture boundaries in `docs/ARCHITECTURE.md` must be honored; no business logic in API layer.

## Success Criteria
- End users can ingest a folder, generate structured metadata, and retrieve photos via semantic and filtered search, all locally.
- Pipelines remain modular so each stage (ingest, vision, embedding, index, search) can be developed and tested independently.
