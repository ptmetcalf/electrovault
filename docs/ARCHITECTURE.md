# Architecture

This system is modular, agent-friendly, and local-first.

## Component Graph

```mermaid
flowchart LR
A[Raw Photos] --> B[ingest/scanner.py]
B --> C[ingest/exif_reader.py]
C --> D[vision/captioner.py]
D --> E[vision/classifier.py]
C --> F[faces/detector.py]
F --> G[faces/recognizer.py]
D --> H[embedding/text_embedder.py]

C --> I[index/schema + relational]
D --> I
E --> I
G --> I
H --> J[index/vector_backend]

I --> K[events/grouper]
K --> L[events/summarizer]

M[api/http_api] --> N[search/planner]
N --> O[search/executor]
O --> J
```

## Core Data Models

All in core/models.py:
- PhotoFile
- ExifData
- VisionDescription
- Classification
- FaceDetection
- FaceIdentity
- TextEmbedding
- PhotoRecord
- MemoryEvent

## Storage

Primary:
- Postgres
- pgvector extension

Optional:
- Qdrant HTTP service

All access is abstracted behind index/vector_backend.py
