# Face Recognition Workflow (Proposed)

This document describes the intended end-to-end behavior for face assignment, grouping, and suggestions. It prioritizes safe auto-assigns, easy cleanup, and explainability.

## Goals
- Minimize bad auto-assigns.
- Make bulk cleanup easy and safe.
- Keep the model/UX explainable.

## Data Shapes (aligned to current models)
- **FaceIdentity**: add/use `auto_assigned` and `confidence` to track provenance.
- **Person**: treat people with user-confirmed faces as eligible for auto-assign; others are “pseudo”.
- **Centroids**: maintain a per-person centroid (mean of embeddings) plus counts; prefer trimmed means in a background job after assignments.

Suggested auxiliary cache (future):
- PersonStats: `{person_id, embedding_centroid, embedding_count, last_seen_at, auto_assign_enabled, is_user_confirmed, cluster_spread?}`.

## Matching & Thresholds
Assume cosine similarity in [0,1].
- `T_auto` (auto-assign): 0.92–0.95.
- `T_suggest` (proposal): 0.82–0.88.
- `T_conflict_gap`: 0.03–0.05 (margin between best and second best).

New face pipeline:
1) Compute embedding.
2) Compare to centroids of user-confirmed persons (and `auto_assign_enabled=true`).
3) Take top K (3–5) matches; let `s1` best, `s2` second.
4) Auto-assign only if:
   - person is confirmed,
   - `s1 >= T_auto`,
   - `s1 - s2 >= T_conflict_gap`.
5) Otherwise, if `s1 >= T_suggest`, create a suggestion referencing the best person; if multiple persons exceed `T_suggest`, surface a conflicting suggestion for user choice.
6) If below `T_suggest`, leave unassigned; optionally cluster unknowns for “Unknown” suggestions.

## Person Lifecycle
- Eligibility for auto-assign: user-confirmed people only (e.g., at least one manual confirmation or >=3 confirmed faces).
- Pseudo/temporary clusters are not eligible for auto-assign; they can be used for grouping UI only until confirmed.
- On new assignment, append embedding; recompute centroid asynchronously to stay robust to pose/lighting.

## Rebuild / Suggestions Job
- Inputs: `include_assigned` (default false), `include_pseudo_persons` (default false), `max_suggestions_per_run`, thresholds.
- Candidates: all unassigned faces (default); optionally assigned for merge/cleanup.
- Actions:
  - Auto-corrections if a face now clears `T_auto`.
  - Suggestions when `s1 >= T_suggest`.
  - Conflicting suggestions if multiple persons exceed `T_suggest`.
  - Merge proposals if a person’s faces are closer to another centroid than their own.

## Grouping & UI
- Show clusters with similarity range (e.g., “0.91–0.96”), representative thumbnails, and actions:
  - Unknown cluster: “Name this person”, “Assign to existing…”, “Ignore”.
  - Suggested known person: primary “Confirm as <name>”, secondary “Assign to another…”, “Nope”.
- Per-face overrides inside a cluster (uncheck or reassign single faces).
- Provenance badges: `manual`, `auto`, `batch_confirm`; “undo” for recent auto/batch actions.

## Config Controls
- Modes: conservative | balanced | aggressive presets.
- Advanced sliders: `T_auto`, `T_suggest`, `T_conflict_gap`, `max_suggestions_per_run`, `max_suggestions_per_person`.

## Implementation Notes for This App
- Store all embeddings; keep fast centroids/stats cache for matching.
- Auto-assign only to confirmed people above `T_auto` and `T_conflict_gap`; skip pseudo IDs.
- Suggestions: default unassigned-only; provide an option to include assigned for merge cleanup.
- Frontend: Faces view should show similarity ranges, suggested person, provenance badges, per-face overrides, and autocomplete for names.

## Current Implementation Status (April 2025)
- Auto-assign: enabled only to existing named persons with a high threshold (`FACE_ASSIGN_THRESHOLD` default 0.93) and conflict gap (`FACE_ASSIGN_CONFLICT_GAP` default 0.04); uses centroids built from stored encodings.
- Suggestions: threshold defaults to `FACE_GROUP_THRESHOLD` (0.85) and are rebuilt via `/face_groups/rebuild`; UI now lets you set threshold and include assigned faces before rebuilding.
- Provenance: identities track `auto_assigned`, and UI shows an “Auto” badge on face cards.
- Gaps vs spec: no explicit `is_user_confirmed` flag or person stats cache yet; conflicting suggestions/merge proposals remain future work.
