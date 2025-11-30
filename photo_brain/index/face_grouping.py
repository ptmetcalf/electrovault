from __future__ import annotations

import logging
import os
from collections import Counter, defaultdict
from typing import Iterable, List, Optional, Tuple

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from photo_brain.core.models import FaceDetection, FaceGroupProposal, FaceIdentity, FacePreview, PhotoFile

from .schema import (
    FaceDetectionRow,
    FaceGroupProposalMemberRow,
    FaceGroupProposalRow,
    FacePersonLinkRow,
    PersonRow,
    PersonStatsRow,
    PhotoFileRow,
)
from .updates import set_detection_person, upsert_person

logger = logging.getLogger(__name__)
DEFAULT_GROUP_EPS = float(os.getenv("FACE_GROUP_EPS", os.getenv("FACE_GROUP_THRESHOLD", "0.85")))
DEFAULT_GROUP_MIN_SAMPLES = int(os.getenv("FACE_GROUP_MIN_SAMPLES", "2"))
DEFAULT_GROUP_MIN_CONF = float(os.getenv("FACE_GROUP_MIN_CONF", "0.4"))
DEFAULT_GROUP_MIN_SIZE = int(os.getenv("FACE_GROUP_MIN_SIZE", "8"))
# By default group unassigned faces only; can be overridden.
DEFAULT_UNASSIGNED_ONLY = os.getenv("FACE_GROUP_UNASSIGNED_ONLY", "true").lower() in {
    "1",
    "true",
    "yes",
}
MAX_GROUP_BATCH = int(os.getenv("FACE_GROUP_BATCH", "800"))


def _normalize_vec(vec: list[float] | None) -> Optional[np.ndarray]:
    if not vec:
        return None
    arr = np.array(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return None
    return arr / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _is_detection_eligible(det: FaceDetectionRow) -> bool:
    width = det.bbox_x2 - det.bbox_x1
    height = det.bbox_y2 - det.bbox_y1
    if width < DEFAULT_GROUP_MIN_SIZE or height < DEFAULT_GROUP_MIN_SIZE:
        return False
    if det.confidence < DEFAULT_GROUP_MIN_CONF:
        return False
    return True


def _build_photo_model(row: PhotoFileRow) -> PhotoFile:
    return PhotoFile(
        id=row.id,
        path=row.path,
        sha256=row.sha256,
        size_bytes=row.size_bytes,
        mtime=row.mtime,
    )


def _build_detection_model(row: FaceDetectionRow) -> FaceDetection:
    return FaceDetection(
        id=row.id,
        photo_id=row.photo_id,
        bbox=(row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2),
        confidence=row.confidence,
        encoding=row.encoding,
        created_at=row.created_at,
    )


def _build_face_preview(
    detection_row: FaceDetectionRow,
    photo_row: PhotoFileRow,
    person_row: PersonRow | None,
    link_row: FacePersonLinkRow | None,
) -> FacePreview:
    identity = None
    if person_row or link_row:
        label = person_row.display_name if person_row else link_row.person_id if link_row else None
        person_id = person_row.id if person_row else link_row.person_id if link_row else None
        identity = (
            FaceIdentity(
                detection_id=detection_row.id,
                person_id=person_id or "",
                label=label or person_id,
            )
            if person_id
            else None
        )
    return FacePreview(
        detection=_build_detection_model(detection_row),
        identity=identity,
        photo=_build_photo_model(photo_row),
    )


def _dbscan_clusters(
    candidates: list[tuple[int, np.ndarray, str | None]],
    eps: float,
    min_samples: int,
) -> list[tuple[list[int], list[float], list[str | None]]]:
    """
    Simple DBSCAN-style clustering on cosine similarity.
    Returns (ids, sims_to_centroid, labels) per cluster.
    """
    if not candidates or min_samples < 2:
        return []

    ids = [c[0] for c in candidates]
    vecs = [c[1] for c in candidates]
    labels = [c[2] for c in candidates]
    n = len(vecs)
    visited = [False] * n
    clusters: list[list[int]] = []

    # Precompute similarity matrix
    sim_matrix = np.clip(np.matmul(np.stack(vecs), np.stack(vecs).T), -1.0, 1.0)

    def neighbors(idx: int) -> list[int]:
        return [j for j, score in enumerate(sim_matrix[idx]) if score >= eps]

    for idx in range(n):
        if visited[idx]:
            continue
        visited[idx] = True
        nbrs = neighbors(idx)
        if len(nbrs) < min_samples:
            continue  # noise
        cluster = []
        queue = nbrs.copy()
        while queue:
            j = queue.pop()
            if not visited[j]:
                visited[j] = True
                j_neighbors = neighbors(j)
                if len(j_neighbors) >= min_samples:
                    queue.extend(j_neighbors)
            if j not in cluster:
                cluster.append(j)
        if cluster:
            clusters.append(cluster)

    results: list[tuple[list[int], list[float], list[str | None]]] = []
    for cluster_indices in clusters:
        det_ids = [ids[i] for i in cluster_indices]
        if len(det_ids) < 2:
            continue
        cluster_vecs = [vecs[i] for i in cluster_indices]
        centroid = _normalize_vec(np.mean(np.stack(cluster_vecs), axis=0).tolist())
        sims = []
        if centroid is not None:
            sims = [_cosine(vec, centroid) for vec in cluster_vecs]
        else:
            sims = [1.0 for _ in cluster_vecs]
        cluster_labels = [labels[i] for i in cluster_indices]
        results.append((det_ids, sims, cluster_labels))
    return results


def rebuild_face_group_proposals(
    session: Session,
    *,
    threshold: float | None = None,
    unassigned_only: bool | None = None,
    limit: int | None = None,
) -> list[FaceGroupProposalRow]:
    """Rebuild face grouping proposals using face encodings."""
    effective_eps = threshold or DEFAULT_GROUP_EPS
    min_samples = max(2, DEFAULT_GROUP_MIN_SAMPLES)
    include_unassigned_only = DEFAULT_UNASSIGNED_ONLY if unassigned_only is None else unassigned_only
    max_candidates = limit or MAX_GROUP_BATCH
    stmt = (
        select(FaceDetectionRow, PhotoFileRow, FacePersonLinkRow, PersonRow)
        .join(PhotoFileRow, FaceDetectionRow.photo_id == PhotoFileRow.id)
        .outerjoin(FacePersonLinkRow, FacePersonLinkRow.detection_id == FaceDetectionRow.id)
        .outerjoin(PersonRow, PersonRow.id == FacePersonLinkRow.person_id)
        .where(FaceDetectionRow.encoding.is_not(None))
        .limit(max_candidates)
    )
    if include_unassigned_only:
        stmt = stmt.where(FacePersonLinkRow.person_id.is_(None))

    rows = session.execute(stmt).all()
    candidates: list[tuple[int, np.ndarray, str | None]] = []
    vec_map: dict[int, np.ndarray] = {}
    for det_row, _, link_row, person_row in rows:
        if not _is_detection_eligible(det_row):
            continue
        norm = _normalize_vec(det_row.encoding)
        if norm is None:
            continue
        label = person_row.display_name if person_row else link_row.person_id if link_row else None
        candidates.append((det_row.id, norm, label))
        vec_map[det_row.id] = norm

    clusters = _dbscan_clusters(candidates, effective_eps, min_samples=min_samples)
    logger.info(
        "Face grouping: %d candidates clustered into %d proposals (eps %.2f, min_samples=%d)",
        len(candidates),
        len(clusters),
        effective_eps,
        min_samples,
    )

    pending_ids = select(FaceGroupProposalRow.id).where(FaceGroupProposalRow.status != "accepted")
    session.execute(
        delete(FaceGroupProposalMemberRow).where(
            FaceGroupProposalMemberRow.proposal_id.in_(pending_ids)
        )
    )
    session.execute(delete(FaceGroupProposalRow).where(FaceGroupProposalRow.status != "accepted"))
    if not clusters:
        session.flush()
        return []
    session.flush()

    # Load confirmed person stats for suggestions
    stats_rows = session.execute(
        select(PersonStatsRow, PersonRow)
        .join(PersonRow, PersonRow.id == PersonStatsRow.person_id)
        .where(
            PersonRow.is_user_confirmed.is_(True),
            PersonRow.auto_assign_enabled.is_(True),
            PersonStatsRow.embedding_centroid.is_not(None),
        )
    ).all()
    stats: list[tuple[str, str, np.ndarray]] = []
    for stats_row, person_row in stats_rows:
        norm = _normalize_vec(stats_row.embedding_centroid)
        if norm is not None:
            stats.append((person_row.id, person_row.display_name or person_row.id, norm))

    created: list[FaceGroupProposalRow] = []
    for ids, sims, labels in clusters:
        label_counter = Counter(lbl for lbl in labels if lbl)
        suggested = label_counter.most_common(1)[0][0] if label_counter else None
        score_min = round(min(sims), 3) if sims else None
        score_max = round(max(sims), 3) if sims else None
        score_mean = round(float(np.mean(sims)), 3) if sims else None
        suggested_person_id = None
        if stats and ids:
            # Compute cluster centroid for suggestion matching
            vecs = [vec_map[i] for i in ids if i in vec_map]
            if vecs:
                centroid = np.mean(np.stack(vecs), axis=0)
                norm = np.linalg.norm(centroid)
                if norm:
                    centroid = centroid / norm
                    best_id = None
                    best_label = None
                    best_score = 0.0
                    second = 0.0
                    for pid, pname, pvec in stats:
                        score = float(np.clip(np.dot(centroid, pvec), -1.0, 1.0))
                        if score > best_score:
                            second = best_score
                            best_score = score
                            best_id = pid
                            best_label = pname
                        elif score > second:
                            second = score
                    if best_id and best_score >= effective_eps and (best_score - second) >= 0.03:
                        suggested_person_id = best_id
                        if not suggested:
                            suggested = best_label

        proposal = FaceGroupProposalRow(
            status="pending",
            suggested_label=suggested,
            suggested_person_id=suggested_person_id,
            score_min=score_min,
            score_max=score_max,
            score_mean=score_mean,
            size=len(ids),
        )
        session.add(proposal)
        session.flush()
        for detection_id, sim in zip(ids, sims):
            session.add(
                FaceGroupProposalMemberRow(
                    proposal_id=proposal.id,
                    detection_id=detection_id,
                    similarity=round(sim, 3),
                )
            )
        created.append(proposal)
    session.flush()
    return created


def list_face_group_proposals(
    session: Session,
    *,
    status: str | None = "pending",
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[FaceGroupProposal], int]:
    base = select(FaceGroupProposalRow)
    if status:
        base = base.where(FaceGroupProposalRow.status == status)
    total = session.scalar(select(func.count()).select_from(base.subquery())) or 0

    proposal_rows = (
        session.scalars(
            base.order_by(FaceGroupProposalRow.created_at.desc()).limit(limit).offset(offset)
        ).all()
        if total
        else []
    )
    if not proposal_rows:
        return [], int(total)

    proposal_ids = [p.id for p in proposal_rows]
    member_rows = session.scalars(
        select(FaceGroupProposalMemberRow).where(
            FaceGroupProposalMemberRow.proposal_id.in_(proposal_ids)
        )
    ).all()
    by_proposal: dict[str, list[FaceGroupProposalMemberRow]] = defaultdict(list)
    detection_ids: set[int] = set()
    for member in member_rows:
        by_proposal[member.proposal_id].append(member)
        detection_ids.add(member.detection_id)

    if not detection_ids:
        return [], int(total)

    preview_rows = session.execute(
        select(FaceDetectionRow, PhotoFileRow, FacePersonLinkRow, PersonRow)
        .join(PhotoFileRow, FaceDetectionRow.photo_id == PhotoFileRow.id)
        .outerjoin(FacePersonLinkRow, FacePersonLinkRow.detection_id == FaceDetectionRow.id)
        .outerjoin(PersonRow, PersonRow.id == FacePersonLinkRow.person_id)
        .where(FaceDetectionRow.id.in_(detection_ids))
    ).all()
    preview_map: dict[int, FacePreview] = {}
    for det_row, photo_row, link_row, person_row in preview_rows:
        preview_map[det_row.id] = _build_face_preview(det_row, photo_row, person_row, link_row)

    proposals: list[FaceGroupProposal] = []
    for row in proposal_rows:
        members = by_proposal.get(row.id, [])
        previews = [preview_map[m.detection_id] for m in members if m.detection_id in preview_map]
        proposals.append(
            FaceGroupProposal(
                id=row.id,
                status=row.status,
                suggested_label=row.suggested_label,
                suggested_person_id=row.suggested_person_id,
                score_min=row.score_min,
                score_max=row.score_max,
                score_mean=row.score_mean,
                size=row.size,
                members=previews,
                created_at=row.created_at,
            )
        )
    return proposals, int(total)


def accept_face_group(
    session: Session,
    proposal_id: str,
    *,
    target_person_id: str | None = None,
    target_label: str | None = None,
) -> PersonRow:
    proposal = session.get(FaceGroupProposalRow, proposal_id)
    if not proposal:
        raise ValueError("Proposal not found")
    member_rows = session.scalars(
        select(FaceGroupProposalMemberRow).where(FaceGroupProposalMemberRow.proposal_id == proposal_id)
    ).all()
    detection_ids = [m.detection_id for m in member_rows]
    if not detection_ids:
        raise ValueError("Proposal has no members")

    label = (target_label or proposal.suggested_label or f"person-{proposal_id[:8]}").strip()
    person = upsert_person(session, display_name=label, person_id=target_person_id)

    for member in member_rows:
        set_detection_person(
            session,
            detection_id=member.detection_id,
            person=person,
            confidence=member.similarity,
        )
    proposal.status = "accepted"
    session.flush()
    return person


def reject_face_group(session: Session, proposal_id: str) -> FaceGroupProposalRow:
    proposal = session.get(FaceGroupProposalRow, proposal_id)
    if not proposal:
        raise ValueError("Proposal not found")
    proposal.status = "rejected"
    session.flush()
    return proposal
