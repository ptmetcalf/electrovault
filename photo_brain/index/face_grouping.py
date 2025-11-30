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
    PhotoFileRow,
)
from .updates import set_detection_person, upsert_person

logger = logging.getLogger(__name__)
DEFAULT_GROUP_THRESHOLD = float(os.getenv("FACE_GROUP_THRESHOLD", "0.85"))
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


def _cluster_encodings(
    candidates: list[tuple[int, np.ndarray, str | None]],
    threshold: float,
) -> list[tuple[list[int], list[float], list[str | None]]]:
    """
    Greedy centroid clustering. Returns list of (detection_ids, similarities, labels).
    """
    clusters: list[dict[str, object]] = []
    for detection_id, vec, label in candidates:
        best_idx = None
        best_sim = 0.0
        for idx, cluster in enumerate(clusters):
            centroid = cluster["centroid"]  # type: ignore[index]
            sim = _cosine(vec, centroid) if centroid is not None else 0.0
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        if best_idx is not None and best_sim >= threshold:
            cluster = clusters[best_idx]
            cluster["ids"].append(detection_id)  # type: ignore[index]
            cluster["sims"].append(best_sim)  # type: ignore[index]
            cluster["labels"].append(label)  # type: ignore[index]
            centroid_sum: np.ndarray = cluster["sum"]  # type: ignore[index]
            centroid_sum += vec
            new_centroid = _normalize_vec(centroid_sum.tolist())
            if new_centroid is not None:
                cluster["centroid"] = new_centroid  # type: ignore[index]
        else:
            clusters.append(
                {
                    "ids": [detection_id],
                    "sims": [1.0],
                    "labels": [label],
                    "sum": vec.copy(),
                    "centroid": vec,
                }
            )

    results: list[tuple[list[int], list[float], list[str | None]]] = []
    for cluster in clusters:
        ids: list[int] = cluster["ids"]  # type: ignore[assignment]
        if len(ids) < 2:
            continue
        results.append(
            (
                ids,
                cluster["sims"],  # type: ignore[arg-type]
                cluster["labels"],  # type: ignore[arg-type]
            )
        )
    return results


def rebuild_face_group_proposals(
    session: Session,
    *,
    threshold: float | None = None,
    unassigned_only: bool | None = None,
    limit: int | None = None,
) -> list[FaceGroupProposalRow]:
    """Rebuild face grouping proposals using face encodings."""
    effective_threshold = threshold or DEFAULT_GROUP_THRESHOLD
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
    person_labels: dict[int, str] = {}
    for det_row, _, link_row, person_row in rows:
        norm = _normalize_vec(det_row.encoding)
        if norm is None:
            continue
        label = person_row.display_name if person_row else link_row.person_id if link_row else None
        candidates.append((det_row.id, norm, label))
        if label:
            person_labels[det_row.id] = label

    clusters = _cluster_encodings(candidates, effective_threshold)
    logger.info(
        "Face grouping: %d candidates clustered into %d proposals (threshold %.2f)",
        len(candidates),
        len(clusters),
        effective_threshold,
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

    created: list[FaceGroupProposalRow] = []
    for ids, sims, labels in clusters:
        label_counter = Counter(lbl for lbl in labels if lbl)
        suggested = label_counter.most_common(1)[0][0] if label_counter else None
        score_min = round(min(sims), 3) if sims else None
        score_max = round(max(sims), 3) if sims else None
        score_mean = round(float(np.mean(sims)), 3) if sims else None
        proposal = FaceGroupProposalRow(
            status="pending",
            suggested_label=suggested,
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
