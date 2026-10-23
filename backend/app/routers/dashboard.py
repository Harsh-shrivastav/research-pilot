"""Dashboard aggregate stats. Recent activity is derived from document and
workspace timestamps rather than a separate activity-log table, since Phase 1's
schema intentionally stays to the four core tables."""
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends

from app.database import get_db
from app.deps import get_current_user
from app.models import Claim, ComparisonMatrix, Document, ResearchWorkspace
from app.routers.documents import _to_out as document_to_out
from app.schemas import ActivityItem, DashboardStats

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("", response_model=DashboardStats)
def get_dashboard(db: Session = Depends(get_db)) -> DashboardStats:
    user = get_current_user(db)

    documents = db.scalars(
        select(Document).where(Document.owner_id == user.id).order_by(Document.uploaded_at.desc())
    ).all()
    workspaces = db.scalars(
        select(ResearchWorkspace).where(ResearchWorkspace.owner_id == user.id)
    ).all()

    activity: list[ActivityItem] = []
    for doc in documents:
        activity.append(
            ActivityItem(
                id=f"doc-{doc.id}",
                type="document_uploaded",
                message=f'Uploaded "{doc.original_filename}"',
                timestamp=doc.uploaded_at,
            )
        )
    for ws in workspaces:
        activity.append(
            ActivityItem(
                id=f"ws-{ws.id}",
                type="workspace_created",
                message=f'Created workspace "{ws.name}"',
                timestamp=ws.created_at,
            )
        )
    activity.sort(key=lambda a: a.timestamp, reverse=True)

    extras: dict = {}
    try:
        extras = _extra_stats(db, user.id, documents, workspaces)
    except Exception:
        # The visuals are a bonus: never let them break the core dashboard.
        logger.exception("dashboard extra stats failed")

    return DashboardStats(
        **extras,
        total_documents=len(documents),
        active_workspaces=len(workspaces),
        recent_documents=[document_to_out(d) for d in documents[:5]],
        recent_activity=activity[:8],
    )


def _extra_stats(db: Session, owner_id: str, documents: list, workspaces: list) -> dict:
    status_counts: dict[str, int] = {}
    for doc in documents:
        status_counts[doc.processing_status] = status_counts.get(doc.processing_status, 0) + 1

    today = datetime.now(timezone.utc).date()
    days = [today - timedelta(days=i) for i in range(13, -1, -1)]
    per_day = {d: 0 for d in days}
    for doc in documents:
        uploaded = doc.uploaded_at.date() if doc.uploaded_at else None
        if uploaded in per_day:
            per_day[uploaded] += 1

    total_comparisons = db.scalar(
        select(func.count()).select_from(ComparisonMatrix).where(ComparisonMatrix.owner_id == owner_id)
    ) or 0
    citation_counts = {
        status: count
        for status, count in db.execute(
            select(Claim.support_status, func.count())
            .join(Document, Document.id == Claim.document_id)
            .where(Document.owner_id == owner_id)
            .group_by(Claim.support_status)
        ).all()
    }
    recent = sorted(workspaces, key=lambda w: w.updated_at, reverse=True)[:4]
    return {
        "total_comparisons": total_comparisons,
        "status_counts": status_counts,
        "uploads_by_day": [{"date": d.isoformat(), "count": c} for d, c in per_day.items()],
        "citation_counts": citation_counts,
        "recent_workspaces": [
            {"id": w.id, "name": w.name, "document_count": len(w.document_links), "updated_at": w.updated_at.isoformat()}
            for w in recent
        ],
    }
