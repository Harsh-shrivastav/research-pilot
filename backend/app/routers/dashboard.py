"""Dashboard aggregate stats. Recent activity is derived from document and
workspace timestamps rather than a separate activity-log table, since Phase 1's
schema intentionally stays to the four core tables."""
from sqlalchemy import select
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends

from app.database import get_db
from app.deps import get_current_user
from app.models import Document, ResearchWorkspace
from app.routers.documents import _to_out as document_to_out
from app.schemas import ActivityItem, DashboardStats

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

    return DashboardStats(
        total_documents=len(documents),
        active_workspaces=len(workspaces),
        recent_documents=[document_to_out(d) for d in documents[:5]],
        recent_activity=activity[:8],
    )
