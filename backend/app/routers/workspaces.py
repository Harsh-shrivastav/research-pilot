"""Research workspace endpoints — Phase 6A.

Complete workspace API:
  - create, list, get
  - add document (with duplicate + max 5 + ownership checks)
  - remove document
  - workspace chat (SSE streaming, same architecture as single-doc chat)
  - workspace conversations
"""
import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import SessionLocal, get_db
from app.deps import get_current_user
from app.models import (
    ChatConversation,
    ChatMessage,
    Document,
    ResearchWorkspace,
    WorkspaceDocument,
)
from app.routers.chat import _message_out
from app.routers.documents import _to_out as document_to_out
from app.schemas import (
    ChatMessageOut,
    WorkspaceAddDocument,
    WorkspaceChatAskIn,
    WorkspaceChatConversationOut,
    WorkspaceCreate,
    WorkspaceDetailOut,
    WorkspaceOut,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/workspaces", tags=["workspaces"])
settings = get_settings()

MAX_WORKSPACE_DOCUMENTS = 5


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _workspace_to_out(ws: ResearchWorkspace) -> WorkspaceOut:
    return WorkspaceOut(
        id=ws.id,
        name=ws.name,
        description=ws.description,
        created_at=ws.created_at,
        updated_at=ws.updated_at,
        document_count=len(ws.document_links),
    )


def _get_owned_workspace(db: Session, workspace_id: str) -> ResearchWorkspace:
    user = get_current_user(db)
    ws = db.scalar(
        select(ResearchWorkspace).where(
            ResearchWorkspace.id == workspace_id,
            ResearchWorkspace.owner_id == user.id,
        )
    )
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    return ws


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Workspace CRUD
# ---------------------------------------------------------------------------


@router.get("", response_model=list[WorkspaceOut])
def list_workspaces(db: Session = Depends(get_db)) -> list[WorkspaceOut]:
    user = get_current_user(db)
    workspaces = db.scalars(
        select(ResearchWorkspace)
        .where(ResearchWorkspace.owner_id == user.id)
        .order_by(ResearchWorkspace.updated_at.desc())
    ).all()
    return [_workspace_to_out(w) for w in workspaces]


@router.post("", response_model=WorkspaceOut)
def create_workspace(payload: WorkspaceCreate, db: Session = Depends(get_db)) -> WorkspaceOut:
    user = get_current_user(db)
    ws = ResearchWorkspace(name=payload.name, description=payload.description, owner_id=user.id)
    db.add(ws)
    db.flush()

    if payload.document_id:
        doc = db.scalar(
            select(Document).where(Document.id == payload.document_id, Document.owner_id == user.id)
        )
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found.")
        db.add(WorkspaceDocument(workspace_id=ws.id, document_id=doc.id))

    db.commit()
    db.refresh(ws)
    return _workspace_to_out(ws)


@router.get("/by-document/{document_id}", response_model=WorkspaceOut | None)
def get_workspace_by_document(document_id: str, db: Session = Depends(get_db)):
    user = get_current_user(db)
    link = db.scalar(
        select(WorkspaceDocument)
        .join(ResearchWorkspace)
        .where(
            WorkspaceDocument.document_id == document_id,
            ResearchWorkspace.owner_id == user.id,
        )
    )
    if link is None:
        return None
    return _workspace_to_out(link.workspace)


@router.get("/{workspace_id}", response_model=WorkspaceDetailOut)
def get_workspace(workspace_id: str, db: Session = Depends(get_db)) -> WorkspaceDetailOut:
    ws = _get_owned_workspace(db, workspace_id)
    docs = [document_to_out(link.document) for link in ws.document_links]
    base = _workspace_to_out(ws)
    return WorkspaceDetailOut(**base.model_dump(), documents=docs)


# ---------------------------------------------------------------------------
# Document management
# ---------------------------------------------------------------------------


@router.post("/{workspace_id}/documents", response_model=WorkspaceDetailOut)
def add_document_to_workspace(
    workspace_id: str, payload: WorkspaceAddDocument, db: Session = Depends(get_db)
) -> WorkspaceDetailOut:
    """Add a document to a workspace.

    Prevents:
      - documents not owned by the user
      - duplicate documents (same document already in workspace)
      - more than MAX_WORKSPACE_DOCUMENTS documents
    """
    user = get_current_user(db)
    ws = _get_owned_workspace(db, workspace_id)

    # Verify document exists and belongs to user
    doc = db.scalar(
        select(Document).where(
            Document.id == payload.document_id, Document.owner_id == user.id
        )
    )
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Check duplicate
    existing = db.scalar(
        select(WorkspaceDocument).where(
            WorkspaceDocument.workspace_id == workspace_id,
            WorkspaceDocument.document_id == payload.document_id,
        )
    )
    if existing is not None:
        raise HTTPException(
            status_code=409, detail="This document is already in the workspace."
        )

    # Check max documents
    current_count = db.scalar(
        select(func.count(WorkspaceDocument.id)).where(
            WorkspaceDocument.workspace_id == workspace_id
        )
    )
    if current_count >= MAX_WORKSPACE_DOCUMENTS:
        raise HTTPException(
            status_code=409,
            detail=f"A workspace may contain at most {MAX_WORKSPACE_DOCUMENTS} documents.",
        )

    db.add(WorkspaceDocument(workspace_id=ws.id, document_id=doc.id))
    db.commit()
    db.refresh(ws)

    docs = [document_to_out(link.document) for link in ws.document_links]
    base = _workspace_to_out(ws)
    return WorkspaceDetailOut(**base.model_dump(), documents=docs)


@router.delete("/{workspace_id}/documents/{document_id}", status_code=200, response_model=WorkspaceDetailOut)
def remove_document_from_workspace(
    workspace_id: str, document_id: str, db: Session = Depends(get_db)
) -> WorkspaceDetailOut:
    """Remove a document from a workspace."""
    ws = _get_owned_workspace(db, workspace_id)

    link = db.scalar(
        select(WorkspaceDocument).where(
            WorkspaceDocument.workspace_id == workspace_id,
            WorkspaceDocument.document_id == document_id,
        )
    )
    if link is None:
        raise HTTPException(
            status_code=404, detail="Document not found in this workspace."
        )

    db.delete(link)
    db.commit()
    db.refresh(ws)

    docs = [document_to_out(link.document) for link in ws.document_links]
    base = _workspace_to_out(ws)
    return WorkspaceDetailOut(**base.model_dump(), documents=docs)


# ---------------------------------------------------------------------------
# Workspace Chat
# ---------------------------------------------------------------------------


def _get_or_create_workspace_conversation(
    db: Session, workspace: ResearchWorkspace, owner_id: str, conversation_id: str | None
) -> ChatConversation:
    """Get or create a conversation for workspace chat."""
    if conversation_id:
        conversation = db.scalar(
            select(ChatConversation).where(
                ChatConversation.id == conversation_id,
                ChatConversation.workspace_id == workspace.id,
            )
        )
        if conversation is not None:
            return conversation
    conversation = ChatConversation(
        workspace_id=workspace.id,
        owner_id=owner_id,
    )
    db.add(conversation)
    db.flush()
    return conversation


@router.get("/{workspace_id}/chat/conversations", response_model=list[WorkspaceChatConversationOut])
def list_workspace_conversations(
    workspace_id: str, db: Session = Depends(get_db)
) -> list[WorkspaceChatConversationOut]:
    ws = _get_owned_workspace(db, workspace_id)
    conversations = db.scalars(
        select(ChatConversation)
        .where(ChatConversation.workspace_id == ws.id)
        .order_by(ChatConversation.updated_at.desc())
    ).all()
    return [
        WorkspaceChatConversationOut(
            id=c.id,
            workspace_id=ws.id,
            title=c.title,
            created_at=c.created_at,
            updated_at=c.updated_at,
            message_count=len(c.messages),
        )
        for c in conversations
    ]


@router.post("/{workspace_id}/chat/ask")
async def workspace_ask(workspace_id: str, payload: WorkspaceChatAskIn):
    """Server-Sent Events stream for workspace chat: same event protocol as
    single-document chat (meta, token, citations, done, error).

    Uses its own SessionLocal (same pattern as the single-doc /ask endpoint)
    because FastAPI closes yield-style dependencies before the SSE stream
    finishes."""
    from app.rag.workspace_chat import WorkspaceChatService

    session = SessionLocal()
    try:
        user = get_current_user(session)
        workspace = session.scalar(
            select(ResearchWorkspace).where(
                ResearchWorkspace.id == workspace_id,
                ResearchWorkspace.owner_id == user.id,
            )
        )
        if workspace is None:
            session.close()
            raise HTTPException(status_code=404, detail="Workspace not found.")

        conversation = _get_or_create_workspace_conversation(
            session, workspace, user.id, payload.conversation_id
        )
        session.commit()
        conversation_id = conversation.id
    except HTTPException:
        raise
    except Exception:
        session.close()
        raise

    async def event_stream():
        try:
            yield _sse("meta", {"conversation_id": conversation_id, "workspace_id": workspace_id})
            service = WorkspaceChatService(session, settings)
            # Re-fetch workspace inside the stream's session scope
            ws = session.get(ResearchWorkspace, workspace_id)
            conv = session.get(ChatConversation, conversation_id)
            user = get_current_user(session)
            async for event in service.answer(ws, conv, payload.question, user.id):
                if event.type == "token":
                    yield _sse("token", {"text": event.text})
                elif event.type == "citations":
                    yield _sse(
                        "citations",
                        {
                            "citations": event.citations or [],
                            "insufficient_evidence": event.insufficient_evidence,
                            "message_id": event.message_id,
                        },
                    )
                elif event.type == "done":
                    yield _sse("done", {"message_id": event.message_id})
                elif event.type == "error":
                    yield _sse("error", {"message": event.error})
        except Exception as exc:
            logger.exception("workspace chat stream crashed for workspace %s", workspace_id)
            yield _sse("error", {"message": f"Unexpected server error: {exc}"})
        finally:
            session.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
