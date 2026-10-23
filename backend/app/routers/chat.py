"""Phase 3 chat endpoints: conversations, the streaming ask endpoint, RAG
indexing status/reindex, and the fixed suggested-question set.

The streaming `/ask` endpoint intentionally does not use `Depends(get_db)`:
FastAPI closes yield-style dependencies once the route handler *returns* the
response object, which for a `StreamingResponse` is before the body has
actually finished streaming — closing the DB session out from under an
in-flight generator. Instead it opens its own `SessionLocal()` for the whole
request and closes it itself once the stream is exhausted.
"""
import json
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import SessionLocal, get_db
from app.deps import get_current_user
from app.models import ChatConversation, ChatMessage, Document, ProcessingStatus, RagStatus
from app.rag.service import SUGGESTED_QUESTIONS, RAGChatService, get_or_create_conversation
from app.routers.documents import _get_owned_document
from app.schemas import (
    ChatAskIn,
    ChatConversationDetailOut,
    ChatConversationOut,
    ChatMessageOut,
    CitationOut,
    RagStatusOut,
    RetrievalDebugOut,
    SuggestedQuestionsOut,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])
settings = get_settings()


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _message_out(message: ChatMessage) -> ChatMessageOut:
    citations = [CitationOut(**c) for c in (message.citations or [])]
    # Gated at *read* time, independent of whether it was gated at write
    # time: if DEBUG_MODE is flipped off after a message was answered with it
    # on, that message's debug detail stops being served from here on,
    # exactly like a normal user hitting this same endpoint always would.
    debug = None
    if settings.debug_mode and message.retrieval_debug:
        debug = RetrievalDebugOut(**message.retrieval_debug)
    return ChatMessageOut(
        id=message.id,
        conversation_id=message.conversation_id,
        role=message.role,
        content=message.content,
        citations=citations,
        insufficient_evidence=bool(message.insufficient_evidence),
        llm_provider=message.llm_provider,
        llm_model=message.llm_model,
        error=message.error,
        incomplete=bool(message.incomplete),
        created_at=message.created_at,
        retrieval_debug=debug,
    )


def _conversation_out(conversation: ChatConversation, message_count: int | None = None) -> ChatConversationOut:
    return ChatConversationOut(
        id=conversation.id,
        document_id=conversation.document_id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        message_count=message_count if message_count is not None else len(conversation.messages),
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# RAG indexing status / reindex
# ---------------------------------------------------------------------------


@router.get("/api/documents/{document_id}/rag-status", response_model=RagStatusOut)
def get_rag_status(document_id: str, db: Session = Depends(get_db)) -> RagStatusOut:
    doc = _get_owned_document(db, document_id)
    return RagStatusOut(
        id=doc.id,
        rag_status=doc.rag_status or RagStatus.NOT_INDEXED,
        rag_error=doc.rag_error,
        chunk_count=doc.chunk_count or 0,
        embedding_provider=doc.embedding_provider,
        embedding_model=doc.embedding_model,
    )


def _run_reindex(document_id: str) -> None:
    from app.worker import index_document_rag

    index_document_rag(document_id)


@router.post("/api/documents/{document_id}/reindex", response_model=RagStatusOut)
def reindex_document(
    document_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
) -> RagStatusOut:
    doc = _get_owned_document(db, document_id)
    if doc.processing_status != ProcessingStatus.READY:
        raise HTTPException(
            status_code=409,
            detail="Document must finish Phase 2 processing (structure extraction) before it can be indexed for chat.",
        )
    if doc.rag_status == RagStatus.INDEXING:
        raise HTTPException(status_code=409, detail="This document is already being indexed.")

    doc.rag_status = RagStatus.INDEXING
    doc.rag_error = None
    db.commit()
    db.refresh(doc)

    background_tasks.add_task(_run_reindex, document_id)

    return RagStatusOut(
        id=doc.id,
        rag_status=doc.rag_status,
        rag_error=doc.rag_error,
        chunk_count=doc.chunk_count or 0,
        embedding_provider=doc.embedding_provider,
        embedding_model=doc.embedding_model,
    )


# ---------------------------------------------------------------------------
# Suggested questions
# ---------------------------------------------------------------------------


@router.get("/api/documents/{document_id}/chat/suggested-questions", response_model=SuggestedQuestionsOut)
def suggested_questions(document_id: str, db: Session = Depends(get_db)) -> SuggestedQuestionsOut:
    _get_owned_document(db, document_id)  # 404s / ownership check
    return SuggestedQuestionsOut(questions=list(SUGGESTED_QUESTIONS))


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


@router.get("/api/documents/{document_id}/chat/conversations", response_model=list[ChatConversationOut])
def list_conversations(document_id: str, db: Session = Depends(get_db)) -> list[ChatConversationOut]:
    doc = _get_owned_document(db, document_id)
    conversations = db.scalars(
        select(ChatConversation)
        .where(ChatConversation.document_id == doc.id)
        .order_by(ChatConversation.updated_at.desc())
    ).all()
    return [_conversation_out(c) for c in conversations]


@router.post("/api/documents/{document_id}/chat/conversations", response_model=ChatConversationOut)
def create_conversation(document_id: str, db: Session = Depends(get_db)) -> ChatConversationOut:
    doc = _get_owned_document(db, document_id)
    user = get_current_user(db)
    conversation = ChatConversation(document_id=doc.id, owner_id=user.id)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return _conversation_out(conversation, message_count=0)


@router.get("/api/chat/conversations/{conversation_id}", response_model=ChatConversationDetailOut)
def get_conversation(conversation_id: str, db: Session = Depends(get_db)) -> ChatConversationDetailOut:
    user = get_current_user(db)
    conversation = db.scalar(
        select(ChatConversation).where(
            ChatConversation.id == conversation_id, ChatConversation.owner_id == user.id
        )
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    base = _conversation_out(conversation)
    return ChatConversationDetailOut(
        **base.model_dump(), messages=[_message_out(m) for m in conversation.messages]
    )


@router.delete("/api/chat/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)) -> None:
    """Backs the UI's "clear conversation" action."""
    user = get_current_user(db)
    conversation = db.scalar(
        select(ChatConversation).where(
            ChatConversation.id == conversation_id, ChatConversation.owner_id == user.id
        )
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    db.delete(conversation)
    db.commit()


# ---------------------------------------------------------------------------
# Ask (streaming)
# ---------------------------------------------------------------------------


@router.post("/api/documents/{document_id}/chat/ask")
async def ask(document_id: str, payload: ChatAskIn):
    """Server-Sent Events stream: a `meta` event (conversation id), then
    `token` events as the answer generates, then a `citations` event, then
    `done` (or `error` at any point)."""
    session = SessionLocal()
    try:
        document = session.get(Document, document_id)
        if document is None:
            session.close()
            raise HTTPException(status_code=404, detail="Document not found.")
        user = get_current_user(session)
        if document.owner_id != user.id:
            session.close()
            raise HTTPException(status_code=404, detail="Document not found.")

        conversation = get_or_create_conversation(session, document, user.id, payload.conversation_id)
        session.commit()
        conversation_id = conversation.id
    except HTTPException:
        raise
    except Exception:
        session.close()
        raise

    async def event_stream():
        try:
            yield _sse("meta", {"conversation_id": conversation_id})
            service = RAGChatService(session, settings)
            async for event in service.answer(document, conversation, payload.question):
                if event.type == "token":
                    yield _sse("token", {"text": event.text})
                elif event.type == "debug":
                    # RAGChatService only ever produces this event when
                    # settings.debug_mode is on (see rag/service.py), so no
                    # second gate is needed here — but the comment stays as a
                    # tripwire for future readers: this branch must never
                    # forward debug data unconditionally.
                    yield _sse("debug", {"debug": event.debug})
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
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("chat stream crashed for document %s", document_id)
            yield _sse("error", {"message": f"Unexpected server error: {exc}"})
        finally:
            session.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
