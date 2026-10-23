"""RAGChatService — orchestrates the full USER QUERY FLOW:

    question -> query processing -> embedding -> vector retrieval ->
    relevant chunks -> context construction -> LLM -> answer + citations

This is the one place that wires the `Retriever` + `EmbeddingProvider` +
`LLMProvider` interfaces together into an actual chat turn, and the only
place `routers/chat.py` needs to import from `app/rag/`.
"""
from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.evidence import claim_service, evidence_service
from app.models import ChatConversation, ChatMessage, ChatRole, ClaimType, Document, RagStatus, SupportStatus
from app.rag import context as ctx
from app.rag.factory import get_embedding_provider, get_llm_provider, get_reranker
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.interfaces import Citation, LLMMessage, Retriever, RetrievedChunk
from app.rag.retriever import VectorRetriever

logger = logging.getLogger(__name__)

_MARKER_RE = re.compile(r"\[(\d+)\]")

SUGGESTED_QUESTIONS = [
    "What is the main contribution?",
    "What methodology was used?",
    "What dataset was used?",
    "What are the limitations?",
    "What are the key findings?",
]


class ChatServiceError(RuntimeError):
    """A user-facing error (not indexed yet, empty question, etc.)."""


@dataclass
class ChatEvent:
    type: str  # "token" | "citations" | "debug" | "done" | "error"
    text: str | None = None
    citations: list[dict] | None = None
    message_id: str | None = None
    insufficient_evidence: bool = False
    error: str | None = None
    debug: dict | None = None  # RetrievalDebug.to_dict() — only emitted when DEBUG_MODE is on


def build_retriever(db: Session, settings: Settings) -> Retriever:
    """Resolve the configured `Retriever` — the Phase 4 counterpart to
    `get_embedding_provider` / `get_llm_provider` in `app/rag/factory.py`.
    Kept here rather than in `factory.py` because, unlike the embedding/LLM
    provider seam, both retrievers are already public classes in
    `app/rag/`, not concrete implementations hidden behind the factory."""
    embedding_provider = get_embedding_provider(settings)
    mode = (settings.retrieval_mode or "hybrid").strip().lower()
    if mode == "vector_only":
        return VectorRetriever(db, embedding_provider)
    if mode != "hybrid":
        logger.warning("unknown RETRIEVAL_MODE %r; using hybrid", mode)
    return HybridRetriever(
        db,
        embedding_provider,
        get_reranker(settings),
        candidate_pool_size=settings.retrieval_candidate_pool_size,
        rrf_k=settings.retrieval_rrf_k,
        enable_reranking=settings.retrieval_enable_reranking,
    )


def get_or_create_conversation(
    db: Session, document: Document, owner_id: str, conversation_id: str | None
) -> ChatConversation:
    if conversation_id:
        conversation = db.scalar(
            select(ChatConversation).where(
                ChatConversation.id == conversation_id,
                ChatConversation.document_id == document.id,
            )
        )
        if conversation is not None:
            return conversation
    conversation = ChatConversation(document_id=document.id, owner_id=owner_id)
    db.add(conversation)
    db.flush()
    return conversation


def _history_pairs(conversation: ChatConversation, max_turns: int) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for message in conversation.messages:
        if message.role not in (ChatRole.USER, ChatRole.ASSISTANT):
            continue
        # Failed / partial turns are not real answers to build on, and an
        # empty assistant turn makes Gemini reject the whole request.
        if message.role == ChatRole.ASSISTANT and (message.error or message.incomplete or not (message.content or "").strip()):
            continue
        pairs.append((message.role, message.content))
    return pairs[-(max_turns * 2) :] if max_turns > 0 else []


def llm_model_used(llm) -> str:
    """The model that actually answered (a Gemini call may have fallen back
    from the configured one); the configured model for other providers."""
    used = getattr(llm, "last_model_used", None)
    return used if isinstance(used, str) and used else llm.model


def generation_error_message(exc: BaseException, partial: str) -> str:
    """User-facing text for a failed generation (shared by workspace chat)."""
    if getattr(exc, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(exc):
        reason = "The AI model's usage quota is exhausted right now. Please try again later."
    else:
        reason = "Please retry."
    if partial:
        return f"Generation stopped partway through; the partial answer above is incomplete. {reason}"
    return f"Generation failed. {reason}"


def below_min_score(chunks: list[RetrievedChunk], min_score: float) -> bool:
    """The optional pre-LLM gate (RETRIEVAL_MIN_SCORE, 0 = off): True when
    the best final retrieval score is below the threshold."""
    if min_score <= 0:
        return False
    return not chunks or max(c.score for c in chunks) < min_score


def _parse_citations(answer: str, used_chunks: list[RetrievedChunk]) -> tuple[list[Citation], bool]:
    insufficient = ctx.NOT_FOUND_SENTENCE.lower() in answer.lower() or not used_chunks

    markers = []
    for match in _MARKER_RE.finditer(answer):
        n = int(match.group(1))
        if n not in markers:
            markers.append(n)

    citations: list[Citation] = []
    if markers and not insufficient:
        for n in markers:
            if 1 <= n <= len(used_chunks):
                chunk = used_chunks[n - 1]
                citations.append(
                    Citation(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        page_number=chunk.page_number,
                        section_title=chunk.section_title,
                        snippet=_snippet(chunk.child_text),
                        score=chunk.score,
                        marker=n,
                    )
                )
    elif not insufficient and used_chunks:
        # The model answered but did not include [N] markers (can happen
        # with some hosted LLMs despite instructions) — still expose the
        # retrieval evidence that grounded the answer, per spec.
        for i, chunk in enumerate(used_chunks[:3], start=1):
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                    snippet=_snippet(chunk.child_text),
                    score=chunk.score,
                    marker=None,
                )
            )

    return citations, insufficient


def _snippet(text: str, max_chars: int = 320) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "\u2026"


class RAGChatService:
    def __init__(self, db: Session, settings: Settings | None = None):
        self.db = db
        self.settings = settings or get_settings()

    async def answer(
        self,
        document: Document,
        conversation: ChatConversation,
        question: str,
    ) -> AsyncIterator[ChatEvent]:
        question = (question or "").strip()
        if not question:
            yield ChatEvent(type="error", error="Question cannot be empty.")
            return

        if document.rag_status != RagStatus.INDEXED or not document.chunk_count:
            yield ChatEvent(
                type="error",
                error=(
                    "This document has not finished indexing for chat yet "
                    f"(status: {document.rag_status}). Try again shortly, or "
                    "trigger a reindex."
                ),
            )
            return

        user_message = ChatMessage(conversation_id=conversation.id, role=ChatRole.USER, content=question)
        self.db.add(user_message)
        # Commit now: an open transaction would hold SQLite's single write
        # lock for the whole (slow) LLM call and block indexing / other chats.
        self.db.commit()

        try:
            retriever = build_retriever(self.db, self.settings)
            # Off the event loop: embedding the question is a network call
            # (with retries) and must not freeze every other request.
            retrieved = await asyncio.to_thread(
                retriever.retrieve, document.id, question, self.settings.retrieval_top_k
            )
        except Exception as exc:
            logger.exception("retrieval failed for document %s", document.id)
            self._persist_error(conversation.id, f"Retrieval failed: {exc}")
            yield ChatEvent(type="error", error="Retrieval failed. Please try again.")
            return

        # Retrieval Debug panel data (see app/rag/interfaces.py:RetrievalDebug)
        # — only computed cost is what HybridRetriever already did; gating is
        # purely about *exposure*, per "Do NOT expose unnecessary technical
        # information to normal users" (see app/routers/chat.py, app/schemas.py).
        debug_dict: dict | None = None
        retrieval_debug = getattr(retriever, "last_debug", None)
        if self.settings.debug_mode and retrieval_debug is not None:
            debug_dict = retrieval_debug.to_dict()

        user_content, used_chunks = ctx.build_user_message(
            question, retrieved, max_chars=self.settings.retrieval_max_context_chars
        )

        if debug_dict is not None:
            yield ChatEvent(type="debug", debug=debug_dict)

        if below_min_score(retrieved, self.settings.retrieval_min_score):
            logger.info("document %s: best retrieval score below retrieval_min_score; skipping LLM", document.id)
            for event in self._not_found(conversation, question, debug_dict):
                yield event
            return

        history = _history_pairs(conversation, self.settings.chat_history_turns)
        # Drop the just-added user turn from history (it's the current
        # question, sent separately as the final, context-laden message).
        if history and history[-1] == (ChatRole.USER, question):
            history = history[:-1]
        messages = ctx.build_history_messages(history, self.settings.chat_history_turns)
        messages.append(LLMMessage(role="user", content=user_content))

        try:
            llm = get_llm_provider(self.settings)
        except Exception as exc:
            logger.exception("LLM provider unavailable")
            self._persist_error(conversation.id, f"LLM provider unavailable: {exc}")
            yield ChatEvent(type="error", error="The language model is currently unavailable.")
            return

        answer_parts: list[str] = []
        try:
            async for token in llm.stream(ctx.SYSTEM_PROMPT, messages):
                answer_parts.append(token)
                yield ChatEvent(type="token", text=token)
        except (GeneratorExit, asyncio.CancelledError):
            # Client went away mid-answer: keep what was generated, flagged
            # incomplete, then let the cancellation propagate.
            self._persist_error(
                conversation.id, "Client disconnected before the answer finished.",
                partial_answer="".join(answer_parts), incomplete=True,
            )
            raise
        except Exception as exc:
            logger.exception("generation failed for document %s", document.id)
            partial = "".join(answer_parts)
            self._persist_error(conversation.id, f"Generation failed: {exc}", partial_answer=partial, incomplete=True)
            yield ChatEvent(type="error", error=generation_error_message(exc, partial))
            return

        answer = "".join(answer_parts).strip()
        if not answer:
            answer = ctx.NOT_FOUND_SENTENCE

        citations, insufficient = _parse_citations(answer, used_chunks)

        assistant_message = ChatMessage(
            conversation_id=conversation.id,
            role=ChatRole.ASSISTANT,
            content=answer,
            citations=[c.to_dict() for c in citations],
            insufficient_evidence=insufficient,
            llm_provider=llm.name,
            llm_model=llm_model_used(llm),
            retrieval_debug=debug_dict,
        )
        self.db.add(assistant_message)
        self.db.flush()  # assigns assistant_message.id, needed by the Claim below

        # Evidence & Research Object Layer (Task 6): ground this answer's
        # citations in persisted Evidence rows, via a Claim representing the
        # answer itself. Never blocks the chat turn — a failure here still
        # leaves the answer + plain citations intact, exactly as before this
        # layer existed (see app/rag/interfaces.py:Citation.evidence_id).
        self._ground_citations_in_evidence(document, conversation, assistant_message, citations, used_chunks, insufficient)
        assistant_message.citations = [c.to_dict() for c in citations]

        conversation.updated_at = datetime.now(timezone.utc)
        if not conversation.title:
            conversation.title = question[:120]
        self.db.commit()

        yield ChatEvent(
            type="citations",
            citations=[c.to_dict() for c in citations],
            insufficient_evidence=insufficient,
            message_id=assistant_message.id,
        )
        yield ChatEvent(type="done", message_id=assistant_message.id)

    def _ground_citations_in_evidence(
        self,
        document: Document,
        conversation: ChatConversation,
        assistant_message: ChatMessage,
        citations: list[Citation],
        used_chunks: list[RetrievedChunk],
        insufficient: bool,
    ) -> None:
        """Task 6: convert the chunks that backed this answer into Evidence
        rows, attach them to a Claim representing the answer, and stamp each
        Citation with its evidence_id. Uses `used_chunks` (the exact output
        of the existing retrieval pipeline) as the source of evidence — no
        second retrieval, per spec."""
        if not citations or insufficient:
            return
        try:
            # A SAVEPOINT, not the outer transaction: if evidence/claim
            # creation fails partway through, only this nested work rolls
            # back — the already-flushed user/assistant messages (and their
            # ids) are untouched, so the caller's commit still saves the
            # plain answer + citations exactly as it would have before this
            # layer existed.
            with self.db.begin_nested():
                claim = claim_service.create_claim(
                    self.db,
                    document_id=document.id,
                    claim_text=assistant_message.content,
                    claim_type=ClaimType.UNKNOWN,
                    support_status=SupportStatus.UNCERTAIN,
                    conversation_id=conversation.id,
                    message_id=assistant_message.id,
                    commit=False,
                )
                chunks_by_id = {chunk.chunk_id: chunk for chunk in used_chunks}
                for citation in citations:
                    chunk = chunks_by_id.get(citation.chunk_id)
                    if chunk is None:
                        continue
                    kwargs = evidence_service.evidence_from_retrieved_chunk(chunk)
                    evidence = evidence_service.create_evidence(
                        self.db, claim_id=claim.id, commit=False, **kwargs
                    )
                    citation.evidence_id = evidence.id
                self.db.flush()
        except Exception:
            logger.exception(
                "evidence grounding failed for document %s (answer still returned)", document.id
            )
            for citation in citations:
                citation.evidence_id = None

    def _not_found(self, conversation: ChatConversation, question: str, debug_dict: dict | None):
        """Answer NOT_FOUND_SENTENCE without calling the LLM (min-score gate)."""
        assistant_message = ChatMessage(
            conversation_id=conversation.id,
            role=ChatRole.ASSISTANT,
            content=ctx.NOT_FOUND_SENTENCE,
            citations=[],
            insufficient_evidence=True,
            retrieval_debug=debug_dict,
        )
        self.db.add(assistant_message)
        conversation.updated_at = datetime.now(timezone.utc)
        if not conversation.title:
            conversation.title = question[:120]
        self.db.commit()
        yield ChatEvent(type="token", text=ctx.NOT_FOUND_SENTENCE)
        yield ChatEvent(type="citations", citations=[], insufficient_evidence=True, message_id=assistant_message.id)
        yield ChatEvent(type="done", message_id=assistant_message.id)

    def _persist_error(
        self, conversation_id: str, message: str, partial_answer: str = "", incomplete: bool = False
    ) -> None:
        try:
            self.db.add(
                ChatMessage(
                    conversation_id=conversation_id,
                    role=ChatRole.ASSISTANT,
                    content=partial_answer or "",
                    citations=[],
                    error=message[:2000],
                    incomplete=incomplete and bool(partial_answer),
                )
            )
            self.db.commit()
        except Exception:
            logger.exception("could not persist chat error message")
            self.db.rollback()
