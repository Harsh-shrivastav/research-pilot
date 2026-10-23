"""Phase 6A — Workspace chat service.

Orchestrates multi-paper workspace chat using the EXISTING RAG
infrastructure.  This is the workspace counterpart to ``RAGChatService``
in ``app/rag/service.py`` (single-document chat).

Architecture:

    Workspace
       ↓
    Workspace Retrieval (app/rag/workspace_retrieval.py)
       ↓  calls existing HybridRetriever per-document
    Merged RetrievedChunks with document_id preserved
       ↓
    Workspace Context Construction (app/rag/workspace_context.py)
       ↓  annotates each source with paper title
    LLM (existing LLMProvider)
       ↓
    Answer + Citations (with paper identity)
       ↓
    Evidence Grounding (existing Evidence layer)

Nothing here replaces or duplicates the single-document ``RAGChatService``.
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.evidence import claim_service, evidence_service
from app.models import (
    ChatConversation,
    ChatMessage,
    ChatRole,
    ClaimType,
    Document,
    ResearchWorkspace,
    SupportStatus,
    WorkspaceDocument,
)
from app.rag import context as ctx
from app.rag.factory import get_embedding_provider, get_llm_provider, get_reranker
from app.rag.interfaces import Citation, LLMMessage, RetrievedChunk
from app.rag.query_rewrite import rewrite_enabled, rewrite_for_papers
from app.rag.service import (
    ChatEvent,
    ChatServiceError,
    _history_pairs,
    _snippet,
    build_retriever,
    generation_error_message,
    llm_model_used,
)
from app.rag.workspace_context import (
    WORKSPACE_NOT_FOUND_SENTENCE,
    WORKSPACE_SYSTEM_PROMPT,
    build_workspace_user_message,
    get_document_titles,
)
from app.rag.workspace_retrieval import (
    WorkspaceRetrievalError,
    indexed_workspace_documents,
    retrieve_for_workspace,
)

logger = logging.getLogger(__name__)

_MARKER_RE = re.compile(r"\[(\d+)\]")


def _parse_workspace_citations(
    answer: str, used_chunks: list[RetrievedChunk]
) -> tuple[list[Citation], bool]:
    """Parse [N] citation markers from the answer, mapping them to the
    used chunks.  Same logic as single-doc ``_parse_citations`` but
    preserves workspace-specific document_id on each citation."""
    insufficient = (
        WORKSPACE_NOT_FOUND_SENTENCE.lower() in answer.lower()
        or ctx.NOT_FOUND_SENTENCE.lower() in answer.lower()
        or not used_chunks
    )

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


class WorkspaceChatService:
    """Multi-paper workspace chat, reusing the existing RAG pipeline."""

    def __init__(self, db: Session, settings: Settings | None = None):
        self.db = db
        self.settings = settings or get_settings()

    async def answer(
        self,
        workspace: ResearchWorkspace,
        conversation: ChatConversation,
        question: str,
        owner_id: str,
    ) -> AsyncIterator[ChatEvent]:
        question = (question or "").strip()
        if not question:
            yield ChatEvent(type="error", error="Question cannot be empty.")
            return

        user_message = ChatMessage(
            conversation_id=conversation.id,
            role=ChatRole.USER,
            content=question,
        )
        self.db.add(user_message)
        # Commit now so the write lock isn't held during the slow LLM calls.
        self.db.commit()

        try:
            llm = get_llm_provider(self.settings)
        except Exception as exc:
            logger.exception("LLM provider unavailable")
            self._persist_error(conversation.id, f"LLM provider unavailable: {exc}")
            yield ChatEvent(type="error", error="The language model is currently unavailable.")
            return

        # Retrieve across workspace documents, one (rewritten) sub-query per paper
        try:
            documents = indexed_workspace_documents(self.db, workspace.id, owner_id)
            queries_by_doc = await self._rewrite_queries(llm, question, documents)
            retriever = build_retriever(self.db, self.settings)
            retrieved = await asyncio.to_thread(
                retrieve_for_workspace,
                retriever,
                self.db,
                workspace.id,
                owner_id,
                question,
                self.settings.retrieval_top_k,
                queries_by_doc=queries_by_doc,
                min_per_paper=self.settings.workspace_min_chunks_per_paper,
                min_score=self.settings.retrieval_min_score,
                documents=documents,
            )
        except WorkspaceRetrievalError as wre:
            self._persist_error(conversation.id, str(wre))
            yield ChatEvent(type="error", error=str(wre))
            return
        except Exception as exc:
            logger.exception("workspace retrieval failed for workspace %s", workspace.id)
            self._persist_error(conversation.id, f"Retrieval failed: {exc}")
            yield ChatEvent(type="error", error="Retrieval failed. Please try again.")
            return

        if self.settings.retrieval_min_score > 0 and not retrieved:
            # Every paper fell below the optional pre-LLM gate.
            for event in self._not_found(conversation, question):
                yield event
            return

        # Build multi-paper context
        doc_ids = list({chunk.document_id for chunk in retrieved})
        doc_titles = get_document_titles(self.db, doc_ids)

        user_content, used_chunks = build_workspace_user_message(
            question,
            retrieved,
            doc_titles,
            max_chars=self.settings.retrieval_max_context_chars,
        )

        # Build conversation history
        history = self._history_pairs(conversation, self.settings.chat_history_turns)
        if history and history[-1] == (ChatRole.USER, question):
            history = history[:-1]
        messages = ctx.build_history_messages(history, self.settings.chat_history_turns)
        messages.append(LLMMessage(role="user", content=user_content))

        answer_parts: list[str] = []
        try:
            async for token in llm.stream(WORKSPACE_SYSTEM_PROMPT, messages):
                answer_parts.append(token)
                yield ChatEvent(type="token", text=token)
        except (GeneratorExit, asyncio.CancelledError):
            self._persist_error(
                conversation.id, "Client disconnected before the answer finished.",
                partial_answer="".join(answer_parts), incomplete=True,
            )
            raise
        except Exception as exc:
            logger.exception("generation failed for workspace %s", workspace.id)
            partial = "".join(answer_parts)
            self._persist_error(conversation.id, f"Generation failed: {exc}", partial_answer=partial, incomplete=True)
            yield ChatEvent(type="error", error=generation_error_message(exc, partial))
            return

        answer = "".join(answer_parts).strip()
        if not answer:
            answer = WORKSPACE_NOT_FOUND_SENTENCE

        citations, insufficient = _parse_workspace_citations(answer, used_chunks)

        assistant_message = ChatMessage(
            conversation_id=conversation.id,
            role=ChatRole.ASSISTANT,
            content=answer,
            citations=[c.to_dict() for c in citations],
            insufficient_evidence=insufficient,
            llm_provider=llm.name,
            llm_model=llm_model_used(llm),
        )
        self.db.add(assistant_message)
        self.db.flush()

        # Ground citations in Evidence (reuse Phase 5)
        self._ground_citations_in_evidence(
            workspace, conversation, assistant_message, citations, used_chunks, insufficient
        )
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
        workspace: ResearchWorkspace,
        conversation: ChatConversation,
        assistant_message: ChatMessage,
        citations: list[Citation],
        used_chunks: list[RetrievedChunk],
        insufficient: bool,
    ) -> None:
        """Ground workspace chat citations in Evidence rows, one per chunk,
        each attached to its correct document_id.  Uses the same Evidence layer
        as single-document chat (Phase 5)."""
        if not citations or insufficient:
            return
        try:
            with self.db.begin_nested():
                chunks_by_id = {chunk.chunk_id: chunk for chunk in used_chunks}
                for citation in citations:
                    chunk = chunks_by_id.get(citation.chunk_id)
                    if chunk is None:
                        continue
                    # Create a claim for this specific document
                    claim = claim_service.create_claim(
                        self.db,
                        document_id=chunk.document_id,
                        claim_text=assistant_message.content,
                        claim_type=ClaimType.UNKNOWN,
                        support_status=SupportStatus.UNCERTAIN,
                        workspace_id=workspace.id,
                        conversation_id=conversation.id,
                        message_id=assistant_message.id,
                        commit=False,
                    )
                    kwargs = evidence_service.evidence_from_retrieved_chunk(chunk)
                    evidence = evidence_service.create_evidence(
                        self.db, claim_id=claim.id, commit=False, **kwargs
                    )
                    citation.evidence_id = evidence.id
                self.db.flush()
        except Exception:
            logger.exception(
                "evidence grounding failed for workspace %s (answer still returned)",
                workspace.id,
            )
            for citation in citations:
                citation.evidence_id = None

    def _history_pairs(
        self, conversation: ChatConversation, max_turns: int
    ) -> list[tuple[str, str]]:
        return _history_pairs(conversation, max_turns)

    async def _rewrite_queries(self, llm, question: str, documents: list[Document]) -> dict[str, str] | None:
        """One focused sub-query per paper (app/rag/query_rewrite.py), or
        None (= original question everywhere) when off, single-paper, or slow.
        Runs in a thread with a timeout: rewriting is an optimisation and must
        never hold the answer hostage to a slow or overloaded model."""
        if len(documents) < 2 or not rewrite_enabled(self.settings.query_rewrite_enabled, llm):
            return None
        papers = [{"document_id": d.id, "title": d.title or d.original_filename} for d in documents]
        try:
            queries = await asyncio.wait_for(
                asyncio.to_thread(rewrite_for_papers, llm, question, papers),
                timeout=self.settings.query_rewrite_timeout_seconds,
            )
        except Exception as exc:  # includes TimeoutError
            logger.warning("query rewrite skipped (%s); using the original question", type(exc).__name__)
            return None
        logger.info("workspace query rewrite: %s", queries)
        return queries

    def _not_found(self, conversation: ChatConversation, question: str):
        """Answer WORKSPACE_NOT_FOUND_SENTENCE without calling the LLM."""
        assistant_message = ChatMessage(
            conversation_id=conversation.id,
            role=ChatRole.ASSISTANT,
            content=WORKSPACE_NOT_FOUND_SENTENCE,
            citations=[],
            insufficient_evidence=True,
        )
        self.db.add(assistant_message)
        conversation.updated_at = datetime.now(timezone.utc)
        if not conversation.title:
            conversation.title = question[:120]
        self.db.commit()
        yield ChatEvent(type="token", text=WORKSPACE_NOT_FOUND_SENTENCE)
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
            logger.exception("could not persist workspace chat error message")
            self.db.rollback()
