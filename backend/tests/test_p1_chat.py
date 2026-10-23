"""P1 evidence chat: query rewrite fallback, per-paper quota merge, the
optional min-score gate, and partial-answer persistence on stream failure.
Mocks only — no DB, no network.

Run: python -m unittest tests.test_p1_chat   (from backend/)
"""
import asyncio
import os
import sys
import types as pytypes
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["DATABASE_URL"] = "sqlite://"
# Env vars beat backend/.env: keep tests on the offline providers so they
# never call a real API, whatever the developer has configured.
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["LLM_PROVIDER"] = "local"
os.environ["GEMINI_API_KEY"] = ""

from app.config import Settings  # noqa: E402
from app.models import ChatMessage, RagStatus  # noqa: E402
from app.rag import context as ctx  # noqa: E402
from app.rag.interfaces import RetrievedChunk  # noqa: E402
from app.rag.query_rewrite import rewrite_enabled, rewrite_for_papers  # noqa: E402
from app.rag.service import RAGChatService, _history_pairs, below_min_score  # noqa: E402
from app.rag.workspace_retrieval import merge_with_quota  # noqa: E402

PAPERS = [{"document_id": "d1", "title": "Paper One"}, {"document_id": "d2", "title": "Paper Two"}]


def _chunk(doc, i, score):
    return RetrievedChunk(
        chunk_id=f"{doc}-{i}", parent_chunk_id=None, document_id=doc, page_number=i,
        section_id=None, section_title="S", child_text=f"text {doc} {i}",
        context_text=f"ctx {doc} {i}", score=score,
    )


def _llm(reply=None, exc=None, name="gemini"):
    llm = MagicMock()
    llm.name, llm.model = name, "m"
    llm.generate.side_effect = exc if exc else (lambda system, messages: reply)
    return llm


class TestQueryRewrite(unittest.TestCase):
    def test_valid_fenced_json(self):
        reply = '```json\n{"queries": [{"document_id": "d1", "query": "What method does Paper One use?"},' \
                ' {"document_id": "d2", "query": "What method does Paper Two use?"}]}\n```'
        out = rewrite_for_papers(_llm(reply), "compare their methods", PAPERS)
        self.assertEqual(out, {"d1": "What method does Paper One use?", "d2": "What method does Paper Two use?"})

    def test_fallbacks(self):
        q = "compare their methods"
        for llm in (_llm("not json at all"), _llm(exc=RuntimeError("503")), _llm('{"queries": "nope"}')):
            with self.assertLogs("app.rag.query_rewrite", "WARNING"):
                self.assertEqual(rewrite_for_papers(llm, q, PAPERS), {"d1": q, "d2": q})

    def test_missing_and_unknown_ids_fall_back_per_paper(self):
        reply = '{"queries": [{"document_id": "d1", "query": "Q1"}, {"document_id": "zzz", "query": "x"}]}'
        with self.assertLogs("app.rag.query_rewrite", "WARNING"):
            out = rewrite_for_papers(_llm(reply), "orig", PAPERS)
        self.assertEqual(out, {"d1": "Q1", "d2": "orig"})

    def test_auto_enable_only_for_gemini(self):
        self.assertTrue(rewrite_enabled(None, _llm(name="gemini")))
        self.assertFalse(rewrite_enabled(None, _llm(name="local")))
        self.assertTrue(rewrite_enabled(True, _llm(name="local")))


class TestQuotaMerge(unittest.TestCase):
    def test_every_paper_gets_quota_before_extras(self):
        strong = [_chunk("A", i, 0.9 - i * 0.01) for i in range(5)]
        weak = [_chunk("B", i, 0.2 - i * 0.01) for i in range(3)]
        single = [_chunk("C", 0, 0.1)]
        out = merge_with_quota({"A": strong, "B": weak, "C": single}, top_k=6, min_per_paper=2)
        by_doc = [c.document_id for c in out]
        self.assertEqual(by_doc[:5], ["A", "B", "C", "A", "B"])  # quota first, round-robin
        self.assertEqual(len(out), 6)
        self.assertEqual(out[5].chunk_id, "A-2")  # extra slot goes to the best leftover
        self.assertEqual(by_doc.count("B"), 2)

    def test_quota_may_exceed_top_k(self):
        docs = {d: [_chunk(d, i, 0.5) for i in range(3)] for d in "ABCD"}
        self.assertEqual(len(merge_with_quota(docs, top_k=6, min_per_paper=2)), 8)


class TestGateAndHistory(unittest.TestCase):
    def test_below_min_score(self):
        self.assertFalse(below_min_score([], 0.0))  # gate off
        self.assertTrue(below_min_score([], 0.3))
        self.assertTrue(below_min_score([_chunk("A", 0, 0.2)], 0.3))
        self.assertFalse(below_min_score([_chunk("A", 0, 0.4)], 0.3))

    def test_history_skips_failed_and_empty_turns(self):
        msgs = [
            pytypes.SimpleNamespace(role="user", content="q1", error=None, incomplete=False),
            pytypes.SimpleNamespace(role="assistant", content="", error="boom", incomplete=False),
            pytypes.SimpleNamespace(role="user", content="q2", error=None, incomplete=False),
            pytypes.SimpleNamespace(role="assistant", content="half", error="x", incomplete=True),
            pytypes.SimpleNamespace(role="assistant", content="ok [1]", error=None, incomplete=None),
        ]
        conv = pytypes.SimpleNamespace(messages=msgs)
        self.assertEqual(_history_pairs(conv, 6), [("user", "q1"), ("user", "q2"), ("assistant", "ok [1]")])


class TestSingleDocService(unittest.TestCase):
    def _run(self, llm, chunks, min_score=0.0):
        db = MagicMock()
        added = []
        db.add.side_effect = added.append
        document = pytypes.SimpleNamespace(id="d1", rag_status=RagStatus.INDEXED, chunk_count=3)
        conversation = pytypes.SimpleNamespace(id="c1", messages=[], title=None, updated_at=None)
        retriever = MagicMock()
        retriever.retrieve.return_value = chunks
        retriever.last_debug = None
        settings = Settings(retrieval_min_score=min_score)
        with patch("app.rag.service.build_retriever", return_value=retriever), \
             patch("app.rag.service.get_llm_provider", return_value=llm):
            async def collect():
                return [e async for e in RAGChatService(db, settings).answer(document, conversation, "q?")]
            events = asyncio.run(collect())
        return events, [m for m in added if isinstance(m, ChatMessage) and m.role == "assistant"]

    def test_stream_failure_midway_persists_partial_as_incomplete(self):
        async def stream(system, messages):
            yield "The method "
            yield "uses"
            raise RuntimeError("503 UNAVAILABLE")

        llm = MagicMock()
        llm.name, llm.model = "gemini", "m"
        llm.stream = stream
        with self.assertLogs("app.rag.service", "ERROR"):
            events, saved = self._run(llm, [_chunk("d1", 1, 0.8)])

        self.assertEqual([e.type for e in events], ["token", "token", "error"])
        self.assertIn("incomplete", events[-1].error)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].content, "The method uses")
        self.assertTrue(saved[0].incomplete)
        self.assertIn("503", saved[0].error)

    def test_min_score_gate_skips_llm(self):
        llm = MagicMock()
        events, saved = self._run(llm, [_chunk("d1", 1, 0.1)], min_score=0.5)
        llm.stream.assert_not_called()
        self.assertEqual([e.type for e in events], ["token", "citations", "done"])
        self.assertEqual(events[0].text, ctx.NOT_FOUND_SENTENCE)
        self.assertTrue(events[1].insufficient_evidence)
        self.assertTrue(saved[0].insufficient_evidence)


if __name__ == "__main__":
    unittest.main()
