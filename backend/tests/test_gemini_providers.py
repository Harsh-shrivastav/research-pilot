"""Offline tests for the Gemini providers (F0 + P0 hardening). No API key /
network needed: `google.genai` is replaced by a tiny stub, and the SDK client
is a fake. Uses unittest + in-memory SQLite like tests/test_comparison.py.

Run: python -m unittest tests.test_gemini_providers   (from backend/)
"""
import asyncio
import math
import os
import sys
import threading
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

from app.rag import gemini_limits  # noqa: E402
from app.rag.interfaces import LLMMessage  # noqa: E402


def _stub_genai_modules() -> dict:
    google = pytypes.ModuleType("google")
    genai = pytypes.ModuleType("google.genai")
    t = pytypes.ModuleType("google.genai.types")

    class _Obj:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class Part:
        @staticmethod
        def from_text(text):
            return _Obj(text=text)

    t.EmbedContentConfig = _Obj
    t.GenerateContentConfig = _Obj
    t.Content = _Obj
    t.Part = Part
    t.HttpOptions = _Obj
    t.ThinkingConfig = _Obj
    genai.types = t
    genai.Client = lambda **kw: None
    google.genai = genai
    return {"google": google, "google.genai": genai, "google.genai.types": t}


class APIError(Exception):
    """Shaped like google.genai.errors.APIError (HTTP status on `.code`)."""

    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def embed_content(self, model, contents, config):
        self.calls.append((model, list(contents), config.task_type, config.output_dimensionality))
        emb = [pytypes.SimpleNamespace(values=[2.0] * config.output_dimensionality) for _ in contents]
        return pytypes.SimpleNamespace(embeddings=emb)


class _GeminiTestBase(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(sys.modules, _stub_genai_modules())
        patcher.start()
        self.addCleanup(patcher.stop)
        # Never actually sleep during backoff.
        for target in ("app.rag.llm.gemini_provider.time.sleep",
                       "app.rag.embeddings.gemini_provider.time.sleep"):
            p = patch(target, lambda s: None)
            p.start()
            self.addCleanup(p.stop)
        gemini_limits.reset_limiter(4)
        self.addCleanup(gemini_limits.reset_limiter, None)
        gemini_limits.reset_rate_limit(0)  # no pacing unless a test opts in
        self.addCleanup(gemini_limits.reset_rate_limit, None)
        gemini_limits.reset_model_registry()
        self.addCleanup(gemini_limits.reset_model_registry)


class TestEmbeddings(_GeminiTestBase):
    def test_dimension_normalisation_and_task_types(self):
        from app.rag.embeddings.gemini_provider import GeminiEmbeddingProvider

        fake = FakeEmbeddings()
        p = GeminiEmbeddingProvider(api_key="x", dimensions=768, client=pytypes.SimpleNamespace(models=fake))

        docs = p.embed_documents([f"chunk {i}" for i in range(120)])
        self.assertEqual(len(docs), 120)
        self.assertTrue(all(len(v) == 768 for v in docs))
        self.assertTrue(math.isclose(math.sqrt(sum(x * x for x in docs[0])), 1.0, rel_tol=1e-6))
        self.assertEqual({c[2] for c in fake.calls}, {"RETRIEVAL_DOCUMENT"})
        self.assertEqual([len(c[1]) for c in fake.calls], [50, 50, 20])  # batching

        q = p.embed_query("what dataset?")
        self.assertEqual(len(q), 768)
        self.assertEqual(fake.calls[-1][2], "RETRIEVAL_QUERY")
        self.assertEqual(fake.calls[-1][0], "gemini-embedding-001")

    def test_retries_then_raises(self):
        from app.rag.embeddings.gemini_provider import GeminiEmbeddingProvider

        models = MagicMock()
        models.embed_content.side_effect = APIError(429)
        p = GeminiEmbeddingProvider(api_key="x", client=pytypes.SimpleNamespace(models=models))
        with self.assertRaisesRegex(RuntimeError, "failed after"):
            p.embed_query("hi")
        self.assertEqual(models.embed_content.call_count, gemini_limits.MAX_RETRIES)

    def test_transient_then_succeeds_and_client_errors_fail_fast(self):
        from app.rag.embeddings.gemini_provider import GeminiEmbeddingProvider

        ok = pytypes.SimpleNamespace(embeddings=[pytypes.SimpleNamespace(values=[1.0] * 8)])
        models = MagicMock()
        models.embed_content.side_effect = [APIError(503), TimeoutError("slow"), ok]
        p = GeminiEmbeddingProvider(api_key="x", dimensions=8, client=pytypes.SimpleNamespace(models=models))
        self.assertEqual(len(p.embed_query("hi")), 8)
        self.assertEqual(models.embed_content.call_count, 3)

        models.reset_mock(side_effect=True)
        models.embed_content.side_effect = APIError(403)
        with self.assertRaises(APIError):
            p.embed_query("a different question")  # "hi" would be served from the query cache
        self.assertEqual(models.embed_content.call_count, 1)


class TestLLMGenerate(_GeminiTestBase):
    def _llm(self, models=None, aio_models=None):
        from app.rag.llm.gemini_provider import GeminiLLMProvider

        client = pytypes.SimpleNamespace(models=models, aio=pytypes.SimpleNamespace(models=aio_models))
        return GeminiLLMProvider(api_key="x", client=client)

    def test_role_mapping_generate_and_stream(self):
        seen = {}

        class Models:
            def generate_content(self, model, contents, config):
                seen["roles"] = [c.role for c in contents]
                seen["system"] = config.system_instruction
                seen["model"] = model
                return pytypes.SimpleNamespace(text="answer [1]")

        class AioModels:
            async def generate_content_stream(self, model, contents, config):
                async def gen():
                    for piece in ["hel", None, "lo"]:
                        yield pytypes.SimpleNamespace(text=piece)
                return gen()

        llm = self._llm(Models(), AioModels())
        msgs = [LLMMessage("user", "q1"), LLMMessage("assistant", "a1"), LLMMessage("user", "q2")]

        self.assertEqual(llm.generate("SYS", msgs), "answer [1]")
        self.assertEqual(seen["roles"], ["user", "model", "user"])
        self.assertEqual(seen["system"], "SYS")
        self.assertEqual(seen["model"], "gemini-3.8-flash")

        async def collect():
            return [t async for t in llm.stream("SYS", msgs)]

        self.assertEqual("".join(asyncio.run(collect())), "hello")

    def test_generate_retries_transient_then_succeeds(self):
        models = MagicMock()
        models.generate_content.side_effect = [
            APIError(429), APIError(503), TimeoutError("slow"),
            pytypes.SimpleNamespace(text="ok"),
        ]
        self.assertEqual(self._llm(models).generate("S", [LLMMessage("user", "q")]), "ok")
        self.assertEqual(models.generate_content.call_count, 4)

    def test_generate_gives_up_after_max_attempts(self):
        models = MagicMock()
        models.generate_content.side_effect = APIError(500)
        with self.assertRaisesRegex(RuntimeError, "failed after 4 attempts"):
            self._llm(models).generate("S", [LLMMessage("user", "q")])
        self.assertEqual(models.generate_content.call_count, 4)

    def test_generate_does_not_retry_client_errors(self):
        models = MagicMock()
        models.generate_content.side_effect = APIError(400)
        with self.assertRaises(APIError):
            self._llm(models).generate("S", [LLMMessage("user", "q")])
        self.assertEqual(models.generate_content.call_count, 1)


class TestModelFallback(_GeminiTestBase):
    DAILY = "429 RESOURCE_EXHAUSTED. {'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'retryDelay': '13049s'}"

    @staticmethod
    def _daily_cap():
        exc = APIError(429)
        exc.args = (TestModelFallback.DAILY,)
        return exc

    def _llm(self, models=None, aio_models=None):
        from app.rag.llm.gemini_provider import GeminiLLMProvider

        client = pytypes.SimpleNamespace(models=models, aio=pytypes.SimpleNamespace(models=aio_models))
        return GeminiLLMProvider(api_key="x", client=client, model="primary", fallback_models=["backup"])

    def test_generate_falls_back_when_daily_cap_hit_and_remembers_it(self):
        seen = []

        class Models:
            def generate_content(inner, model, contents, config):
                seen.append(model)
                if model == "primary":
                    raise self._daily_cap()
                return pytypes.SimpleNamespace(text="from backup")

        llm = self._llm(Models())
        self.assertEqual(llm.generate("S", [LLMMessage("user", "q")]), "from backup")
        self.assertEqual(llm.last_model_used, "backup")
        self.assertEqual(seen, ["primary", "backup"])
        # Next call skips the exhausted model entirely.
        llm.generate("S", [LLMMessage("user", "q")])
        self.assertEqual(seen, ["primary", "backup", "backup"])

    def test_generate_falls_back_when_primary_stays_overloaded(self):
        seen = []

        class Models:
            def generate_content(inner, model, contents, config):
                seen.append(model)
                if model == "primary":
                    raise APIError(503)
                return pytypes.SimpleNamespace(text="ok")

        llm = self._llm(Models())
        self.assertEqual(llm.generate("S", [LLMMessage("user", "q")]), "ok")
        self.assertEqual(seen, ["primary", "primary", "backup"])  # FALLBACK_ATTEMPTS tries, then switch

    def test_client_errors_do_not_fall_back(self):
        models = MagicMock()
        models.generate_content.side_effect = APIError(400)
        with self.assertRaises(APIError):
            self._llm(models).generate("S", [LLMMessage("user", "q")])
        self.assertEqual(models.generate_content.call_count, 1)

    def test_stream_falls_back_before_first_chunk(self):
        seen = []

        class AioModels:
            async def generate_content_stream(inner, model, contents, config):
                seen.append(model)
                if model == "primary":
                    raise self._daily_cap()

                async def gen():
                    yield pytypes.SimpleNamespace(text="hi")
                return gen()

        llm = self._llm(aio_models=AioModels())

        async def run():
            return [t async for t in llm.stream("S", [LLMMessage("user", "q")])]

        self.assertEqual(asyncio.run(run()), ["hi"])
        self.assertEqual(seen, ["primary", "backup"])
        self.assertEqual(llm.last_model_used, "backup")

    def test_factory_wires_fallback_models(self):
        from app.config import Settings
        from app.rag.factory import get_llm_provider

        on = get_llm_provider(Settings(llm_provider="gemini", gemini_api_key="k", llm_model="m1"))
        self.assertEqual(on.model, "m1")
        self.assertGreaterEqual(len(on.fallback_models), 1)
        custom = get_llm_provider(Settings(llm_provider="gemini", gemini_api_key="k", llm_fallback_models="a, b"))
        self.assertEqual(custom.fallback_models, ("a", "b"))
        off = get_llm_provider(Settings(llm_provider="gemini", gemini_api_key="k", llm_fallback_models=""))
        self.assertEqual(off.fallback_models, ())


class TestLLMStream(_GeminiTestBase):
    def setUp(self):
        super().setUp()
        p = patch("app.rag.llm.gemini_provider.asyncio.sleep", MagicMock(side_effect=self._no_sleep))
        p.start()
        self.addCleanup(p.stop)

    @staticmethod
    async def _no_sleep(_seconds):
        return None

    def _llm(self, aio_models):
        from app.rag.llm.gemini_provider import GeminiLLMProvider

        client = pytypes.SimpleNamespace(models=None, aio=pytypes.SimpleNamespace(models=aio_models))
        return GeminiLLMProvider(api_key="x", client=client)

    @staticmethod
    def _collect(llm):
        async def run():
            return [t async for t in llm.stream("S", [LLMMessage("user", "q")])]
        return asyncio.run(run())

    def test_retries_when_failure_is_before_first_chunk(self):
        calls = {"n": 0}

        class AioModels:
            async def generate_content_stream(self, **kw):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise APIError(429)  # fails to open the stream

                async def gen():
                    if calls["n"] == 2:
                        raise APIError(503)  # opens, fails before any text
                    yield pytypes.SimpleNamespace(text="hi")
                return gen()

        self.assertEqual(self._collect(self._llm(AioModels())), ["hi"])
        self.assertEqual(calls["n"], 3)

    def test_no_retry_after_first_chunk(self):
        calls = {"n": 0}

        class AioModels:
            async def generate_content_stream(self, **kw):
                calls["n"] += 1

                async def gen():
                    yield pytypes.SimpleNamespace(text="partial")
                    raise APIError(503)
                return gen()

        llm = self._llm(AioModels())
        received = []

        async def run():
            async for t in llm.stream("S", [LLMMessage("user", "q")]):
                received.append(t)

        with self.assertRaises(APIError):
            asyncio.run(run())
        self.assertEqual(received, ["partial"])  # no duplicated text
        self.assertEqual(calls["n"], 1)

    def test_stream_releases_slot_on_error(self):
        gemini_limits.reset_limiter(1)

        class AioModels:
            async def generate_content_stream(self, **kw):
                raise APIError(400)

        with self.assertRaises(APIError):
            self._collect(self._llm(AioModels()))
        # The single slot must be free again.
        self.assertTrue(gemini_limits._get_semaphore().acquire(blocking=False))
        gemini_limits._get_semaphore().release()


class TestConcurrencyLimiter(_GeminiTestBase):
    def test_limit_shared_by_embedding_and_llm_providers(self):
        from app.rag.embeddings.gemini_provider import GeminiEmbeddingProvider
        from app.rag.llm.gemini_provider import GeminiLLMProvider

        gemini_limits.reset_limiter(2)
        state = {"active": 0, "peak": 0}
        lock = threading.Lock()

        def tracked(result):
            def call(*a, **kw):
                with lock:
                    state["active"] += 1
                    state["peak"] = max(state["peak"], state["active"])
                threading.Event().wait(0.05)  # time.sleep is patched out
                with lock:
                    state["active"] -= 1
                return result
            return call

        emb_models = MagicMock()
        emb_models.embed_content.side_effect = tracked(
            pytypes.SimpleNamespace(embeddings=[pytypes.SimpleNamespace(values=[1.0] * 8)])
        )
        llm_models = MagicMock()
        llm_models.generate_content.side_effect = tracked(pytypes.SimpleNamespace(text="x"))

        embedder = GeminiEmbeddingProvider(api_key="x", dimensions=8, client=pytypes.SimpleNamespace(models=emb_models))
        llm = GeminiLLMProvider(api_key="x", client=pytypes.SimpleNamespace(models=llm_models, aio=None))

        jobs = [lambda: embedder.embed_query("q")] * 4 + [lambda: llm.generate("S", [LLMMessage("user", "q")])] * 4
        threads = [threading.Thread(target=j) for j in jobs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(emb_models.embed_content.call_count, 4)
        self.assertEqual(llm_models.generate_content.call_count, 4)
        self.assertEqual(state["peak"], 2)

    def test_limit_read_from_settings(self):
        gemini_limits.reset_limiter(None)
        with patch("app.config.get_settings", return_value=MagicMock(gemini_max_concurrency=3)):
            sem = gemini_limits._get_semaphore()
        self.assertEqual(sem._initial_value, 3)

    def test_embed_pacer_waits_for_window(self):
        clock = {"now": 1000.0}
        sleeps = []

        def fake_sleep(seconds):
            sleeps.append(seconds)
            clock["now"] += seconds

        gemini_limits.reset_rate_limit(10)
        with patch("app.rag.gemini_limits.time.monotonic", lambda: clock["now"]), \
             patch("app.rag.gemini_limits.time.sleep", fake_sleep):
            gemini_limits.throttle_embed_texts(6)
            gemini_limits.throttle_embed_texts(4)  # exactly at the limit: no wait
            self.assertEqual(sleeps, [])
            clock["now"] += 20
            gemini_limits.throttle_embed_texts(5)  # must wait out the first batch
            self.assertEqual(len(sleeps), 1)
            self.assertAlmostEqual(sleeps[0], 40.05, places=2)
            gemini_limits.throttle_embed_texts(50)  # bigger than the limit still passes
        self.assertEqual(len(sleeps), 2)

    def test_backoff_honours_server_retry_hint(self):
        exc = APIError(429)
        exc.args = ("429 RESOURCE_EXHAUSTED. {'message': 'Please retry in 59.17s.', "
                    "'retryDelay': '59s'}",)
        self.assertAlmostEqual(gemini_limits.retry_after_seconds(exc), 59.17)
        self.assertGreaterEqual(gemini_limits.backoff_seconds(0, exc), 59.17)
        self.assertLess(gemini_limits.backoff_seconds(0, APIError(503)), 2.0)
        huge = RuntimeError("retry in 3600s")
        self.assertLessEqual(gemini_limits.backoff_seconds(0, huge), gemini_limits.MAX_RETRY_DELAY_SECONDS)

    def test_short_error_truncates_sdk_json(self):
        msg = gemini_limits.short_error(RuntimeError("429 " + "x" * 5000))
        self.assertLessEqual(len(msg), 160)
        self.assertNotIn("\n", gemini_limits.short_error(RuntimeError("a\nb")))

    def test_query_embeddings_skip_pacer_and_are_cached(self):
        from app.rag.embeddings.gemini_provider import GeminiEmbeddingProvider

        fake = FakeEmbeddings()
        gemini_limits.reset_rate_limit(1)
        p = GeminiEmbeddingProvider(api_key="x", dimensions=8, client=pytypes.SimpleNamespace(models=fake))
        with patch("app.rag.gemini_limits.time.sleep", side_effect=AssertionError("query must not wait")):
            p.embed_documents(["only chunk"])
            first = p.embed_query("what dataset?")
            again = p.embed_query("what dataset?")
        self.assertEqual(first, again)
        self.assertEqual([c[2] for c in fake.calls], ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"])  # 2nd query cached

    def test_is_retryable(self):
        self.assertTrue(gemini_limits.is_retryable(APIError(429)))
        self.assertTrue(gemini_limits.is_retryable(APIError(502)))
        self.assertTrue(gemini_limits.is_retryable(TimeoutError()))
        self.assertTrue(gemini_limits.is_retryable(type("ReadTimeout", (Exception,), {})()))
        self.assertFalse(gemini_limits.is_retryable(APIError(400)))
        daily = APIError(429)
        daily.args = ("429 RESOURCE_EXHAUSTED. {'retryDelay': '13636s'}",)
        self.assertFalse(gemini_limits.is_retryable(daily))  # daily cap: fail fast
        self.assertFalse(gemini_limits.is_retryable(ValueError("bad")))


class TestFactoryAndStatus(_GeminiTestBase):
    def test_factory_selects_gemini_and_falls_back_without_key(self):
        from app.config import Settings
        from app.rag.factory import get_embedding_provider, get_llm_provider, provider_status

        s = Settings(embedding_provider="gemini", llm_provider="gemini", gemini_api_key=None)
        self.assertEqual(get_embedding_provider(s).name, "local")  # no key -> graceful fallback
        self.assertEqual(get_llm_provider(s).name, "local")
        st = provider_status(s)
        self.assertFalse(st["embedding_configured"])
        self.assertEqual(st["active_embedding_provider"], "local")
        self.assertEqual(st["embedding_model"], "hashing-bow-v1")

        s2 = Settings(embedding_provider="gemini", llm_provider="gemini", gemini_api_key="k")
        self.assertEqual(get_embedding_provider(s2).name, "gemini")
        self.assertEqual(get_llm_provider(s2).name, "gemini")
        self.assertEqual(get_embedding_provider(s2).dimensions, 768)
        st = provider_status(s2)
        self.assertTrue(st["embedding_configured"] and st["llm_configured"])
        self.assertIn("gemini", st["available_llm_providers"])
        self.assertEqual(st["embedding_model"], "gemini-embedding-001")
        self.assertEqual(st["llm_model"], "gemini-3.8-flash")

    def test_model_overrides_come_from_settings(self):
        from app.config import Settings
        from app.rag.factory import provider_status

        s = Settings(embedding_provider="gemini", llm_provider="gemini", gemini_api_key="k",
                     embedding_model="emb-x", llm_model="llm-y")
        st = provider_status(s)
        self.assertEqual((st["embedding_model"], st["llm_model"]), ("emb-x", "llm-y"))


class _DBTestBase(unittest.TestCase):
    """Documents indexed with various providers in an in-memory SQLite DB."""

    def setUp(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool

        from app.database import Base
        from app.models import Document, ProcessingStatus, RagStatus, User

        self.engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        with self.Session() as db:
            db.add(User(id="u1", email="t@example.com", name="T"))
            rows = [
                ("stale-1", ProcessingStatus.READY, RagStatus.INDEXED, "local"),
                ("stale-2", ProcessingStatus.READY, RagStatus.INDEXED, "openai"),
                ("current", ProcessingStatus.READY, RagStatus.INDEXED, "gemini"),
                ("never", ProcessingStatus.READY, RagStatus.NOT_INDEXED, None),
                ("not-ready", ProcessingStatus.PROCESSING, RagStatus.INDEXED, "local"),
                ("busy", ProcessingStatus.READY, RagStatus.INDEXING, "local"),
            ]
            for doc_id, ps, rs, emb in rows:
                db.add(Document(
                    id=doc_id, filename=f"{doc_id}.pdf", original_filename=f"{doc_id}.pdf",
                    file_size=1, storage_path=f"uploads/{doc_id}.pdf", owner_id="u1",
                    processing_status=ps, rag_status=rs, embedding_provider=emb,
                ))
            db.commit()

    def tearDown(self):
        self.engine.dispose()


class TestStaleDocumentsStatus(_DBTestBase):
    def test_system_rag_reports_models_and_stale_count(self):
        from fastapi.testclient import TestClient

        from app import main
        from app.config import Settings
        from app.database import get_db

        def override_db():
            with self.Session() as db:
                yield db

        main.app.dependency_overrides[get_db] = override_db
        self.addCleanup(main.app.dependency_overrides.clear)
        settings = Settings(embedding_provider="gemini", llm_provider="gemini", gemini_api_key="k")
        with patch.dict(sys.modules, _stub_genai_modules()), patch.object(main, "settings", settings):
            body = TestClient(main.app).get("/api/system/rag").json()

        self.assertEqual(body["embedding_model"], "gemini-embedding-001")
        self.assertEqual(body["llm_model"], "gemini-3.8-flash")
        # stale-1, stale-2, not-ready, busy: indexed with a non-gemini provider.
        self.assertEqual(body["stale_documents"], 4)

    def test_stale_count_failure_degrades_to_null(self):
        from fastapi.testclient import TestClient

        from app import main
        from app.database import get_db

        broken = MagicMock()
        broken.scalar.side_effect = RuntimeError("db down")
        main.app.dependency_overrides[get_db] = lambda: broken
        self.addCleanup(main.app.dependency_overrides.clear)
        resp = TestClient(main.app).get("/api/system/rag")
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(resp.json()["stale_documents"])
        self.assertIn("llm_model", resp.json())


class TestReindexAll(_DBTestBase):
    def _run(self, provider_name="gemini", configured="gemini", index_side_effect=None, argv=()):
        from scripts import reindex_all

        provider = MagicMock()
        provider.name = provider_name
        provider.model = "m"
        seen = []
        inner = index_side_effect or (lambda db, doc, s: {"children": 3})

        def index(db, doc, settings):
            seen.append(doc.id)
            return inner(db, doc, settings)
        with patch.object(reindex_all, "SessionLocal", self.Session), \
             patch.object(reindex_all, "run_migrations"), \
             patch.object(reindex_all, "get_settings", return_value=MagicMock(embedding_provider=configured)), \
             patch.object(reindex_all, "get_embedding_provider", return_value=provider), \
             patch.object(reindex_all, "index_document", index), \
             patch("builtins.print"):
            code = reindex_all.main(list(argv))
        return code, seen

    def test_reindexes_only_ready_stale_documents(self):
        code, indexed = self._run()
        self.assertEqual(code, 0)
        self.assertEqual(sorted(indexed), ["stale-1", "stale-2"])

    def test_dry_run_indexes_nothing(self):
        code, indexed = self._run(argv=["--dry-run"])
        self.assertEqual((code, indexed), (0, []))

    def test_failure_is_rolled_back_and_others_continue(self):
        from app.models import Document

        def flaky(db, doc, settings):
            doc.embedding_provider = "gemini"  # partial write that must not persist
            if doc.id == "stale-1":
                raise RuntimeError("429")
            return {"children": 1}

        code, indexed = self._run(index_side_effect=flaky)
        self.assertEqual(code, 1)
        self.assertEqual(sorted(indexed), ["stale-1", "stale-2"])
        with self.Session() as db:
            self.assertEqual(db.get(Document, "stale-1").embedding_provider, "local")
            self.assertEqual(db.get(Document, "stale-2").embedding_provider, "gemini")

    def test_refuses_when_configured_provider_fell_back(self):
        code, indexed = self._run(provider_name="local", configured="gemini")
        self.assertEqual((code, indexed), (2, []))


if __name__ == "__main__":
    unittest.main()
