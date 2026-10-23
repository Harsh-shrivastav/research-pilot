"""P2: structured comparison (one JSON call per paper) + quote verification.

Offline only: the LLM is a fake with a canned `generate_json`; nothing here
can reach the real Gemini API. Uses a temp-file SQLite DB for the same reason
tests/test_comparison.py does (the hybrid retriever and the per-paper workers
open their own Sessions on the same engine).

Run: python -m unittest tests.test_comparison_structured   (from backend/)
"""
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["DATABASE_URL"] = "sqlite://"
# Env vars beat backend/.env: keep tests on the offline providers so they
# never call a real API, whatever the developer has configured.
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["LLM_PROVIDER"] = "local"
os.environ["GEMINI_API_KEY"] = ""

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.comparison import comparison_service, paper_extractor  # noqa: E402
from app.comparison.dimensions import resolve_dimensions  # noqa: E402
from app.config import Settings  # noqa: E402
from app.database import Base  # noqa: E402
from app.models import (  # noqa: E402
    ComparisonStatus,
    Document,
    DocumentElement,
    Evidence,
    ProcessingStatus,
    RagStatus,
    ResearchWorkspace,
    User,
    WorkspaceDocument,
)
from app.rag.chunking import build_chunks  # noqa: E402
from app.rag.comparison_context import CELL_NOT_FOUND_SENTENCE  # noqa: E402
from app.rag.interfaces import Citation, LLMProvider, RetrievedChunk  # noqa: E402
from app.rag.persistence import persist_chunks  # noqa: E402

# (page, text). Ligature + curly quote + a hyphen variant on purpose: the
# verifier must see through the same normalisation the chunker applies.
PAPERS = {
    "doc-1": [
        (1, "We evaluate on the SQuAD v1.1 dataset, which contains 100,000 questions."),
        (2, "Our encoder uses ﬁne-tuned BERT and it’s a state–of–the–art 85.2 F1 on the test set."),
    ],
    "doc-2": [
        (1, "The training corpus consists of 8 million web pages from outbound links."),
        (2, "We report exact match accuracy as the main evaluation metric."),
    ],
}


class FakeLLM(LLMProvider):
    """`generate_json` answers from a per-document script (looked up by the
    paper title in the prompt); `generate` is the per-cell fallback."""

    name = "gemini"
    model = "fake-model"

    def __init__(self, script):
        self.script = script  # {"Paper One": dict | Exception}
        self.json_calls = []
        self.generate_mock = MagicMock(return_value="Per-cell answer [1]")

    def generate(self, system, messages):
        return self.generate_mock(system, messages)

    def generate_json(self, system, messages, schema):
        text = messages[-1].content
        self.json_calls.append({"system": system, "text": text, "schema": schema})
        for title, response in self.script.items():
            if f"Paper: {title}" in text:
                if isinstance(response, Exception):
                    raise response
                return response
        raise AssertionError("unscripted paper")

    async def stream(self, system, messages):  # pragma: no cover - unused
        yield ""


def field(key, found=True, value="An answer.", quote="", page=1):
    return {"key": key, "found": found, "value": value, "quote": quote, "page": page}


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.engine = create_engine(f"sqlite:///{self._tmp.name}")
        Base.metadata.create_all(bind=self.engine)
        self.db = sessionmaker(bind=self.engine)()
        self._seed()

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()
        os.unlink(self._tmp.name)

    def _seed(self, parent_tokens=1400):
        self.db.add(User(id="user-1", email="t@example.com", name="T"))
        self.db.add(ResearchWorkspace(id="ws-1", name="WS", owner_id="user-1"))
        for n, (doc_id, paragraphs) in enumerate(PAPERS.items(), start=1):
            self.db.add(Document(
                id=doc_id, filename=f"{doc_id}.pdf", original_filename=f"{doc_id}.pdf", file_size=1,
                storage_path=f"uploads/{doc_id}.pdf", owner_id="user-1", title=["Paper One", "Paper Two"][n - 1],
                processing_status=ProcessingStatus.READY, rag_status=RagStatus.INDEXED, chunk_count=2,
            ))
            self.db.add(WorkspaceDocument(workspace_id="ws-1", document_id=doc_id))
            elements = [
                DocumentElement(
                    id=f"{doc_id}-el-{i}", document_id=doc_id, page_number=page, element_type="paragraph",
                    content=text, order_index=i,
                )
                for i, (page, text) in enumerate(paragraphs)
            ]
            self.db.add_all(elements)
            self.db.flush()
            drafts = build_chunks(doc_id, elements, {}, parent_tokens=parent_tokens, child_tokens=320, child_overlap_tokens=60)
            children = [d for d in drafts if d.chunk_level == "child"]
            persist_chunks(self.db, doc_id, drafts, {c.id: [0.1] * 768 for c in children})
        self.db.commit()

    def run_comparison(self, llm, settings=None, dimensions=("dataset", "evaluation_metrics"), retriever=None):
        # These tests are about extraction; keep the (P3) comparability step off the LLM so
        # the call counts they assert stay about extraction only.
        settings = (settings or Settings()).model_copy(update={"comparability_use_llm": False})
        with patch.object(comparison_service, "get_llm_provider", return_value=llm), \
             patch.object(comparison_service, "build_retriever", return_value=retriever or MagicMock()):
            return comparison_service.create_and_run_comparison(
                self.db, "ws-1", "user-1", ["doc-1", "doc-2"], list(dimensions), settings=settings
            )

    @staticmethod
    def cell(matrix, doc_id, key):
        return next(c for c in matrix.cells if c.document_id == doc_id and c.dimension_key == key)


class TestStructuredPath(_Base):
    SCRIPT = {
        "Paper One": {"fields": [
            field("dataset", quote="the SQuAD v1.1 dataset, which contains 100,000 questions", page=1,
                  value="SQuAD v1.1 (100,000 questions)."),
            # curly apostrophe, en-dashes and the fi ligature in the source; model cites the wrong page
            field("evaluation_metrics", quote="fine-tuned BERT and it's a state-of-the-art 85.2 F1", page=1,
                  value="85.2 F1."),
        ]},
        "Paper Two": {"fields": [
            field("dataset", quote="8 million web pages from outbound links", page=1, value="8M web pages."),
            field("evaluation_metrics", quote="exact match accuracy as the main evaluation metric", page=2,
                  value="Exact match accuracy."),
        ]},
    }

    def test_one_call_per_paper_and_every_answer_is_verified_with_evidence(self):
        llm = FakeLLM(self.SCRIPT)
        matrix = self.run_comparison(llm)

        self.assertEqual(matrix.status, ComparisonStatus.COMPLETED)
        self.assertEqual(len(llm.json_calls), 2)  # ONE call per paper, not per cell
        llm.generate_mock.assert_not_called()
        self.assertEqual(len(matrix.cells), 4)
        self.assertTrue(all(c.quote_verified is True and not c.insufficient_evidence for c in matrix.cells))
        self.assertEqual([c.order_index for c in matrix.cells], [0, 1, 2, 3])
        self.assertEqual({c.llm_model for c in matrix.cells}, {"fake-model"})

        # The prompt carries page-labelled text and a schema listing the keys.
        call = next(c for c in llm.json_calls if "Paper: Paper One" in c["text"])  # papers run in parallel: order varies
        self.assertIn("[Pages 1-2]", call["text"])
        self.assertIn("SQuAD v1.1", call["text"])
        self.assertEqual(call["schema"]["properties"]["fields"]["items"]["properties"]["key"]["enum"],
                         ["dataset", "evaluation_metrics"])

        # Citations keep the Citation.to_dict() shape the frontend reads.
        cell = self.cell(matrix, "doc-1", "evaluation_metrics")
        self.assertEqual(set(cell.citations[0]), set(Citation("c", "d", 1, None, "s", 1.0).to_dict()))
        # The model said page 1; the stored text has it on page 2.
        self.assertEqual(cell.citations[0]["page_number"], 2)
        self.assertEqual(cell.value_text, "85.2 F1.")

        # Evidence rows exist, snippet == the quote, attached to a claim.
        evidence = self.db.get(Evidence, cell.citations[0]["evidence_id"])
        self.assertIsNotNone(evidence.claim_id)
        self.assertEqual(evidence.snippet, "fine-tuned BERT and it's a state-of-the-art 85.2 F1")
        self.assertEqual(evidence.document_id, "doc-1")
        self.assertEqual(self.db.query(Evidence).count(), 4)

    def test_invented_quote_is_blanked_when_strict(self):
        script = {"Paper One": {"fields": [
            field("dataset", quote="We evaluate on ImageNet with 14 million pictures", value="ImageNet."),
            field("evaluation_metrics", found=False, value="", quote="", page=0),
        ]}, "Paper Two": self.SCRIPT["Paper Two"]}
        matrix = self.run_comparison(FakeLLM(script), settings=Settings(comparison_strict_quotes=True))

        bad = self.cell(matrix, "doc-1", "dataset")
        self.assertEqual(bad.value_text, CELL_NOT_FOUND_SENTENCE)
        self.assertTrue(bad.insufficient_evidence)
        self.assertIs(bad.quote_verified, False)
        self.assertEqual(bad.citations, [])
        # Only Paper Two's two verified cells produced evidence.
        self.assertEqual(self.db.query(Evidence).count(), 2)

    def test_invented_quote_keeps_value_but_flags_it_when_not_strict(self):
        script = {"Paper One": {"fields": [
            field("dataset", quote="We evaluate on ImageNet with 14 million pictures", value="ImageNet."),
            field("evaluation_metrics", quote="", value="Some metric."),  # no quote at all
        ]}, "Paper Two": self.SCRIPT["Paper Two"]}
        matrix = self.run_comparison(FakeLLM(script), settings=Settings(comparison_strict_quotes=False))

        for key in ("dataset", "evaluation_metrics"):
            cell = self.cell(matrix, "doc-1", key)
            self.assertIn(cell.value_text, ("ImageNet.", "Some metric."))
            self.assertTrue(cell.insufficient_evidence)
            self.assertIs(cell.quote_verified, False)
            self.assertEqual(cell.citations, [])

    def test_not_found_and_omitted_fields_read_not_found(self):
        script = {"Paper One": {"fields": [field("dataset", found=False, value="", quote="", page=0)]},  # metric omitted
                  "Paper Two": self.SCRIPT["Paper Two"]}
        matrix = self.run_comparison(FakeLLM(script))
        for key in ("dataset", "evaluation_metrics"):
            cell = self.cell(matrix, "doc-1", key)
            self.assertEqual(cell.value_text, CELL_NOT_FOUND_SENTENCE)
            self.assertTrue(cell.insufficient_evidence)
            self.assertIsNone(cell.quote_verified)  # not applicable: nothing to verify

    def test_one_failing_paper_falls_back_to_per_cell_for_that_paper_only(self):
        script = {"Paper One": self.SCRIPT["Paper One"], "Paper Two": ValueError("invalid JSON twice")}
        retriever = MagicMock()
        retriever.retrieve.return_value = [RetrievedChunk(
            chunk_id="c1", parent_chunk_id=None, document_id="doc-2", page_number=1, section_id=None,
            section_title=None, child_text="8 million web pages", context_text="8 million web pages", score=0.9,
        )]
        llm = FakeLLM(script)
        with self.assertLogs("app.comparison.comparison_service", "WARNING"):
            matrix = self.run_comparison(llm, retriever=retriever)

        self.assertEqual(matrix.status, ComparisonStatus.COMPLETED)
        self.assertEqual(llm.generate_mock.call_count, 2)  # Paper Two's two cells, per-cell
        for key in ("dataset", "evaluation_metrics"):
            self.assertTrue(self.cell(matrix, "doc-1", key).quote_verified)      # structured
            fallback = self.cell(matrix, "doc-2", key)
            self.assertEqual(fallback.value_text, "Per-cell answer [1]")        # per-cell
            self.assertIsNone(fallback.quote_verified)
        self.assertEqual([c.order_index for c in matrix.cells], [0, 1, 2, 3])

    def test_provider_without_json_support_falls_back_to_per_cell(self):
        local = MagicMock(spec=LLMProvider)
        local.name, local.model = "local", "extractive-v1"
        local.generate_json.side_effect = NotImplementedError("no json")
        local.generate.return_value = "Per-cell answer [1]"
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        with self.assertLogs("app.comparison.comparison_service", "WARNING"):
            matrix = self.run_comparison(local, settings=Settings(comparison_mode="structured"), retriever=retriever)
        self.assertEqual(matrix.status, ComparisonStatus.COMPLETED)
        self.assertEqual(local.generate.call_count, 4)


class TestModeSwitch(unittest.TestCase):
    def test_auto_is_structured_only_for_gemini(self):
        gem, loc = MagicMock(), MagicMock()
        gem.name, loc.name = "gemini", "local"
        resolve = comparison_service.resolve_comparison_mode
        self.assertEqual(resolve(Settings(), gem), "structured")
        self.assertEqual(resolve(Settings(), loc), "per_cell")
        self.assertEqual(resolve(Settings(comparison_mode="per_cell"), gem), "per_cell")
        self.assertEqual(resolve(Settings(comparison_mode="structured"), loc), "structured")
        self.assertEqual(resolve(Settings(comparison_mode="Per_Cell "), gem), "per_cell")

    def test_blank_env_value_means_auto(self):
        with patch.dict(os.environ, {"COMPARISON_MODE": ""}):
            self.assertIsNone(Settings().comparison_mode)

    def test_per_cell_mode_never_calls_generate_json(self):
        llm = FakeLLM({})
        with patch.object(comparison_service, "get_llm_provider", return_value=llm):
            self.assertEqual(comparison_service.resolve_comparison_mode(Settings(comparison_mode="per_cell"), llm), "per_cell")


class TestContextSelection(_Base):
    def _seed(self, parent_tokens=1400):
        super()._seed(parent_tokens=8)  # tiny parents: each paragraph becomes its own parent chunk

    def test_full_paper_when_it_fits(self):
        material = paper_extractor.load_paper_material(self.db, "doc-1")
        self.assertEqual(len(material.parents), 2)
        doc = self.db.get(Document, "doc-1")
        dims = resolve_dimensions(["dataset"])
        parents, scope = paper_extractor.select_context(self.db, Settings(), doc, material, dims)
        self.assertEqual((scope, len(parents)), ("full", 2))
        text = paper_extractor.build_paper_context(parents)
        self.assertLess(text.index("[Page 1]"), text.index("[Page 2]"))  # page order, each block labelled

    def test_long_paper_uses_the_top_retrieved_parents_only(self):
        material = paper_extractor.load_paper_material(self.db, "doc-1")
        doc = self.db.get(Document, "doc-1")
        dims = resolve_dimensions(["dataset", "evaluation_metrics"])
        page2_parent = next(p for p in material.parents if p.page == 2)
        retriever = MagicMock()
        retriever.retrieve.return_value = [RetrievedChunk(
            chunk_id="x", parent_chunk_id=page2_parent.id, document_id="doc-1", page_number=2, section_id=None,
            section_title=None, child_text="t", context_text="t", score=0.5,
        )]
        settings = Settings(comparison_full_paper_max_chars=50)  # far below the paper's length
        parents, scope = paper_extractor.select_context(self.db, settings, doc, material, dims, retriever)
        self.assertEqual(scope, "retrieved")
        self.assertEqual([p.id for p in parents], [page2_parent.id])  # de-duplicated across dimensions
        self.assertEqual(retriever.retrieve.call_count, 2)  # one retrieval per dimension

    def test_extraction_with_long_paper_sends_only_selected_text(self):
        doc = self.db.get(Document, "doc-1")
        material = paper_extractor.load_paper_material(self.db, "doc-1")
        page2_parent = next(p for p in material.parents if p.page == 2)
        retriever = MagicMock()
        retriever.retrieve.return_value = [RetrievedChunk(
            chunk_id="x", parent_chunk_id=page2_parent.id, document_id="doc-1", page_number=2, section_id=None,
            section_title=None, child_text="t", context_text="t", score=0.5,
        )]
        llm = FakeLLM({"Paper One": {"fields": [field("dataset", found=False, value="", quote="", page=0)]}})
        out = paper_extractor.extract_paper_dimensions(
            self.db, Settings(comparison_full_paper_max_chars=50), llm, doc, resolve_dimensions(["dataset"]),
            material=material, retriever=retriever,
        )
        sent = llm.json_calls[0]["text"]
        self.assertIn("85.2 F1", sent)                 # page-2 paragraph
        self.assertNotIn("SQuAD v1.1", sent)          # page-1 paragraph was not selected
        self.assertEqual(out["dataset"]["found"], False)

    def test_retrieval_failure_on_a_long_paper_raises_for_fallback(self):
        doc = self.db.get(Document, "doc-1")
        material = paper_extractor.load_paper_material(self.db, "doc-1")
        retriever = MagicMock()
        retriever.retrieve.side_effect = RuntimeError("down")
        with self.assertLogs("app.comparison.paper_extractor", "WARNING"), \
             self.assertRaises(paper_extractor.ExtractionError):
            paper_extractor.select_context(
                self.db, Settings(comparison_full_paper_max_chars=50), doc, material, resolve_dimensions(["dataset"]), retriever
            )


class TestQuoteVerification(_Base):
    def setUp(self):
        super().setUp()
        self.material = paper_extractor.load_paper_material(self.db, "doc-1")

    def test_normalisation_sees_through_ligatures_quotes_dashes_and_whitespace(self):
        norm = paper_extractor.normalize_for_match
        self.assertEqual(norm("ﬁne‑tuned  it’s\n a “quote” – x"), 'fine-tuned it\'s a "quote" - x')

    def test_exact_substring_verified_and_page_comes_from_stored_text(self):
        check = paper_extractor.check_quote(self.material, "state-of-the-art 85.2 F1 on the test set", cited_page=1)
        self.assertTrue(check.verified)
        self.assertEqual(check.page, 2)
        self.assertTrue(check.page_corrected)
        self.assertIsNotNone(check.chunk)

    def test_not_in_paper_or_too_short_or_empty_is_not_verified(self):
        for quote in ("a completely invented sentence about vision transformers", "SQuAD", "", "   "):
            self.assertFalse(paper_extractor.check_quote(self.material, quote, 1).verified, quote)

    def test_case_must_match(self):
        self.assertFalse(paper_extractor.check_quote(self.material, "WE EVALUATE ON THE SQUAD V1.1 DATASET", 1).verified)

    def test_quote_across_a_page_break_is_accepted(self):
        quote = "which contains 100,000 questions. Our encoder uses fine-tuned BERT"
        check = paper_extractor.check_quote(self.material, quote, cited_page=1)
        self.assertTrue(check.verified)
        self.assertEqual(check.page, 1)

    def test_response_normalisation(self):
        dims = resolve_dimensions(["dataset", "baselines"])
        out = paper_extractor.normalise_response({"fields": [
            {"key": "dataset", "found": True, "value": " V ", "quote": " a   b ", "page": "3"},
            {"key": "dataset", "found": True, "value": "dup", "quote": "x", "page": 1},       # duplicate ignored
            {"key": "bogus", "found": True, "value": "x", "quote": "y", "page": 1},           # unknown key ignored
        ]}, dims)
        self.assertEqual(out["dataset"], {"found": True, "value": "V", "quote": "a b", "page": 3})
        self.assertEqual(out["baselines"]["found"], False)
        with self.assertRaises(paper_extractor.ExtractionError):
            paper_extractor.normalise_response({"nope": 1}, dims)


class TestColumnTolerantVerification(unittest.TestCase):
    """Two-column PDFs are often extracted with the columns interleaved line
    by line. This is real text from the DPR paper's stored elements."""

    STORED = (
        "Our Dense Passage Retriever (DPR) is exception- k , a retriever can be evaluated in isolation on top-k "
        "ally strong. It not only outperforms BM25 by a retrieval accuracy , which is the fraction of ques- "
        "large margin (65.2% vs. 42.9% in Top-5 accuracy), tions for which C F contains a span that answers the "
        "but also results in a substantial improvement on the end-to-end QA accuracy compared to ORQA "
        "(41.5% vs. 33.3%) in the open Natural Questions setting"
    )
    # What a model reading that text quotes: the correct reading order.
    QUOTE = (
        "Our Dense Passage Retriever (DPR) is exceptionally strong. It not only outperforms BM25 by a large margin "
        "(65.2% vs. 42.9% in Top-5 accuracy), but also results in a substantial improvement on the end-to-end QA "
        "accuracy compared to ORQA (41.5% vs. 33.3%) in the open Natural Questions setting"
    )

    def setUp(self):
        chunk = paper_extractor.ChunkInfo(
            id="p1", parent_id=None, level="parent", page=1, end_page=1, order=0, text=self.STORED,
            section_id=None, section_title="Results", element_types=("paragraph",), bbox=None,
        )
        element = paper_extractor.ElementInfo(page=1, text=paper_extractor.normalize_for_match(self.STORED), bbox=None)
        self.material = paper_extractor.PaperMaterial([chunk], [], [element])

    def test_interleaved_columns_pass_only_the_tolerant_tier(self):
        exact_only = paper_extractor.check_quote(self.material, self.QUOTE, 1, tolerant=False)
        self.assertFalse(exact_only.verified)

        check = paper_extractor.check_quote(self.material, self.QUOTE, 1)
        self.assertTrue(check.verified)
        self.assertEqual((check.match, check.page), ("column-tolerant", 1))
        self.assertEqual(check.chunk.id, "p1")

    def test_a_literal_substring_is_reported_as_exact(self):
        check = paper_extractor.check_quote(self.material, "the fraction of ques- large margin", 1)
        self.assertEqual(check.match, "exact")

    def test_paraphrases_and_inventions_do_not_pass(self):
        for fake in (
            # same facts, different words
            "Our dense retriever clearly beats the BM25 baseline by a wide margin in top-5 accuracy and improves "
            "end-to-end question answering",
            # invented
            "DPR was trained on 14 million images from ImageNet and reaches 99% accuracy on every task we evaluated",
            # real words, scrambled order
            "setting Questions Natural open the in accuracy QA end-to-end the on improvement substantial a results also but",
            # a real opening followed by an invented tail
            "Our Dense Passage Retriever (DPR) is exceptionally strong. It was also pre-trained on web-scale video "
            "corpora and distilled into a tiny student network for mobile deployment",
        ):
            self.assertFalse(paper_extractor.check_quote(self.material, fake, 1).verified, fake[:50])

    def test_short_quotes_are_not_matched_loosely(self):
        self.assertFalse(paper_extractor.check_quote(self.material, "DPR is exceptionally strong", 1).verified)

    def test_words_far_apart_do_not_count_as_a_passage(self):
        padding = " ".join(f"filler{i}" for i in range(200))
        stored = f"Alpha beta gamma delta {padding} epsilon zeta eta theta"
        chunk = paper_extractor.ChunkInfo("p", None, "parent", 1, 1, 0, stored, None, None, (), None)
        element = paper_extractor.ElementInfo(1, paper_extractor.normalize_for_match(stored), None)
        material = paper_extractor.PaperMaterial([chunk], [], [element])
        self.assertFalse(paper_extractor.check_quote(material, "Alpha beta gamma delta epsilon zeta eta theta", 1).verified)

    def test_service_uses_the_tolerant_tier_by_default_and_honours_the_switch(self):
        words = self.QUOTE
        self.assertTrue(paper_extractor.check_quote(self.material, words, 1, tolerant=Settings().comparison_tolerant_quotes).verified)
        self.assertFalse(paper_extractor.check_quote(
            self.material, words, 1, tolerant=Settings(comparison_tolerant_quotes=False).comparison_tolerant_quotes).verified)


if __name__ == "__main__":
    unittest.main()
