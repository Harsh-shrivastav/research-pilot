"""P3: comparability check - rules engine, the Gemini step's validation, and
the service / endpoint wiring. Offline only: the LLM is a fake.

Run: python -m unittest tests.test_comparability   (from backend/)
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))  # sibling test helpers
os.environ["DATABASE_URL"] = "sqlite://"
# Env vars beat backend/.env: keep tests on the offline providers so they
# never call a real API, whatever the developer has configured.
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["LLM_PROVIDER"] = "local"
os.environ["GEMINI_API_KEY"] = ""

from app.comparison import comparability as cmp  # noqa: E402
from app.comparison import comparison_service  # noqa: E402
from app.comparison.comparability import CellInput, check_comparability  # noqa: E402
from app.config import Settings  # noqa: E402
from app.models import ComparisonMatrix, ComparisonStatus  # noqa: E402
from app.rag.interfaces import LLMProvider  # noqa: E402
from test_comparison_structured import PAPERS, FakeLLM, _Base, field  # noqa: E402

DOCS2 = [("A", "Paper Alpha"), ("B", "Paper Beta")]
DOCS3 = [("A", "Paper Alpha"), ("B", "Paper Beta"), ("C", "Paper Gamma")]
DIMS = ["research_problem", "dataset", "evaluation_metrics", "main_results"]


def cells(doc, usable=True, **by_dim):
    return [CellInput(doc, key, text, usable and bool(text)) for key, text in by_dim.items()]


def check(cell_list, docs=DOCS2, dims=DIMS, llm=None, use_llm=False):
    return check_comparability(cell_list, dims, docs, llm=llm, use_llm=use_llm)


class ScriptedJsonLLM(LLMProvider):
    """Gemini stand-in for the comparability step: returns a canned object."""

    name = "gemini"
    model = "fake"

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate(self, system, messages):  # pragma: no cover - unused
        return ""

    async def stream(self, system, messages):  # pragma: no cover - unused
        yield ""

    def generate_json(self, system, messages, schema):
        self.calls.append({"system": system, "text": messages[-1].content, "schema": schema})
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def ref(doc, key):
    return {"document_id": doc, "dimension_key": key}


class TestSpecExample(unittest.TestCase):
    def test_plantvillage_vs_own_lab_dataset_is_not_comparable(self):
        out = check(
            cells("A", research_problem="Classify plant leaf diseases from images.", dataset="Evaluated on the PlantVillage dataset.",
                  evaluation_metrics="Accuracy.", main_results="Reaches 91% accuracy on PlantVillage.")
            + cells("B", research_problem="Classify plant leaf diseases from images.",
                    dataset="Evaluated on its own lab-collected dataset of 4,000 leaf images.",
                    evaluation_metrics="Accuracy.", main_results="Reaches 94% accuracy on its own lab-collected dataset.")
        )
        self.assertEqual(out["overall"], "not_comparable")
        self.assertEqual(out["checked_by"], "rules")
        (issue,) = out["issues"]
        self.assertEqual(issue["type"], "dataset")
        self.assertEqual(sorted(issue["document_ids"]), ["A", "B"])
        self.assertEqual(sorted((r["document_id"], r["dimension_key"]) for r in issue["cell_refs"]),
                         [("A", "dataset"), ("B", "dataset")])
        self.assertIn("PlantVillage", issue["description"])
        self.assertIn("its own dataset", issue["description"])
        # one plain sentence a student could paste
        self.assertTrue(out["caveat"].endswith("."))
        self.assertNotIn("\n", out["caveat"])
        self.assertEqual(len(__import__("re").findall(r"[.!?](?:\s|$)", out["caveat"])), 1)

    def test_same_dataset_and_metric_is_comparable(self):
        out = check(
            cells("A", dataset="PlantVillage", evaluation_metrics="Accuracy (top-1)", main_results="91%")
            + cells("B", dataset="The PlantVillage dataset", evaluation_metrics="classification accuracy", main_results="94%")
        )
        self.assertEqual((out["overall"], out["issues"]), ("comparable", []))
        self.assertIn("PlantVillage", out["caveat"])

    def test_output_shape_is_exactly_the_stored_contract(self):
        out = check(cells("A", dataset="MNIST", evaluation_metrics="accuracy") + cells("B", dataset="CIFAR-10", evaluation_metrics="accuracy"))
        self.assertEqual(set(out), {"overall", "issues", "caveat", "checked_by"})
        self.assertIn(out["overall"], cmp.OVERALL_VALUES)
        for issue in out["issues"]:
            self.assertEqual(set(issue), {"type", "description", "document_ids", "cell_refs"})
            self.assertIn(issue["type"], cmp.ISSUE_TYPES)
            self.assertTrue(all(set(r) == {"document_id", "dimension_key"} for r in issue["cell_refs"]))


class TestNeverGuess(unittest.TestCase):
    def test_missing_or_insufficient_cell_makes_the_aspect_unknown(self):
        for dataset_b in (dict(dataset="The paper does not provide sufficient information.", usable=False), dict(dataset="", usable=False)):
            usable = dataset_b.pop("usable")
            out = check(
                cells("A", dataset="PlantVillage", evaluation_metrics="accuracy")
                + cells("B", usable=usable, dataset=dataset_b["dataset"]) + cells("B", evaluation_metrics="accuracy")
            )
            self.assertEqual(out["overall"], "unknown")
            self.assertEqual(out["issues"], [])
            self.assertIn("dataset", out["caveat"])

    def test_an_absent_cell_is_not_a_difference(self):
        out = check(cells("A", dataset="PlantVillage", evaluation_metrics="accuracy"), docs=DOCS2)  # B has no cells at all
        self.assertEqual((out["overall"], out["issues"]), ("unknown", []))

    def test_criterion_not_in_the_matrix_is_unknown_with_a_reason(self):
        out = check(cells("A", methodology="CNN") + cells("B", methodology="CNN"), dims=["methodology"])
        self.assertEqual(out["overall"], "unknown")
        self.assertIn("dataset", out["caveat"])

    def test_definite_mismatch_is_still_reported_when_another_paper_lacks_evidence(self):
        out = check(
            cells("A", dataset="MNIST", evaluation_metrics="accuracy") + cells("B", dataset="CIFAR-10", evaluation_metrics="accuracy")
            + cells("C", usable=False, dataset="not found"),
            docs=DOCS3,
        )
        self.assertEqual(out["overall"], "not_comparable")
        self.assertEqual(sorted(out["issues"][0]["document_ids"]), ["A", "B"])  # C is not blamed

    def test_dataset_cell_naming_nothing_is_not_decided_by_rules(self):
        out = check(cells("A", dataset="A large corpus of text.", evaluation_metrics="accuracy")
                    + cells("B", dataset="Evaluated on PlantVillage.", evaluation_metrics="accuracy"))
        self.assertEqual(out["overall"], "unknown")
        self.assertEqual(out["issues"], [])

    def test_no_metric_named_is_not_decided_by_rules(self):
        out = check(cells("A", dataset="MNIST", evaluation_metrics="Standard measures.") + cells("B", dataset="MNIST", evaluation_metrics="accuracy"))
        self.assertEqual(out["overall"], "unknown")


class TestDatasetRules(unittest.TestCase):
    def relation(self, a, b):
        return check(cells("A", dataset=a, evaluation_metrics="accuracy") + cells("B", dataset=b, evaluation_metrics="accuracy"))

    def test_aliases_and_punctuation(self):
        for a, b in [("CIFAR-10", "cifar10"), ("CIFAR-10", "the CIFAR 10 benchmark"), ("ImageNet", "ILSVRC 2012"),
                     ("ImageNet-1k", "ImageNet"), ("Natural Questions", "NQ"), ("SST-2", "SST2"), ("WikiText-103", "wikitext103")]:
            self.assertEqual(self.relation(a, b)["overall"], "comparable", (a, b))

    def test_look_alikes_are_not_merged(self):
        for a, b in [("CIFAR-10", "CIFAR-100"), ("ImageNet", "ImageNet-21k"), ("MNIST", "Fashion-MNIST"),
                     ("SQuAD v1.1", "SQuAD 2.0"), ("ImageNet", "ImageNet-C"), ("GLUE", "SuperGLUE")]:
            out = self.relation(a, b)
            self.assertEqual(out["overall"], "not_comparable", (a, b))
            self.assertEqual(out["issues"][0]["type"], "dataset")

    def test_squad_without_a_version_matches_a_versioned_mention(self):
        self.assertEqual(self.relation("SQuAD", "SQuAD v1.1")["overall"], "comparable")

    def test_partial_overlap_is_partially_comparable(self):
        out = self.relation("ImageNet and CIFAR-10", "ImageNet")
        self.assertEqual(out["overall"], "partially_comparable")
        self.assertIn("only partly overlap", out["issues"][0]["description"])

    def test_own_data_next_to_a_public_set_is_partial_and_alone_is_different(self):
        self.assertEqual(self.relation("ImageNet and our own collected dataset", "ImageNet")["overall"], "partially_comparable")
        self.assertEqual(self.relation("We collected a new dataset of 5,000 recordings.", "LibriSpeech")["overall"], "not_comparable")
        self.assertEqual(self.relation("Their own in-house dataset.", "Their own in-house dataset.")["overall"], "not_comparable")

    def test_wikipedia_dump_cells_from_real_papers(self):
        out = self.relation(
            "The paper uses a single Wikipedia dump (December 2018) and evaluates on Natural Questions, TriviaQA, WebQuestions and CuratedTREC.",
            "We use five QA datasets: Natural Questions (NQ), TriviaQA, WebQuestions (WQ), CuratedTREC (TREC), and SQuAD v1.1.",
        )
        self.assertEqual(out["overall"], "partially_comparable")

    def test_tools_and_models_are_not_mistaken_for_datasets(self):
        out = self.relation("Evaluated on MNIST using PyTorch and TensorFlow on a GPU.", "MNIST")
        self.assertEqual(out["overall"], "comparable")


class TestMetricRules(unittest.TestCase):
    def relation(self, a, b, dataset="MNIST"):
        return check(cells("A", dataset=dataset, evaluation_metrics=a) + cells("B", dataset=dataset, evaluation_metrics=b))

    def test_aliases(self):
        for a, b in [("F1-score", "F1"), ("F-measure", "f1 score"), ("AUC", "AUROC"), ("ROC-AUC", "area under the ROC curve"),
                     ("Exact Match (EM)", "exact match"), ("top-1 accuracy", "accuracy"), ("PPL", "perplexity"),
                     ("Recall@5", "recall at 5"), ("mAP", "mean average precision")]:
            self.assertEqual(self.relation(a, b)["overall"], "comparable", (a, b))

    def test_different_metrics_are_not_comparable(self):
        for a, b in [("accuracy", "F1"), ("BLEU", "ROUGE"), ("perplexity", "accuracy"), ("top-5 accuracy", "accuracy"),
                     ("exact match", "F1")]:
            out = self.relation(a, b)
            self.assertEqual(out["overall"], "not_comparable", (a, b))
            self.assertEqual(out["issues"][0]["type"], "metric")

    def test_variants_of_one_family_are_partial(self):
        for a, b in [("macro-F1", "micro-F1"), ("ROUGE-1", "ROUGE-L"), ("BLEU", "SacreBLEU")]:
            self.assertEqual(self.relation(a, b)["overall"], "partially_comparable", (a, b))

    def test_shared_metric_among_several_is_partial(self):
        self.assertEqual(self.relation("accuracy and F1", "accuracy")["overall"], "partially_comparable")

    def test_exact_match_accuracy_is_not_plain_accuracy(self):
        self.assertEqual(self.relation("Exact Match (EM) accuracy", "accuracy")["overall"], "not_comparable")


class TestTaskSplitSetup(unittest.TestCase):
    def test_different_tasks_are_not_comparable(self):
        out = check(
            cells("A", research_problem="Improving machine translation quality.", dataset="WMT14", evaluation_metrics="BLEU")
            + cells("B", research_problem="Abstractive summarization of news.", dataset="WMT14", evaluation_metrics="BLEU")
        )
        self.assertEqual(out["overall"], "not_comparable")
        self.assertEqual([i["type"] for i in out["issues"]], ["task"])

    def test_a_shared_task_family_is_enough(self):
        out = check(
            cells("A", research_problem="Open-domain question answering with dense retrieval.", dataset="Natural Questions", evaluation_metrics="exact match")
            + cells("B", research_problem="Question answering over documents.", dataset="Natural Questions", evaluation_metrics="exact match")
        )
        self.assertEqual(out["overall"], "comparable")

    def test_zero_shot_vs_fine_tuned_is_a_setup_caveat(self):
        out = check(
            cells("A", dataset="WikiText-103", evaluation_metrics="perplexity", methodology="Evaluated zero-shot without any fine-tuning.")
            + cells("B", dataset="WikiText-103", evaluation_metrics="perplexity", methodology="The model is fine-tuned on the training split."),
            dims=DIMS + ["methodology"],
        )
        self.assertEqual(out["overall"], "partially_comparable")
        (issue,) = out["issues"]
        self.assertEqual(issue["type"], "setup")
        self.assertEqual({r["dimension_key"] for r in issue["cell_refs"]}, {"methodology"})

    def test_test_set_vs_validation_set_is_a_split_issue(self):
        out = check(
            cells("A", dataset="MNIST", evaluation_metrics="accuracy", main_results="98% accuracy on the test set.")
            + cells("B", dataset="MNIST", evaluation_metrics="accuracy", main_results="99% accuracy on the validation set."),
        )
        self.assertEqual(out["overall"], "partially_comparable")
        self.assertEqual(out["issues"][0]["type"], "split")

    def test_5_fold_vs_10_fold_differ_but_a_silent_paper_is_not_penalised(self):
        two = check(cells("A", dataset="MNIST", evaluation_metrics="accuracy", main_results="5-fold cross-validation")
                    + cells("B", dataset="MNIST", evaluation_metrics="accuracy", main_results="10-fold cross-validation"))
        self.assertEqual(two["issues"][0]["type"], "split")
        one = check(cells("A", dataset="MNIST", evaluation_metrics="accuracy", main_results="5-fold cross-validation")
                    + cells("B", dataset="MNIST", evaluation_metrics="accuracy", main_results="98%"))
        self.assertEqual((one["overall"], one["issues"]), ("comparable", []))


class TestThreePapers(unittest.TestCase):
    def test_two_of_three_sharing_a_dataset_is_partial(self):
        out = check(
            cells("A", dataset="MNIST", evaluation_metrics="accuracy") + cells("B", dataset="MNIST", evaluation_metrics="accuracy")
            + cells("C", dataset="CIFAR-10", evaluation_metrics="accuracy"), docs=DOCS3)
        self.assertEqual(out["overall"], "partially_comparable")
        self.assertEqual(sorted(out["issues"][0]["document_ids"]), ["A", "B", "C"])

    def test_all_three_different_is_not_comparable(self):
        out = check(
            cells("A", dataset="MNIST", evaluation_metrics="accuracy") + cells("B", dataset="CIFAR-10", evaluation_metrics="accuracy")
            + cells("C", dataset="SVHN", evaluation_metrics="accuracy"), docs=DOCS3)
        self.assertEqual(out["overall"], "not_comparable")

    def test_duplicate_titles_get_distinct_labels(self):
        out = check(cells("A", dataset="MNIST", evaluation_metrics="accuracy") + cells("B", dataset="SVHN", evaluation_metrics="accuracy"),
                    docs=[("A", "Same Title"), ("B", "Same Title")])
        self.assertIn("Same Title (2)", out["issues"][0]["description"])


class TestGeminiStep(unittest.TestCase):
    AMBIGUOUS = (cells("A", dataset="A large corpus of scraped text.", evaluation_metrics="accuracy", main_results="Strong results.")
                 + cells("B", dataset="Evaluated on PlantVillage.", evaluation_metrics="accuracy", main_results="94% accuracy."))

    def llm(self, **overrides):
        reply = {
            "aspects": [{"aspect": "dataset", "verdict": "different", "description": "Paper Alpha uses a scraped text corpus while Paper Beta uses PlantVillage.",
                         "cell_refs": [ref("A", "dataset"), ref("B", "dataset")]}],
            "caveat": "Paper Alpha and Paper Beta evaluate on different datasets, so their results should not be compared directly.",
        }
        reply.update(overrides)
        return ScriptedJsonLLM(reply)

    def test_rules_alone_leave_an_ambiguous_aspect_unknown(self):
        out = check(self.AMBIGUOUS)
        self.assertEqual((out["overall"], out["checked_by"]), ("unknown", "rules"))

    def test_gemini_resolves_only_the_ambiguous_aspect_and_code_computes_overall(self):
        llm = self.llm()
        out = check(self.AMBIGUOUS, llm=llm, use_llm=True)
        self.assertEqual((out["overall"], out["checked_by"]), ("not_comparable", "gemini"))
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.calls[0]["schema"]["properties"]["aspects"]["items"]["properties"]["aspect"]["enum"], ["dataset"])
        self.assertEqual(out["caveat"], "Paper Alpha and Paper Beta evaluate on different datasets, so their results should not be compared directly.")
        self.assertEqual(out["issues"][0]["cell_refs"], [ref("A", "dataset"), ref("B", "dataset")])

    def test_the_prompt_contains_only_cell_texts_and_the_hard_rules(self):
        llm = self.llm()
        check(self.AMBIGUOUS, llm=llm, use_llm=True)
        prompt = llm.calls[0]["text"]
        self.assertIn("A large corpus of scraped text.", prompt)
        self.assertIn("cell(document_id=B, dimension_key=dataset)", prompt)
        system = llm.calls[0]["system"]
        for phrase in ("ONLY the", "unknown", "cell_refs", "ONE plain sentence"):
            self.assertIn(phrase, system)

    def test_gemini_is_not_asked_when_rules_are_decisive_or_use_llm_is_off(self):
        decisive = cells("A", dataset="MNIST", evaluation_metrics="accuracy") + cells("B", dataset="SVHN", evaluation_metrics="accuracy")
        llm = self.llm()
        check(decisive, llm=llm, use_llm=True)
        check(self.AMBIGUOUS, llm=llm, use_llm=False)
        self.assertEqual(llm.calls, [])

    def test_insufficient_cells_are_never_sent_and_never_judged(self):
        llm = self.llm()
        out = check(cells("A", dataset="A large corpus.", evaluation_metrics="accuracy")
                    + cells("B", usable=False, dataset="The paper does not provide sufficient information.") + cells("B", evaluation_metrics="accuracy"),
                    llm=llm, use_llm=True)
        self.assertEqual(llm.calls, [])  # a missing dataset cell is "unknown" before the LLM is ever consulted
        self.assertEqual(out["overall"], "unknown")

    def test_invented_cell_refs_drop_the_verdict(self):
        bad = {"aspects": [{"aspect": "dataset", "verdict": "different", "description": "They differ.",
                            "cell_refs": [ref("A", "dataset"), ref("ZZZ", "dataset"), ref("B", "nonexistent")]}], "caveat": "x"}
        out = check(self.AMBIGUOUS, llm=ScriptedJsonLLM(bad), use_llm=True)
        self.assertEqual((out["overall"], out["issues"], out["checked_by"]), ("unknown", [], "rules"))  # only one real ref -> unsupported

    def test_a_different_verdict_needs_cells_from_two_papers(self):
        one_side = {"aspects": [{"aspect": "dataset", "verdict": "different", "description": "They differ.", "cell_refs": [ref("A", "dataset")]}], "caveat": "x"}
        self.assertEqual(check(self.AMBIGUOUS, llm=ScriptedJsonLLM(one_side), use_llm=True)["overall"], "unknown")

    def test_verdict_without_description_or_refs_is_dropped(self):
        for item in ({"aspect": "dataset", "verdict": "different", "description": "", "cell_refs": [ref("A", "dataset"), ref("B", "dataset")]},
                     {"aspect": "dataset", "verdict": "partial", "description": "Overlap.", "cell_refs": []}):
            out = check(self.AMBIGUOUS, llm=ScriptedJsonLLM({"aspects": [item], "caveat": "x"}), use_llm=True)
            self.assertEqual((out["overall"], out["issues"]), ("unknown", []))

    def test_unknown_verdict_stays_unknown(self):
        out = check(self.AMBIGUOUS, llm=self.llm(aspects=[{"aspect": "dataset", "verdict": "unknown", "description": "", "cell_refs": []}]), use_llm=True)
        self.assertEqual(out["overall"], "unknown")

    def test_same_verdict_with_support_makes_it_comparable(self):
        same = {"aspects": [{"aspect": "dataset", "verdict": "same", "description": "Both describe the same corpus.",
                             "cell_refs": [ref("A", "dataset"), ref("B", "dataset")]}], "caveat": "ignored"}
        out = check(self.AMBIGUOUS, llm=ScriptedJsonLLM(same), use_llm=True)
        self.assertEqual(out["overall"], "comparable")  # metric is "accuracy" for both
        self.assertNotEqual(out["caveat"], "ignored")  # LLM caveat only used for a warning backed by its own issue

    def test_unknown_aspect_or_garbage_is_ignored(self):
        for reply in ({"aspects": [{"aspect": "colour", "verdict": "different", "description": "x", "cell_refs": []}], "caveat": "x"},
                      {"aspects": "nope"}, {"caveat": "x"}, ["not", "a", "dict"], None):
            out = check(self.AMBIGUOUS, llm=ScriptedJsonLLM(reply), use_llm=True)
            self.assertEqual((out["overall"], out["checked_by"]), ("unknown", "rules"))

    def test_llm_failure_degrades_to_rules(self):
        out = check(self.AMBIGUOUS, llm=ScriptedJsonLLM(RuntimeError("503")), use_llm=True)
        self.assertEqual((out["overall"], out["checked_by"]), ("unknown", "rules"))

    def test_a_multi_sentence_caveat_is_replaced_by_the_template(self):
        out = check(self.AMBIGUOUS, llm=self.llm(caveat="They differ. Do not compare them. Really."), use_llm=True)
        self.assertEqual(out["overall"], "not_comparable")
        self.assertTrue(out["caveat"].startswith("These papers are not directly comparable"))

    def test_gemini_can_refine_a_pair_the_rules_cannot_read_in_a_three_paper_matrix(self):
        llm = ScriptedJsonLLM({"aspects": [{"aspect": "dataset", "verdict": "partial", "description": "Paper Alpha and Paper Gamma share a corpus; Paper Beta does not.",
                                            "cell_refs": [ref("A", "dataset"), ref("B", "dataset"), ref("C", "dataset")]}], "caveat": "Only Alpha and Gamma share a dataset, so compare Beta separately."})
        out = check(
            self.AMBIGUOUS + cells("C", dataset="A large corpus of scraped text.", evaluation_metrics="accuracy", main_results="x"),
            docs=DOCS3, llm=llm, use_llm=True)
        self.assertEqual(out["overall"], "partially_comparable")
        self.assertEqual(out["caveat"], "Only Alpha and Gamma share a dataset, so compare Beta separately.")


class TestOverall(unittest.TestCase):
    def outcome(self, **statuses):
        return {a: cmp.Outcome(a, s) for a, s in statuses.items()}

    def test_truth_table(self):
        o = self.outcome
        self.assertEqual(cmp.compute_overall(o(dataset="same", metric="same", task="unknown", split="unknown", setup="unknown")), "comparable")
        self.assertEqual(cmp.compute_overall(o(dataset="same", metric="unknown")), "unknown")
        self.assertEqual(cmp.compute_overall(o(dataset="unknown", metric="unknown")), "unknown")
        self.assertEqual(cmp.compute_overall(o(dataset="different", metric="unknown")), "not_comparable")
        self.assertEqual(cmp.compute_overall(o(dataset="same", metric="same", task="different")), "not_comparable")
        self.assertEqual(cmp.compute_overall(o(dataset="same", metric="partial")), "partially_comparable")
        self.assertEqual(cmp.compute_overall(o(dataset="same", metric="same", setup="different")), "partially_comparable")


class TestService(_Base):
    """The check wired into create_and_run_comparison and the endpoint."""

    # Verified quotes from the seeded papers; the VALUE text is what the check reads.
    SCRIPT = {
        "Paper One": {"fields": [
            field("dataset", quote="the SQuAD v1.1 dataset, which contains 100,000 questions", page=1, value="SQuAD v1.1."),
            field("evaluation_metrics", quote="fine-tuned BERT and it's a state-of-the-art 85.2 F1", page=2, value="F1."),
        ]},
        "Paper Two": {"fields": [
            field("dataset", quote="8 million web pages from outbound links", page=1, value="Its own collected dataset of 8 million web pages."),
            field("evaluation_metrics", quote="exact match accuracy as the main evaluation metric", page=2, value="Exact match accuracy."),
        ]},
    }

    def make(self, settings=None, llm=None):
        llm = llm or FakeLLM(self.SCRIPT)
        with patch.object(comparison_service, "get_llm_provider", return_value=llm), \
             patch.object(comparison_service, "build_retriever", return_value=MagicMock()):
            return comparison_service.create_and_run_comparison(
                self.db, "ws-1", "user-1", ["doc-1", "doc-2"], ["dataset", "evaluation_metrics"],
                settings=settings or Settings(comparability_use_llm=False),
            )

    def test_the_check_runs_after_the_matrix_is_filled_and_is_stored(self):
        matrix = self.make()
        self.assertEqual(matrix.status, ComparisonStatus.COMPLETED)
        self.assertIsNotNone(matrix.comparability)
        self.assertEqual(matrix.comparability["overall"], "not_comparable")
        types = {i["type"] for i in matrix.comparability["issues"]}
        self.assertEqual(types, {"dataset", "metric"})
        for issue in matrix.comparability["issues"]:  # every issue points at real cells of this matrix
            self.assertTrue(all(any(c.document_id == r["document_id"] and c.dimension_key == r["dimension_key"] for c in matrix.cells)
                                for r in issue["cell_refs"]))
        self.db.expire_all()
        self.assertEqual(self.db.get(ComparisonMatrix, matrix.id).comparability["overall"], "not_comparable")  # persisted

    def test_works_with_the_local_provider_rules_only(self):
        local = MagicMock(spec=LLMProvider)
        local.name, local.model = "local", "extractive-v1"
        local.generate_json.side_effect = NotImplementedError
        local.generate.return_value = "Per-cell answer [1]"
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        with patch.object(comparison_service, "get_llm_provider", return_value=local), \
             patch.object(comparison_service, "build_retriever", return_value=retriever):
            matrix = comparison_service.create_and_run_comparison(
                self.db, "ws-1", "user-1", ["doc-1", "doc-2"], ["dataset", "evaluation_metrics"], settings=Settings())
        self.assertEqual(matrix.comparability["checked_by"], "rules")
        self.assertIn(matrix.comparability["overall"], cmp.OVERALL_VALUES)
        local.generate_json.assert_not_called()  # use_llm is auto: Gemini only

    def test_a_failing_check_leaves_null_and_still_returns_the_matrix(self):
        with patch.object(comparison_service, "check_comparability", side_effect=RuntimeError("boom")), \
             self.assertLogs("app.comparison.comparison_service", "ERROR"):
            matrix = self.make()
        self.assertEqual(matrix.status, ComparisonStatus.COMPLETED)
        self.assertIsNone(matrix.comparability)
        self.assertEqual(len(matrix.cells), 4)  # the filled matrix is intact

    def test_switched_off_means_null(self):
        matrix = self.make(settings=Settings(comparability_enabled=False))
        self.assertIsNone(matrix.comparability)

    def test_the_cells_are_committed_before_the_check_runs(self):
        seen = {}

        def spy(cells_, dims, docs, llm=None, use_llm=False):
            seen["pending_cells"] = bool(self.db.new or self.db.dirty)  # nothing left uncommitted (no write lock held)
            return {"overall": "unknown", "issues": [], "caveat": "x.", "checked_by": "rules"}

        with patch.object(comparison_service, "check_comparability", side_effect=spy):
            self.make()
        self.assertFalse(seen["pending_cells"])

    def test_rerun_recomputes_and_enforces_ownership_and_status(self):
        matrix = self.make()
        matrix.comparability = None
        self.db.commit()
        again = comparison_service.rerun_comparability(self.db, matrix.id, "user-1", Settings(comparability_use_llm=False))
        self.assertEqual(again.comparability["overall"], "not_comparable")
        with self.assertRaises(comparison_service.ComparisonNotFoundError):
            comparison_service.rerun_comparability(self.db, matrix.id, "someone-else", Settings())
        matrix.status = ComparisonStatus.FAILED
        self.db.commit()
        with self.assertRaises(comparison_service.ComparisonError):
            comparison_service.rerun_comparability(self.db, matrix.id, "user-1", Settings())
        matrix.status = ComparisonStatus.COMPLETED
        self.db.commit()
        with self.assertRaises(comparison_service.ComparisonError):
            comparison_service.rerun_comparability(self.db, matrix.id, "user-1", Settings(comparability_enabled=False))

    def _client(self):
        from fastapi.testclient import TestClient

        from app import main
        from app.database import get_db

        def override():
            session = type(self.db)(bind=self.engine)
            try:
                yield session
            finally:
                session.close()

        main.app.dependency_overrides[get_db] = override
        self.addCleanup(main.app.dependency_overrides.clear)
        patcher = patch("app.deps.DEV_USER_EMAIL", "t@example.com")  # the seeded user-1 is the "current user"
        patcher.start()
        self.addCleanup(patcher.stop)
        return TestClient(main.app)

    def test_endpoint_rerun_get_and_ownership(self):
        matrix = self.make()
        client = self._client()
        with patch("app.routers.comparisons.settings", Settings(comparability_use_llm=False)):
            got = client.get(f"/api/comparisons/{matrix.id}").json()
            self.assertEqual(got["comparability"]["overall"], "not_comparable")
            self.assertEqual(set(got["comparability"]), {"overall", "issues", "caveat", "checked_by"})

            resp = client.post(f"/api/comparisons/{matrix.id}/comparability")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["comparability"]["overall"], "not_comparable")
            self.assertEqual(len(resp.json()["cells"]), 4)

            self.assertEqual(client.post("/api/comparisons/does-not-exist/comparability").status_code, 404)

            matrix.owner_id = "someone-else"
            self.db.commit()
            self.assertEqual(client.post(f"/api/comparisons/{matrix.id}/comparability").status_code, 404)
            self.assertEqual(client.get(f"/api/comparisons/{matrix.id}").status_code, 404)

    def test_endpoint_status_conflict_failure_and_malformed_stored_value(self):
        matrix = self.make()
        client = self._client()
        with patch("app.routers.comparisons.settings", Settings(comparability_use_llm=False)):
            matrix.comparability = {"overall": 7, "issues": "garbage"}  # not the contract
            self.db.commit()
            body = client.get(f"/api/comparisons/{matrix.id}")
            self.assertEqual(body.status_code, 200)
            self.assertIsNone(body.json()["comparability"])  # a bad stored value never breaks the matrix

            with patch.object(comparison_service, "check_comparability", side_effect=RuntimeError("boom")), \
                 self.assertLogs("app.routers.comparisons", "ERROR"):
                self.assertEqual(client.post(f"/api/comparisons/{matrix.id}/comparability").status_code, 500)

            matrix.status = ComparisonStatus.RUNNING
            self.db.commit()
            self.assertEqual(client.post(f"/api/comparisons/{matrix.id}/comparability").status_code, 409)


if __name__ == "__main__":
    unittest.main()
