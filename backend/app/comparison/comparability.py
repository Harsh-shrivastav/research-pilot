"""Comparability check (Phase P3): after a comparison matrix is filled, decide
whether the compared papers are DIRECTLY comparable, and say exactly why not.

    Paper A: 91% accuracy on PlantVillage
    Paper B: 94% accuracy on its own lab-collected dataset
    -> NOT directly comparable (different datasets)

Evidence-first, never guessing
------------------------------
The input is ONLY the already-extracted matrix cells (never the PDFs). For
each aspect a paper is judged from one cell; if that cell is missing, errored
or `insufficient_evidence`, the aspect is "unknown" - nothing is inferred.

    task     <- research_problem   (research_objective if that is not in the matrix)
    dataset  <- dataset
    metric   <- evaluation_metrics
    split    <- dataset / evaluation_metrics / main_results / methodology / model_algorithm text
    setup    <- the same cells (zero-shot vs fine-tuned, ...)

Step 1 - rules (plain code, no LLM, works with any provider)
    The cell text is normalised and mined for dataset names (alias table plus
    name-shaped tokens), metric names (alias table with variants such as
    top-k / macro-F1), task families, evaluation splits and setups. For every
    pair of papers the mentions are compared and each aspect comes out as
    same / partial / different, or AMBIGUOUS when the text gives the rules
    nothing to compare (e.g. a dataset cell that names no dataset).
Step 2 - the LLM, only for ambiguous aspects (Gemini, via generate_json)
    It sees only the cell texts, must answer "unknown" rather than infer, and
    must cite the cells it relies on. Its output is validated by code: an
    issue whose cell_refs are not real, usable cells of this matrix is
    dropped (the aspect stays unknown), and a "different" verdict needs cells
    from at least two papers. The LLM never sets the overall verdict.
Step 3 - the overall verdict is computed by code from the aspect outcomes.

Without an LLM (LLM_PROVIDER=local) steps 1 and 3 still run in full;
ambiguous aspects simply stay "unknown".
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from itertools import combinations

from app.rag.interfaces import LLMMessage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

ISSUE_TYPES = ("dataset", "metric", "task", "split", "setup")
OVERALL_VALUES = ("comparable", "partially_comparable", "not_comparable", "unknown")

# Which matrix dimension feeds which aspect (first one present in the matrix wins).
ASPECT_SOURCES = {
    "task": ["research_problem", "research_objective", "task", "problem"],
    "dataset": ["dataset", "datasets"],
    "metric": ["evaluation_metrics", "metrics", "evaluation_metric", "metric"],
}
RESULTS_KEYS = ["main_results", "results"]
CONTEXT_KEYS = ["methodology", "model_algorithm", "dataset_size", "baselines"]
# Cells that carry split / setup hints.
SIGNAL_KEYS = ["dataset", "datasets", "evaluation_metrics", "metrics", *RESULTS_KEYS, "methodology", "model_algorithm"]

MAX_CELL_CHARS = 700  # what the LLM sees of one cell
MAX_CAVEAT_CHARS = 320


def _squash(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[‐-―−]", "-", text)
    return " ".join(text.split())


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _flex(alias: str) -> str:
    """Alias -> regex tolerant of space / hyphen / underscore / nothing between parts."""
    parts = [re.escape(p) for p in re.split(r"[\s\-_]+", alias.strip().lower()) if p]
    return r"[\s\-_]?".join(parts)


def _compile_alternatives(aliases: list[str]) -> re.Pattern:
    body = "|".join(sorted((_flex(a) for a in aliases), key=len, reverse=True))
    return re.compile(rf"(?<![a-z0-9])(?:{body})(?![a-z0-9])")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

# (key, display, aliases). Order matters: more specific entries first (a match
# is masked out of the text so a shorter alias cannot match inside it again).
_KNOWN_DATASETS: list[tuple[str, str, list[str]]] = [
    ("imagenet21k", "ImageNet-21k", ["imagenet-21k", "imagenet21k", "imagenet-22k"]),
    ("imagenet", "ImageNet", ["imagenet", "imagenet-1k", "imagenet1k", "ilsvrc", "ilsvrc-2012", "ilsvrc2012"]),
    ("tinyimagenet", "Tiny ImageNet", ["tiny imagenet", "tiny-imagenet"]),
    ("cifar100", "CIFAR-100", ["cifar-100", "cifar100"]),
    ("cifar10", "CIFAR-10", ["cifar-10", "cifar10"]),
    ("fashionmnist", "Fashion-MNIST", ["fashion-mnist", "fashionmnist"]),
    ("emnist", "EMNIST", ["emnist"]),
    ("mnist", "MNIST", ["mnist"]),
    ("svhn", "SVHN", ["svhn"]),
    ("stl10", "STL-10", ["stl-10", "stl10"]),
    ("coco", "MS COCO", ["ms coco", "mscoco", "microsoft coco", "coco"]),
    ("pascalvoc", "Pascal VOC", ["pascal voc", "pascalvoc", "voc2007", "voc2012", "voc 2007", "voc 2012"]),
    ("cityscapes", "Cityscapes", ["cityscapes"]),
    ("ade20k", "ADE20K", ["ade20k"]),
    ("kitti", "KITTI", ["kitti"]),
    ("celeba", "CelebA", ["celeba"]),
    ("lfw", "LFW", ["lfw", "labeled faces in the wild"]),
    ("openimages", "Open Images", ["open images", "openimages"]),
    ("places365", "Places365", ["places365", "places-365"]),
    ("food101", "Food-101", ["food-101", "food101"]),
    ("kinetics", "Kinetics", ["kinetics", "kinetics-400", "kinetics-600"]),
    ("ucf101", "UCF101", ["ucf101", "ucf-101"]),
    ("plantvillage", "PlantVillage", ["plantvillage", "plant village"]),
    ("chestxray14", "ChestX-ray14", ["chestx-ray14", "chestxray14", "chest x-ray14"]),
    ("isic", "ISIC", ["isic"]),
    ("brats", "BraTS", ["brats"]),
    ("mimiciii", "MIMIC-III", ["mimic-iii", "mimic iii", "mimiciii"]),
    ("squad", "SQuAD", ["squad"]),
    ("naturalquestions", "Natural Questions", ["natural questions", "naturalquestions", "nq"]),
    ("triviaqa", "TriviaQA", ["triviaqa", "trivia qa", "trivia-qa"]),
    ("webquestions", "WebQuestions", ["webquestions", "web questions", "wq"]),
    ("curatedtrec", "CuratedTREC", ["curatedtrec", "curated trec"]),
    ("msmarco", "MS MARCO", ["ms marco", "msmarco", "ms-marco"]),
    ("hotpotqa", "HotpotQA", ["hotpotqa", "hotpot qa"]),
    ("quac", "QuAC", ["quac"]),
    ("coqa", "CoQA", ["coqa"]),
    ("race", "RACE", ["race dataset", "race benchmark"]),
    ("superglue", "SuperGLUE", ["superglue"]),
    ("glue", "GLUE", ["glue"]),
    ("mnli", "MNLI", ["mnli", "multinli"]),
    ("snli", "SNLI", ["snli"]),
    ("sst", "SST", ["sst-2", "sst-5", "sst2", "sst5", "sst", "stanford sentiment treebank"]),
    ("imdb", "IMDB", ["imdb"]),
    ("agnews", "AG News", ["ag news", "agnews", "ag-news"]),
    ("yelp", "Yelp", ["yelp"]),
    ("wikitext103", "WikiText-103", ["wikitext-103", "wikitext103"]),
    ("wikitext2", "WikiText-2", ["wikitext-2", "wikitext2"]),
    ("pennTreebank", "Penn Treebank", ["penn treebank", "ptb"]),
    ("lambada", "LAMBADA", ["lambada"]),
    ("conll2003", "CoNLL-2003", ["conll-2003", "conll2003", "conll 2003"]),
    ("ontonotes", "OntoNotes", ["ontonotes"]),
    ("wmt14ende", "WMT14 En-De", ["wmt14 en-de", "wmt 2014 en-de", "wmt14 english-german"]),
    ("wmt14enfr", "WMT14 En-Fr", ["wmt14 en-fr", "wmt 2014 en-fr", "wmt14 english-french"]),
    ("wmt", "WMT", ["wmt14", "wmt15", "wmt16", "wmt17", "wmt18", "wmt19", "wmt20", "wmt"]),
    ("iwslt", "IWSLT", ["iwslt14", "iwslt15", "iwslt17", "iwslt"]),
    ("cnndailymail", "CNN/DailyMail", ["cnn/daily mail", "cnn/dailymail", "cnn-dailymail", "cnn daily mail", "cnndm"]),
    ("xsum", "XSum", ["xsum"]),
    ("gigaword", "Gigaword", ["gigaword"]),
    ("webtext", "WebText", ["webtext"]),
    ("openwebtext", "OpenWebText", ["openwebtext"]),
    ("bookcorpus", "BookCorpus", ["bookcorpus", "book corpus"]),
    ("commoncrawl", "Common Crawl", ["common crawl", "commoncrawl"]),
    ("c4", "C4", ["c4"]),
    ("pile", "The Pile", ["the pile"]),
    ("wikipedia", "Wikipedia", ["wikipedia"]),
    ("mmlu", "MMLU", ["mmlu"]),
    ("gsm8k", "GSM8K", ["gsm8k"]),
    ("humaneval", "HumanEval", ["humaneval"]),
    ("hellaswag", "HellaSwag", ["hellaswag"]),
    ("winogrande", "WinoGrande", ["winogrande"]),
    ("boolq", "BoolQ", ["boolq"]),
    ("librispeech", "LibriSpeech", ["librispeech"]),
    ("commonvoice", "Common Voice", ["common voice", "commonvoice"]),
    ("timit", "TIMIT", ["timit"]),
    ("movielens", "MovieLens", ["movielens"]),
    ("atari", "Atari", ["atari", "arcade learning environment"]),
    ("mujoco", "MuJoCo", ["mujoco"]),
]
_KNOWN_DATASET_RES = [(k, d, _compile_alternatives(a)) for k, d, a in _KNOWN_DATASETS]
_VERSIONED = {"squad"}  # versions are part of the identity (SQuAD 1.1 != 2.0)
_VERSION_AFTER = re.compile(r"^\s*[-_ ]?\s*(?:v|version\s*)?(\d(?:\.\d)?)(?![\d.]*\d)")
_VARIANT_AFTER = re.compile(r"^-([a-z][a-z0-9]*)")  # "imagenet-c", "cifar10-c": a different dataset

_CAMEL = re.compile(r"\b[A-Za-z]*[a-z][A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)?\b")
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{2,9}(?:-\d+)?\b")
_ALNUM_NUM = re.compile(r"\b[A-Za-z]{3,}-?\d+(?:\.\d+)?\b")
_NAMED_PHRASE = re.compile(
    r"\b((?:[A-Z][\w'\-]*\s+){1,4}[A-Z][\w'\-]*)\s+(?:dataset|corpus|benchmark|data set|collection)s?\b"
)
_NOT_A_DATASET = {
    "the", "this", "paper", "papers", "english", "table", "figure", "section", "dataset", "datasets", "corpus",
    "bert", "gpt", "lstm", "gru", "cnn", "rnn", "gan", "vae", "gpu", "cpu", "tpu", "api", "nlp", "ai", "ml", "qa",
    "pdf", "url", "html", "sota", "bleu", "rouge", "meteor", "cider", "tfidf", "idf", "ocr", "asr", "llm", "llms",
    "tensorflow", "pytorch", "openai", "github", "huggingface", "python", "numpy", "sklearn", "keras",
    "january", "february", "march", "april", "june", "july", "august", "september", "october", "november", "december",
    "mapreduce", "seq2seq", "pretrained", "finetuned", "kaggle",
}
_NOT_A_DATASET_PREFIX = (
    "gpt", "bert", "roberta", "resnet", "vgg", "t5", "llama", "bm25", "top", "f1", "xlnet", "electra", "bart", "albert",
    "densenet", "efficientnet", "mobilenet", "yolo", "unet", "transformer", "lstm", "word2vec", "glove", "fasttext",
)

# Markers that a paper's data is its own. OWN: always means a private dataset (and can sit
# next to a public one). BUILT: only when no public dataset is named ("we collected ...").
_OWN_RE = re.compile(
    r"\b(?:(?:its|their|our|the authors'?|an?)\s+own\s+(?:[\w\-]+\s+){0,3}(?:dataset|data|corpus|collection|benchmark|images?|"
    r"samples?|recordings?|documents?|set)|in[\s\-]house|self[\s\-](?:collected|built|constructed|made|gathered|curated)|"
    r"lab[\s\-]collected|collected\s+in\s+(?:our|the|a)\s+lab|proprietary|private(?:ly)?\s+(?:dataset|data|collection|collected)|"
    r"custom[\s\-](?:built|made|collected)?\s*(?:dataset|data)|(?:newly|freshly)\s+(?:collected|constructed|created|built|"
    r"curated|annotated|gathered|introduced))\b",
    re.IGNORECASE,
)
_BUILT_RE = re.compile(
    r"\b(?:we|authors?)\s+(?:collected|constructed|created|curated|built|assembled|compiled|gathered|introduce[ds]?|"
    r"propose[ds]?\s+a\s+new)\b|\b(?:collected|constructed|curated|assembled)\s+(?:by\s+the\s+authors|ourselves)\b",
    re.IGNORECASE,
)


@dataclass
class Mentions:
    """What the rules found in one paper's cell: canonical key -> display name."""

    names: dict[str, str] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)  # key -> version, for versioned datasets
    custom: bool = False  # the paper's own / newly built data

    def empty(self) -> bool:
        return not self.names and not self.custom


def extract_datasets(text: str) -> Mentions:
    text = _squash(text)
    lower = text.lower()
    mentions = Mentions()
    work = lower

    for key, display, pattern in _KNOWN_DATASET_RES:
        for match in pattern.finditer(work):
            tail = lower[match.end():match.end() + 12]
            variant = _VARIANT_AFTER.match(tail)
            if variant and key not in _VERSIONED:
                vkey = f"{key}-{variant.group(1)}"
                mentions.names.setdefault(vkey, f"{display}-{variant.group(1).upper() if len(variant.group(1)) <= 2 else variant.group(1)}")
                work = work[: match.start()] + " " * (match.end() - match.start() + variant.end()) + work[match.end() + variant.end():]
                continue
            mentions.names.setdefault(key, display)
            if key in _VERSIONED:
                version = _VERSION_AFTER.match(tail)
                if version:
                    mentions.versions[key] = version.group(1)
            work = work[: match.start()] + " " * (match.end() - match.start()) + work[match.end():]

    # Name-shaped tokens the table does not know (original case; known spans blanked).
    masked = "".join(" " if w == " " and o != " " else o for o, w in zip(text, work))
    if len(masked) != len(text):  # pragma: no cover - defensive
        masked = text
    candidates: list[str] = []
    candidates += _NAMED_PHRASE.findall(masked)
    candidates += _CAMEL.findall(masked)
    candidates += _ACRONYM.findall(masked)
    candidates += _ALNUM_NUM.findall(masked)
    for token in candidates:
        key = _key(token)
        if len(key) < 3 or key in _NOT_A_DATASET or key.startswith(_NOT_A_DATASET_PREFIX) or key.isdigit():
            continue
        if any(key == k or key.startswith(k + "-") for k in mentions.names) or key in {_key(d) for d in mentions.names.values()}:
            continue
        mentions.names.setdefault(key, token.strip())

    if _OWN_RE.search(text) or (not mentions.names and _BUILT_RE.search(text)):
        mentions.custom = True
    return mentions


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_METRIC_FAMILY_SEP = "@"
# (key, display, regex) on lower-case text, most specific first; matches are masked.
_METRICS: list[tuple[str, str, str]] = [
    ("recall@k", "recall@k", r"(?:recall|r)\s*@\s*(\d+|k)|recall\s+at\s+(\d+|k)"),
    ("precision@k", "precision@k", r"(?:precision|p)\s*@\s*(\d+|k)|precision\s+at\s+(\d+|k)"),
    ("ndcg", "NDCG", r"n?dcg(?:\s*@\s*\d+)?|normalized discounted cumulative gain"),
    ("mrr", "MRR", r"\bmrr\b|mean reciprocal rank"),
    ("hit@k", "hit rate", r"hits?\s*@\s*\d+|hit[\s\-]rate"),
    ("map", "mAP", r"mean average precision|\bmap\b"),
    ("aupr", "AUPR", r"aupr|pr[\s\-]?auc|area under the precision[\s\-]recall curve|average precision"),
    ("auroc", "AUROC", r"auroc|roc[\s\-]?auc|area under (?:the )?(?:roc )?curve|\bauc\b"),
    ("exact_match", "exact match", r"exact[\s\-]?match(?:\s*\(em\))?(?:\s+(?:accuracy|score|metric|rate))?|\(em\)(?:\s+(?:accuracy|score|metric|rate))?|\bem\s+(?:score|metric|accuracy)"),
    ("f1_macro", "macro-F1", r"macro[\s\-]*f[\s\-]?1(?:[\s\-]?(?:score|measure))?|macro[\s\-]*f[\s\-]?(?:score|measure)"),
    ("f1_micro", "micro-F1", r"micro[\s\-]*f[\s\-]?1(?:[\s\-]?(?:score|measure))?|micro[\s\-]*f[\s\-]?(?:score|measure)"),
    ("f1_weighted", "weighted F1", r"weighted[\s\-]*f[\s\-]?1(?:[\s\-]?(?:score|measure))?"),
    ("f1", "F1", r"\bf[\s\-]?1(?:[\s\-]?(?:score|measure))?\b|\bf[\s\-]?(?:score|measure)\b"),
    ("sacrebleu", "SacreBLEU", r"sacre[\s\-]?bleu"),
    ("bleu", "BLEU", r"\bbleu(?:[\s\-]?[1-4])?\b"),
    ("rouge1", "ROUGE-1", r"rouge[\s\-]?1\b"),
    ("rouge2", "ROUGE-2", r"rouge[\s\-]?2\b"),
    ("rougel", "ROUGE-L", r"rouge[\s\-]?l(?:sum)?\b"),
    ("rouge", "ROUGE", r"\brouge\b"),
    ("meteor", "METEOR", r"\bmeteor\b"),
    ("cider", "CIDEr", r"\bcider\b"),
    ("bertscore", "BERTScore", r"bert[\s\-]?score"),
    ("perplexity", "perplexity", r"perplexity|\bppl\b"),
    ("bpc", "bits per character", r"bits[\s\-]per[\s\-](?:character|byte|char)|\bbpc\b|\bbpb\b"),
    ("wer", "WER", r"word error rate|\bwer\b"),
    ("cer", "CER", r"character error rate|\bcer\b"),
    ("rmse", "RMSE", r"\brmse\b|root mean squared? error"),
    ("mae", "MAE", r"\bmae\b|mean absolute error"),
    ("mse", "MSE", r"\bmse\b|mean squared error"),
    ("r2", "R²", r"\br2\b|r[\s\-]squared|coefficient of determination"),
    ("miou", "mIoU", r"\bm?iou\b|intersection over union|jaccard"),
    ("dice", "Dice", r"\bdice\b"),
    ("psnr", "PSNR", r"\bpsnr\b"),
    ("ssim", "SSIM", r"\bssim\b"),
    ("fid", "FID", r"\bfid\b|fr[ée]chet inception distance"),
    ("kappa", "Cohen's kappa", r"kappa"),
    ("mcc", "Matthews correlation", r"matthews"),
    ("spearman", "Spearman correlation", r"spearman"),
    ("pearson", "Pearson correlation", r"pearson"),
    ("recall", "recall", r"\brecall\b|\bsensitivity\b|\btrue positive rate\b"),
    ("precision", "precision", r"\bprecision\b"),
    ("specificity", "specificity", r"\bspecificity\b"),
    ("latency", "latency", r"\blatency\b|inference time"),
    ("throughput", "throughput", r"\bthroughput\b"),
]
_TOPK_RE = re.compile(r"top[\s\-]?(\d+|k)(?:\s+(?:retrieval|classification|test))?\s+accuracy|top[\s\-]?(\d+|k)\s*(?:acc)?\b")
_ACCURACY_RE = re.compile(r"\baccuracy\b|\bacc\b|\bclassification rate\b")
_METRIC_RES = [(k, d, re.compile(p)) for k, d, p in _METRICS]


def _metric_family(key: str) -> str:
    if key == "sacrebleu":
        return "bleu"
    if key.startswith("top") and key.endswith("_accuracy"):
        return "topk_accuracy"
    for prefix in ("f1", "rouge", "bleu"):
        if key.startswith(prefix):
            return prefix
    if _METRIC_FAMILY_SEP in key:
        return key.split(_METRIC_FAMILY_SEP)[0] + "@k"
    return key


def extract_metrics(text: str) -> dict[str, str]:
    """{canonical key: display} of the metrics named in `text`."""
    work = _squash(text).lower()
    found: dict[str, str] = {}

    def take(key: str, display: str, match: re.Match) -> None:
        nonlocal work
        found.setdefault(key, display)
        work = work[: match.start()] + " " * (match.end() - match.start()) + work[match.end():]

    for key, display, pattern in _METRIC_RES:
        for match in list(pattern.finditer(work)):
            if key in ("recall@k", "precision@k"):
                k = next((g for g in match.groups() if g), "k")
                take(f"{key.split('@')[0]}@{k}", f"{key.split('@')[0]}@{k}", match)
            else:
                take(key, display, match)
    for match in list(_TOPK_RE.finditer(work)):
        k = next((g for g in match.groups() if g), "k")
        if k == "1":
            take("accuracy", "accuracy", match)
        else:
            take(f"top{k}_accuracy", f"top-{k} accuracy", match)
    for match in list(_ACCURACY_RE.finditer(work)):
        take("accuracy", "accuracy", match)
    return found


# ---------------------------------------------------------------------------
# Tasks, splits, setups
# ---------------------------------------------------------------------------

_TASKS: dict[str, tuple[str, re.Pattern]] = {
    "question answering": ("question answering", re.compile(r"question[\s\-]answering|\bqa\b|reading comprehension|open[\s\-]domain (?:qa|question)")),
    "language modeling": ("language modeling", re.compile(r"language model(?:l)?ing|next[\s\-](?:word|token) prediction|zero[\s\-]shot task transfer")),
    "machine translation": ("machine translation", re.compile(r"machine translation|\bnmt\b|\btranslation\b")),
    "summarization": ("summarization", re.compile(r"summari[sz]ation|summari[sz]e")),
    "text classification": ("text classification", re.compile(r"sentiment|text classification|document classification|topic classification|spam detection|natural language inference|textual entailment")),
    "image classification": ("image classification", re.compile(r"image classification|object recognition|visual recognition|classif\w+ (?:of )?(?:plant|leaf|leaves|images?|photos?)|(?:plant|leaf|crop|skin|medical)[\w\s]{0,20}(?:disease|lesion)s? (?:classification|recognition|identification|detection)")),
    "object detection": ("object detection", re.compile(r"object detection|\bdetect(?:ing|ion)? (?:of )?objects")),
    "segmentation": ("segmentation", re.compile(r"segmentation")),
    "retrieval": ("retrieval", re.compile(r"(?:information|passage|document|dense|text|image|cross[\s\-]modal) retrieval|\bretriev(?:er|ing|al)\b|document ranking|re-?ranking|\branking\b")),
    "text generation": ("text generation", re.compile(r"text generation|language generation|natural language generation|abstractive|\bgeneration\b|\bgenerate\b")),
    "speech recognition": ("speech recognition", re.compile(r"speech recognition|\basr\b|speech[\s\-]to[\s\-]text")),
    "sequence tagging": ("sequence tagging", re.compile(r"named entity|\bner\b|part[\s\-]of[\s\-]speech|sequence (?:tagging|labeling)|\bpos tagging\b")),
    "recommendation": ("recommendation", re.compile(r"recommend(?:ation|er)")),
    "forecasting": ("forecasting", re.compile(r"forecast|time[\s\-]series|\bregression\b|price prediction")),
    "clustering": ("clustering", re.compile(r"clustering")),
    "reinforcement learning": ("reinforcement learning", re.compile(r"reinforcement learning|\bpolicy (?:learning|optimi[sz]ation)\b")),
    "dialogue": ("dialogue", re.compile(r"dialogue|dialog|conversational")),
    "anomaly detection": ("anomaly detection", re.compile(r"anomaly detection|outlier detection|fraud detection")),
}


def extract_tasks(text: str) -> dict[str, str]:
    lower = _squash(text).lower()
    return {key: display for key, (display, pattern) in _TASKS.items() if pattern.search(lower)}


_SPLITS: list[tuple[str, str, re.Pattern]] = [
    ("kfold", "k-fold cross-validation", re.compile(r"\b(\d+)[\s\-]?fold")),
    ("cv", "cross-validation", re.compile(r"cross[\s\-]?validation|leave[\s\-]one[\s\-]out|\bk[\s\-]fold\b")),
    ("ratio", "train/test split", re.compile(r"\b(\d{2})\s*[/:]\s*(\d{2})(?:\s*[/:]\s*(\d{2}))?\s*(?:train|split|ratio)|(\d{2})\s*%\s*(?:for\s+)?(?:training|train)\b[^.]{0,40}(\d{2})\s*%\s*(?:for\s+)?(?:testing|test)")),
    ("test", "test set", re.compile(r"\btest(?:ing)?[\s\-](?:set|split|data|partition|subset)|held[\s\-]?out test|\bon the test\b|\btest accuracy\b")),
    ("validation", "validation set", re.compile(r"validation[\s\-](?:set|split|data|subset)|\bdev(?:elopment)?[\s\-](?:set|split)|\bvalid(?:ation)? (?:accuracy|score|results?)\b")),
]
_SETUPS: list[tuple[str, str, re.Pattern]] = [
    ("zeroshot", "zero-shot", re.compile(r"zero[\s\-]?shot")),
    ("fewshot", "few-shot", re.compile(r"few[\s\-]?shot|\b\d+[\s\-]?shot\b|in[\s\-]context learning")),
    ("finetuned", "fine-tuned", re.compile(r"fine[\s\-]?tun(?:e|ed|ing)")),
    ("scratch", "trained from scratch", re.compile(r"from[\s\-]scratch|randomly initiali[sz]ed")),
    ("linearprobe", "linear probe", re.compile(r"linear[\s\-]probe|linear evaluation")),
]


def _match_signals(patterns: list[tuple[str, str, re.Pattern]], text: str) -> dict[str, str]:
    lower = _squash(text).lower()
    found: dict[str, str] = {}
    for key, display, pattern in patterns:
        match = pattern.search(lower)
        if not match:
            continue
        if key == "kfold":
            found[f"kfold{match.group(1)}"] = f"{match.group(1)}-fold cross-validation"
        elif key == "cv":
            if not any(k.startswith("kfold") for k in found):
                found["cv"] = display
        else:
            found[key] = display
    return found


def extract_splits(text: str) -> dict[str, str]:
    found = _match_signals(_SPLITS, text)
    if "cv" in found and any(k.startswith("kfold") for k in found):
        found.pop("cv")
    return found


_NEGATED = re.compile(
    r"\b(?:without|no|not|never|nor)\s+(?:any\s+|the\s+|a\s+)?(?:further\s+|additional\s+|extra\s+|task[\s\-]specific\s+)?"
    r"(?:fine[\s\-]?tun\w*|training|train(?:ed)?|parameter updates?|gradient updates?|examples?|demonstrations?)\b"
)


def extract_setups(text: str) -> dict[str, str]:
    """Setups stated by `text`. A negated mention ("zero-shot, without any
    fine-tuning") is removed first: it says the opposite of the setup it names."""
    return _match_signals(_SETUPS, _NEGATED.sub(" ", _squash(text).lower()))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CellInput:
    """One matrix cell, reduced to what the check needs."""

    document_id: str
    dimension_key: str
    text: str
    usable: bool  # has a real answer: no error, not insufficient_evidence, not empty


@dataclass
class Issue:
    type: str
    description: str
    document_ids: list[str]
    cell_refs: list[dict]
    origin: str = "rules"  # rules | gemini

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "description": self.description,
            "document_ids": list(self.document_ids),
            "cell_refs": [dict(ref) for ref in self.cell_refs],
        }


@dataclass
class Outcome:
    """One aspect's result. status: same | partial | different | unknown | ambiguous."""

    aspect: str
    status: str
    issues: list[Issue] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)  # why unknown / unresolved (for the caveat)
    summary: str = ""  # what the papers share, when status is "same"
    origin: str = "rules"
    ambiguous_ids: list[str] = field(default_factory=list)  # papers the rules could not read
    refs: list[dict] = field(default_factory=list)  # all source cells considered


@dataclass
class _Paper:
    id: str
    label: str


class _Matrix:
    """Cells indexed for the check."""

    def __init__(self, cells: list[CellInput], dimension_keys: list[str], papers: list[_Paper]):
        self.papers = papers
        self.dimension_keys = list(dimension_keys)
        self.by_ref: dict[tuple[str, str], CellInput] = {(c.document_id, c.dimension_key): c for c in cells}

    def source_key(self, aspect: str) -> str | None:
        return next((k for k in ASPECT_SOURCES[aspect] if k in self.dimension_keys), None)

    def cell(self, document_id: str, key: str | None) -> CellInput | None:
        return self.by_ref.get((document_id, key)) if key else None

    def usable(self, document_id: str, key: str | None) -> bool:
        cell = self.cell(document_id, key)
        return bool(cell and cell.usable and (cell.text or "").strip())


def _ref(document_id: str, key: str) -> dict:
    return {"document_id": document_id, "dimension_key": key}


# ---------------------------------------------------------------------------
# Step 1: rules
# ---------------------------------------------------------------------------


def _relation_sets(a: dict[str, str], b: dict[str, str], *, families=None, shared_is_same: bool = False) -> str:
    """same / partial / different for two non-empty {key: display} sets."""
    ka, kb = set(a), set(b)
    if ka == kb:
        return "same"
    if ka & kb:
        return "same" if shared_is_same else "partial"
    if families is not None:
        fa, fb = {families(k) for k in ka}, {families(k) for k in kb}
        if fa & fb:
            return "partial"
    return "different"


def _dataset_relation(a: Mentions, b: Mentions, id_a: str, id_b: str) -> str:
    """Mentions -> relation. A paper's own data is a private name that no one
    else can share, so own-vs-anything is 'different'."""
    if a.empty() or b.empty():
        return "ambiguous"
    na, nb = dict(a.names), dict(b.names)
    for key in set(a.versions) & set(b.versions):
        if a.versions[key] != b.versions[key]:  # same family, different version
            na[f"{key}@{a.versions[key]}"], nb[f"{key}@{b.versions[key]}"] = na.pop(key), nb.pop(key)
    if a.custom:
        na[f"own:{id_a}"] = "its own dataset"
    if b.custom:
        nb[f"own:{id_b}"] = "its own dataset"
    return _relation_sets(na, nb)


def _contrast(phrases: list[str]) -> str:
    """"A uses X while B uses Y" for two papers; a semicolon list for more."""
    return f"{phrases[0]} while {phrases[1]}" if len(phrases) == 2 else "; ".join(phrases)


def _describe(phrases: list[str], plural: str, different: bool) -> str:
    tail = "differ" if different else "only partly overlap"
    return _sentence(f"{_contrast(phrases)}, so the {plural} {tail}")


def _aggregate(relations: list[str], missing: int) -> tuple[str, bool]:
    """Pairwise relations -> (status, has_ambiguous). Missing evidence never
    produces 'same'; a definite mismatch is still reported."""
    has_amb = "ambiguous" in relations
    definite = [r for r in relations if r != "ambiguous"]
    if not definite:
        return "ambiguous", True
    if "different" in definite:
        if all(r == "different" for r in definite) and not has_amb:
            return "different", has_amb
        return "partial", has_amb
    if "partial" in definite:
        return "partial", has_amb
    # all definite relations are "same"
    if has_amb or missing:
        return "unknown", has_amb
    return "same", has_amb


def _evaluate_extracted(
    aspect: str,
    matrix: _Matrix,
    extract,  # text -> mentions object
    relate,  # (mentions_a, id_a, mentions_b, id_b) -> relation
    show,  # mentions -> list[str] to display
    verb: str,  # "uses" / "reports" / "addresses"
    plural: str,  # "datasets" / "metrics" / "tasks"
    noun: str,
) -> Outcome:
    key = matrix.source_key(aspect)
    if key is None:
        return Outcome(aspect, "unknown", reasons=[f"the comparison does not include a {noun} criterion"])

    evidenced = [p for p in matrix.papers if matrix.usable(p.id, key)]
    missing = [p for p in matrix.papers if not matrix.usable(p.id, key)]
    refs = [_ref(p.id, key) for p in evidenced]
    reasons = [f"no usable {noun} information for {p.label}" for p in missing]
    if len(evidenced) < 2:
        return Outcome(aspect, "unknown", reasons=reasons or [f"no {noun} information"], refs=refs)

    mentions = {p.id: extract(matrix.cell(p.id, key).text) for p in evidenced}
    relations: dict[tuple[str, str], str] = {}
    for p, q in combinations(evidenced, 2):
        relations[(p.id, q.id)] = relate(mentions[p.id], p.id, mentions[q.id], q.id)

    status, has_amb = _aggregate(list(relations.values()), len(missing))
    outcome = Outcome(aspect, status, reasons=reasons, refs=refs)
    unreadable = {pid for (a, b), r in relations.items() if r == "ambiguous" for pid in (a, b)}
    outcome.ambiguous_ids = [p.id for p in evidenced if p.id in unreadable]
    if has_amb:
        outcome.reasons.append(f"the {noun} text of some papers names nothing the rules can compare")

    if status in ("different", "partial"):
        involved = [p for p in evidenced if p.id not in unreadable or status == "different"]
        phrases = [f"{p.label} {verb} {' and '.join(show(mentions[p.id])) or 'nothing it names'}" for p in involved]
        outcome.issues.append(
            Issue(
                type=aspect,
                description=_describe(phrases, plural, status == "different"),
                document_ids=[p.id for p in involved],
                cell_refs=[_ref(p.id, key) for p in involved],
            )
        )
    elif status == "same":
        names: list[str] = []
        for p in evidenced:
            for item in show(mentions[p.id]):
                if item not in names:
                    names.append(item)
        outcome.summary = ", ".join(names)
    return outcome


def _show_datasets(m: Mentions) -> list[str]:
    shown = [m.names[k] + (f" {m.versions[k]}" if k in m.versions else "") for k in m.names]
    if m.custom:
        shown.append("its own dataset")
    return shown


def evaluate_dataset(matrix: _Matrix) -> Outcome:
    return _evaluate_extracted(
        "dataset", matrix, extract_datasets,
        lambda a, ia, b, ib: _dataset_relation(a, b, ia, ib),
        _show_datasets, "uses", "datasets", "dataset",
    )


def _metric_relation(a: dict, ia: str, b: dict, ib: str) -> str:
    if not a or not b:
        return "ambiguous"
    return _relation_sets(a, b, families=_metric_family)


def evaluate_metric(matrix: _Matrix) -> Outcome:
    return _evaluate_extracted(
        "metric", matrix, extract_metrics, _metric_relation, lambda m: list(m.values()), "reports", "metrics", "evaluation metric",
    )


def _task_relation(a: dict, ia: str, b: dict, ib: str) -> str:
    if not a or not b:
        return "ambiguous"
    return _relation_sets(a, b, shared_is_same=True)  # sharing a task is enough


def evaluate_task(matrix: _Matrix) -> Outcome:
    return _evaluate_extracted(
        "task", matrix, extract_tasks, _task_relation, lambda m: list(m.values()), "addresses", "tasks", "task",
    )


def _evaluate_signal(aspect: str, matrix: _Matrix, extract, verb: str, plural: str) -> Outcome:
    """Split / setup: optional context. Only a definite difference between two
    papers that both state it is reported; silence is never a difference."""
    signals: dict[str, dict[str, str]] = {}
    sources: dict[str, dict[str, set]] = {}
    for paper in matrix.papers:
        found: dict[str, str] = {}
        for key in SIGNAL_KEYS:
            if not matrix.usable(paper.id, key):
                continue
            for k, display in extract(matrix.cell(paper.id, key).text).items():
                found.setdefault(k, display)
                sources.setdefault(paper.id, {}).setdefault(k, set()).add(key)
        if found:
            signals[paper.id] = found
    stating = [p for p in matrix.papers if p.id in signals]
    if len(stating) < 2:
        return Outcome(aspect, "unknown")  # not stated by enough papers: nothing to compare, nothing to claim
    relations = [_relation_sets(signals[p.id], signals[q.id], shared_is_same=True) for p, q in combinations(stating, 2)]
    if all(r == "same" for r in relations):
        return Outcome(aspect, "same", summary=", ".join(sorted({d for s in signals.values() for d in s.values()})))
    involved = [p for p in stating if any(
        _relation_sets(signals[p.id], signals[q.id], shared_is_same=True) == "different" for q in stating if q.id != p.id)]
    refs: list[dict] = []
    for p in involved:
        for keys in sources[p.id].values():
            for key in sorted(keys):
                ref = _ref(p.id, key)
                if ref not in refs:
                    refs.append(ref)
    phrases = [f"{p.label} {verb} {' and '.join(signals[p.id].values())}" for p in involved]
    issue = Issue(
        type=aspect, description=_describe(phrases, plural, True),
        document_ids=[p.id for p in involved], cell_refs=refs,
    )
    return Outcome(aspect, "different" if len(involved) == len(stating) else "partial", issues=[issue])


# ---------------------------------------------------------------------------
# Step 2: the LLM, only for ambiguous aspects
# ---------------------------------------------------------------------------

LLM_SYSTEM_PROMPT = """\
You judge whether research papers' reported results can be compared directly, using ONLY the \
short extracted cell texts provided. Never use outside knowledge about the actual papers.

Rules you must follow:
1. Judge only from the cells provided. Do not infer anything a cell does not state.
2. If a dataset or metric cell says the paper does not provide the information (not found), or a \
needed cell is missing, the verdict for that aspect is "unknown". Do not infer.
3. For each aspect asked, return a verdict: "same" (the cells clearly describe the same dataset / \
metric / task), "different" (clearly not the same), "partial" (some overlap but not identical), or \
"unknown".
4. For "different" and "partial", write ONE plain sentence in "description" naming the papers and the \
difference, and list in "cell_refs" every cell the verdict is based on, copying document_id and \
dimension_key EXACTLY from the cells shown. Every issue must reference its cell_refs. A "different" \
verdict needs cells from at least two different papers.
5. "caveat": ONE plain sentence a student could paste into a literature review saying whether, and \
why or why not, these papers' results can be compared. No lists and no line breaks.
"""


def _llm_schema(aspects: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "aspects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "aspect": {"type": "string", "enum": aspects},
                        "verdict": {"type": "string", "enum": ["same", "different", "partial", "unknown"]},
                        "description": {"type": "string"},
                        "cell_refs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"document_id": {"type": "string"}, "dimension_key": {"type": "string"}},
                                "required": ["document_id", "dimension_key"],
                            },
                        },
                    },
                    "required": ["aspect", "verdict", "description", "cell_refs"],
                },
            },
            "caveat": {"type": "string"},
        },
        "required": ["aspects", "caveat"],
    }


def _allowed_ref_keys(matrix: _Matrix, aspect: str) -> list[str]:
    source = matrix.source_key(aspect)
    keys = [source] if source else []
    keys += [k for k in RESULTS_KEYS + CONTEXT_KEYS if k in matrix.dimension_keys]
    return keys


def _build_llm_message(matrix: _Matrix, ambiguous: list[Outcome], settled: list[Outcome]) -> str:
    lines = ["Papers:"]
    for paper in matrix.papers:
        lines.append(f'- document_id: {paper.id} | title: "{paper.label}"')
    lines.append("")
    for outcome in ambiguous:
        lines.append(f"ASPECT TO JUDGE: {outcome.aspect}")
        for paper in matrix.papers:
            for key in _allowed_ref_keys(matrix, outcome.aspect):
                if matrix.usable(paper.id, key):
                    text = _squash(matrix.cell(paper.id, key).text)[:MAX_CELL_CHARS]
                    lines.append(f'  cell(document_id={paper.id}, dimension_key={key}): "{text}"')
                elif key == matrix.source_key(outcome.aspect):
                    lines.append(f"  cell(document_id={paper.id}, dimension_key={key}): [not found / missing]")
        lines.append("")
    if settled:
        lines.append("Already established from these same cells by code (include in the caveat where relevant):")
        for outcome in settled:
            if outcome.issues:
                lines.extend(f"- {issue.description}" for issue in outcome.issues)
            elif outcome.status == "same" and outcome.summary:
                lines.append(f"- {outcome.aspect}: the same ({outcome.summary})")
        lines.append("")
    lines.append("Return the JSON object: one entry per ASPECT TO JUDGE, plus the one-sentence caveat.")
    return "\n".join(lines)


def _valid_refs(matrix: _Matrix, aspect: str, refs) -> list[dict]:
    allowed = set(_allowed_ref_keys(matrix, aspect))
    valid_ids = {p.id for p in matrix.papers}
    out: list[dict] = []
    for ref in refs if isinstance(refs, list) else []:
        if not isinstance(ref, dict):
            continue
        doc, key = ref.get("document_id"), ref.get("dimension_key")
        clean = _ref(doc, key)
        if doc in valid_ids and key in allowed and matrix.usable(doc, key) and clean not in out:
            out.append(clean)
    return out


def _is_one_sentence(text) -> bool:
    if not isinstance(text, str):
        return False
    text = text.strip()
    return bool(text) and "\n" not in text and len(text) <= MAX_CAVEAT_CHARS and len(re.findall(r"[.!?](?:\s|$)", text)) <= 1


def resolve_with_llm(
    llm, matrix: _Matrix, ambiguous: list[Outcome], settled: list[Outcome]
) -> tuple[dict[str, Outcome], str | None]:
    """Ask the LLM about the ambiguous aspects. Returns ({aspect: validated
    Outcome}, caveat or None). Every failure mode degrades to "no answer"."""
    aspects = [o.aspect for o in ambiguous]
    try:
        data = llm.generate_json(
            LLM_SYSTEM_PROMPT,
            [LLMMessage(role="user", content=_build_llm_message(matrix, ambiguous, settled))],
            _llm_schema(aspects),
        )
    except Exception as exc:
        logger.warning("comparability: LLM step failed (%s: %s); keeping the rules' result", type(exc).__name__, str(exc)[:200])
        return {}, None
    if not isinstance(data, dict) or not isinstance(data.get("aspects"), list):
        logger.warning("comparability: LLM returned no 'aspects' list; ignoring it")
        return {}, None

    labels = {p.id: p.label for p in matrix.papers}
    resolved: dict[str, Outcome] = {}
    by_aspect = {o.aspect: o for o in ambiguous}
    for item in data["aspects"]:
        if not isinstance(item, dict) or item.get("aspect") not in by_aspect or item["aspect"] in resolved:
            continue
        aspect, verdict = item["aspect"], item.get("verdict")
        if verdict not in ("same", "different", "partial", "unknown"):
            continue
        if verdict == "unknown":
            resolved[aspect] = Outcome(aspect, "unknown", reasons=["the cells do not give enough information to judge"], origin="gemini")
            continue
        refs = _valid_refs(matrix, aspect, item.get("cell_refs"))
        docs = {r["document_id"] for r in refs}
        description = _squash(item.get("description") or "")
        if verdict == "same":
            # "same" still has to rest on cells from at least two papers.
            if len(docs) < 2:
                continue
            resolved[aspect] = Outcome(aspect, "same", summary=description[:120], origin="gemini", refs=refs)
            continue
        if not refs or not description or (verdict == "different" and len(docs) < 2):
            logger.info("comparability: dropped an unsupported %s verdict for %s (refs=%d)", verdict, aspect, len(refs))
            continue
        issue = Issue(
            type=aspect, description=_sentence(description), document_ids=[d for d in labels if d in docs],
            cell_refs=refs, origin="gemini",
        )
        resolved[aspect] = Outcome(aspect, verdict, issues=[issue], origin="gemini", refs=refs)
    caveat = data.get("caveat")
    return resolved, (_squash(caveat) if _is_one_sentence(caveat) else None)


# ---------------------------------------------------------------------------
# Step 3: overall verdict + caveat (code, deterministic)
# ---------------------------------------------------------------------------


def _sentence(text: str) -> str:
    text = _squash(text).rstrip(" .;:")
    if len(text) > 400:
        text = text[:397].rstrip() + "..."
    return text[:1].upper() + text[1:] + "."


def compute_overall(outcomes: dict[str, Outcome]) -> str:
    core = [outcomes[a] for a in ("dataset", "metric", "task") if a in outcomes]
    if any(o.status == "different" for o in core):
        return "not_comparable"
    if any(o.status == "partial" for o in core):
        return "partially_comparable"
    if any(o.status in ("different", "partial") for o in (outcomes.get("split"), outcomes.get("setup")) if o):
        return "partially_comparable"
    dataset, metric = outcomes.get("dataset"), outcomes.get("metric")
    if dataset and metric and dataset.status == "same" and metric.status == "same":
        return "comparable"
    return "unknown"


def _template_caveat(overall: str, outcomes: dict[str, Outcome], issues: list[Issue]) -> str:
    if overall == "comparable":
        dataset, metric = outcomes["dataset"], outcomes["metric"]
        parts = []
        if dataset.summary:
            parts.append(f"the same dataset ({dataset.summary})")
        else:
            parts.append("the same dataset")
        parts.append(f"the same metric ({metric.summary})" if metric.summary else "the same metric")
        return _sentence(f"These papers use {' and '.join(parts)}, so their reported results can be compared directly")
    if overall in ("not_comparable", "partially_comparable"):
        lead = "These papers are not directly comparable" if overall == "not_comparable" else "These papers are only partly comparable"
        text = f"{lead}: {'; '.join(i.description.rstrip('.') for i in issues[:2])}"
        if len(text) > MAX_CAVEAT_CHARS:
            text = f"{lead}: {issues[0].description.rstrip('.')}"
        return _sentence(text)
    reasons: list[str] = []
    for outcome in outcomes.values():
        for reason in outcome.reasons:
            if outcome.aspect in ("dataset", "metric", "task") and reason not in reasons:
                reasons.append(reason)
    why = f": {reasons[0]}" if reasons else ""
    return _sentence(f"There is not enough extracted evidence to tell whether these papers' results are comparable{why}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _paper_labels(documents: list[tuple[str, str]]) -> list[_Paper]:
    papers: list[_Paper] = []
    seen: dict[str, int] = {}
    for doc_id, title in documents:
        label = _squash(title or "Untitled paper")
        if len(label) > 48:
            label = label[:45].rstrip() + "..."
        seen[label] = seen.get(label, 0) + 1
        papers.append(_Paper(doc_id, label if seen[label] == 1 else f"{label} ({seen[label]})"))
    return papers


def check_comparability(
    cells: list[CellInput],
    dimension_keys: list[str],
    documents: list[tuple[str, str]],
    llm=None,
    use_llm: bool = False,
) -> dict:
    """Judge whether the papers behind `cells` are directly comparable.

    `documents` is [(document_id, title)] in matrix order. Returns the dict
    stored on ComparisonMatrix.comparability:
    {overall, issues, caveat, checked_by}. Pure and deterministic with
    `use_llm=False`; with it, only ambiguous aspects are sent to `llm`.
    """
    matrix = _Matrix(cells, dimension_keys, _paper_labels(documents))
    outcomes: dict[str, Outcome] = {
        "dataset": evaluate_dataset(matrix),
        "metric": evaluate_metric(matrix),
        "task": evaluate_task(matrix),
        "split": _evaluate_signal("split", matrix, extract_splits, "evaluates on", "evaluation splits"),
        "setup": _evaluate_signal("setup", matrix, extract_setups, "runs", "evaluation setups"),
    }

    caveat_from_llm: str | None = None
    ambiguous = [o for o in outcomes.values() if o.status == "ambiguous" or (o.ambiguous_ids and o.aspect in ("dataset", "metric", "task"))]
    if ambiguous and use_llm and llm is not None:
        settled = [o for o in outcomes.values() if o not in ambiguous]
        resolved, caveat_from_llm = resolve_with_llm(llm, matrix, ambiguous, settled)
        for aspect, outcome in resolved.items():
            outcomes[aspect] = outcome
    # Anything the rules could not read and the LLM did not settle stays unknown.
    for outcome in outcomes.values():
        if outcome.status == "ambiguous":
            outcome.status = "unknown"

    overall = compute_overall(outcomes)
    issues = [issue for aspect in ISSUE_TYPES for issue in outcomes[aspect].issues]
    used_llm = any(o.origin == "gemini" for o in outcomes.values())
    use_llm_caveat = bool(caveat_from_llm) and used_llm and overall in ("not_comparable", "partially_comparable") and any(
        i.origin == "gemini" for i in issues
    )
    caveat = caveat_from_llm if use_llm_caveat else _template_caveat(overall, outcomes, issues)
    return {
        "overall": overall,
        "issues": [issue.to_dict() for issue in issues],
        "caveat": caveat,
        "checked_by": "gemini" if used_llm else "rules",
    }


def cells_from_matrix(matrix) -> list[CellInput]:
    """ORM ComparisonCell rows -> CellInput. A cell is usable only if it has a
    real answer: no error, not flagged insufficient_evidence, not blank."""
    return [
        CellInput(
            document_id=cell.document_id,
            dimension_key=cell.dimension_key,
            text=cell.value_text or "",
            usable=not cell.error and not cell.insufficient_evidence and bool((cell.value_text or "").strip()),
        )
        for cell in matrix.cells
    ]
