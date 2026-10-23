#!/usr/bin/env python3
"""Standalone verification of the document-processing pipeline.

Runs the real parser over real PDFs and asserts on the extracted structure.
Deliberately depends on nothing but the parsing layer — no database, no
FastAPI, no pytest — so it can be run anywhere the backend can:

    # generate the test corpus (needs reportlab + pillow)
    python tests/make_test_pdfs.py

    # verify against it
    python verify_pipeline.py

    # or verify against your own PDFs
    python verify_pipeline.py /path/to/pdf_dir

Exit code is 0 only if every check passes.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.processing import pipeline  # noqa: E402
from app.processing.schema import (  # noqa: E402
    CAPTION,
    FIGURE,
    HEADING,
    PARAGRAPH,
    TABLE,
)

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

passed = 0
failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  {GREEN}PASS{RESET} {label}")
    else:
        failed += 1
        print(f"  {RED}FAIL{RESET} {label}" + (f" {DIM}({detail}){RESET}" if detail else ""))


def describe(doc) -> None:
    s = doc.summary()
    print(
        f"  {DIM}backend={s['backend']} pages={s['pages']} sections={s['sections']} "
        f"elements={s['elements']} tables={s['tables']} figures={s['figures']} "
        f"refs={s['references']} ocr={s['ocr_used']}{RESET}"
    )
    if doc.warnings:
        for w in doc.warnings:
            print(f"  {YELLOW}warn{RESET} {w}")


def section_titles(doc) -> list[str]:
    return [s.title for s in doc.sections]


def verify_structured_paper(path: str) -> None:
    print(f"\n{'=' * 68}\nstructured paper: {os.path.basename(path)}\n{'=' * 68}")
    doc = pipeline.process_pdf(path)
    describe(doc)

    titles = " | ".join(section_titles(doc)).lower()

    check("pages were extracted", doc.page_count >= 1, f"page_count={doc.page_count}")
    check("a title was recovered", bool(doc.title), f"title={doc.title!r}")
    check("headings detected", doc.count_of(HEADING) >= 5, f"n={doc.count_of(HEADING)}")
    check("paragraphs detected", doc.count_of(PARAGRAPH) >= 5, f"n={doc.count_of(PARAGRAPH)}")
    check("abstract section found", "abstract" in titles)
    check("introduction section found", "introduction" in titles)
    check("conclusion section found", "conclusion" in titles)
    check("references section found", "references" in titles)
    check("nested subsection detected", any(s.level >= 2 for s in doc.sections))
    check("a table was extracted", doc.count_of(TABLE) >= 1, f"n={doc.count_of(TABLE)}")
    check("a figure region was detected", doc.count_of(FIGURE) >= 1)
    check("captions detected", doc.count_of(CAPTION) >= 1, f"n={doc.count_of(CAPTION)}")
    check("references parsed", len(doc.references) >= 3, f"n={len(doc.references)}")

    tables = [e for e in doc.elements if e.element_type == TABLE]
    if tables:
        content = tables[0].content
        check("table content is real markdown rows", content.count("|") >= 6)
        check("table cell values came from the PDF", "0.941" in content, content[:80])

    if doc.references:
        with_year = sum(1 for r in doc.references if r.year)
        with_title = sum(1 for r in doc.references if r.title)
        check("most references have a year", with_year >= len(doc.references) * 0.6,
              f"{with_year}/{len(doc.references)}")
        check("most references have a title", with_title >= len(doc.references) * 0.6,
              f"{with_title}/{len(doc.references)}")
        check("a DOI or URL was captured",
              any(r.doi or r.url for r in doc.references))

    check("every element carries a page number",
          all(e.page_number >= 1 for e in doc.elements))
    check("every element is attributed to a section",
          all(e.section_index is not None for e in doc.elements))
    check("section page ranges are coherent",
          all(s.end_page >= s.start_page for s in doc.sections))
    check("no caption swallowed a prose cross-reference",
          not any(
              e.element_type == CAPTION and " shows " in e.content.lower()
              for e in doc.elements
          ))


def verify_plain_report(path: str) -> None:
    print(f"\n{'=' * 68}\nplain report: {os.path.basename(path)}\n{'=' * 68}")
    doc = pipeline.process_pdf(path)
    describe(doc)

    titles = " | ".join(section_titles(doc)).lower()
    check("pages extracted", doc.page_count >= 1)
    check("unnumbered headings detected", doc.count_of(HEADING) >= 4,
          f"n={doc.count_of(HEADING)}")
    check("methods section found", "methods" in titles)
    check("results section found", "results" in titles)
    check("no bibliography invented", len(doc.references) == 0,
          f"n={len(doc.references)}")
    check("placeholder author not accepted as real",
          (doc.authors or "").lower() not in {"anonymous", "unknown", "user"},
          f"authors={doc.authors!r}")


def verify_scanned(path: str) -> None:
    print(f"\n{'=' * 68}\nscanned / image-only: {os.path.basename(path)}\n{'=' * 68}")
    doc = pipeline.process_pdf(path)
    describe(doc)

    check("pages extracted", doc.page_count >= 1)
    check("pages correctly reported as having no text layer",
          all(not p.has_text_layer for p in doc.pages))

    if doc.ocr_used:
        check("OCR was applied", any(p.ocr_applied for p in doc.pages))
        check("OCR recovered real text", doc.word_count > 20, f"words={doc.word_count}")
        check("OCR produced elements", len(doc.elements) > 0)
        check("title recovered from scanned content", bool(doc.title), f"{doc.title!r}")
        check("no absurd heading count from OCR noise",
              doc.count_of(HEADING) <= max(3, len(doc.elements) * 0.6),
              f"headings={doc.count_of(HEADING)} of {len(doc.elements)}")
    else:
        print(f"  {YELLOW}skip{RESET} OCR checks — no OCR engine available here")
        check("missing OCR is reported as a warning", len(doc.warnings) > 0)


def verify_broken(path: str) -> None:
    print(f"\n{'=' * 68}\ncorrupt file: {os.path.basename(path)}\n{'=' * 68}")
    try:
        pipeline.process_pdf(path)
        check("corrupt PDF raises ProcessingError", False, "no exception raised")
    except pipeline.ProcessingError as exc:
        check("corrupt PDF raises ProcessingError", True)
        check("failure reason is reported", len(str(exc)) > 10, str(exc)[:60])
    except Exception as exc:
        check("corrupt PDF raises ProcessingError", False,
              f"raised {type(exc).__name__} instead")


def verify_generic(path: str) -> None:
    print(f"\n{'=' * 68}\n{os.path.basename(path)}\n{'=' * 68}")
    started = time.time()
    try:
        doc = pipeline.process_pdf(path)
    except Exception as exc:
        check("parses without error", False, f"{type(exc).__name__}: {exc}")
        return
    describe(doc)
    print(f"  {DIM}{time.time() - started:.2f}s{RESET}")
    check("pages extracted", doc.page_count >= 1)
    check("some content extracted", len(doc.elements) > 0)
    check("every element has a page number", all(e.page_number >= 1 for e in doc.elements))
    check("section ranges coherent", all(s.end_page >= s.start_page for s in doc.sections))


def main() -> int:
    print(f"parser backends available : {pipeline.available_backends()}")
    print(f"resolved backend order    : {pipeline.resolve_order('auto')}")
    from app.processing import ocr as ocr_module

    print(f"OCR available             : {ocr_module.ocr_available()}")

    if not pipeline.available_backends():
        print(f"\n{RED}No parser backend installed. Run: pip install -r requirements.txt{RESET}")
        return 1

    here = os.path.dirname(os.path.abspath(__file__))
    corpus = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "tests", "testpdfs")

    if not os.path.isdir(corpus):
        print(f"\n{RED}No corpus at {corpus}{RESET}")
        print("Generate it with:  python tests/make_test_pdfs.py")
        return 1

    known = {
        "paper_text.pdf": verify_structured_paper,
        "paper_plain.pdf": verify_plain_report,
        "scanned.pdf": verify_scanned,
        "broken.pdf": verify_broken,
    }

    files = sorted(f for f in os.listdir(corpus) if f.lower().endswith(".pdf"))
    if not files:
        print(f"\n{RED}No PDFs found in {corpus}{RESET}")
        return 1

    for name in files:
        path = os.path.join(corpus, name)
        (known.get(name) or verify_generic)(path)

    total = passed + failed
    colour = GREEN if failed == 0 else RED
    print(f"\n{'=' * 68}")
    print(f"{colour}{passed}/{total} checks passed{RESET}")
    print(f"{'=' * 68}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
