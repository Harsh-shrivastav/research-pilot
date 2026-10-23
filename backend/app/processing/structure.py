"""Turn a flat, ordered list of elements into a section hierarchy.

Every backend emits elements in reading order with headings marked. This module
is what converts that stream into `document_sections` rows and stamps each
element with the section it belongs to.
"""
from __future__ import annotations

from .schema import (
    CAPTION,
    FIGURE,
    FORMULA,
    HEADING,
    LIST_ITEM,
    PARAGRAPH,
    TABLE,
    ParsedElement,
    ParsedSection,
)

# Element types whose content is worth concatenating into section text.
_TEXTUAL = {PARAGRAPH, LIST_ITEM, CAPTION, FORMULA}

FRONT_MATTER_TITLE = "Front matter"


def build_sections(
    elements: list[ParsedElement], page_count: int = 0
) -> list[ParsedSection]:
    """Assign `section_index` to every element and return the sections.

    Mutates `elements` in place (setting `section_index`) and returns the
    ordered section list. A synthetic "Front matter" section captures anything
    appearing before the first heading — title blocks, author lists, abstracts
    in papers whose abstract heading is not detected, and so on.
    """
    sections: list[ParsedSection] = []
    # stack of (level, section_index) for parent resolution
    stack: list[tuple[int, int]] = []
    current: int | None = None
    buffers: dict[int, list[str]] = {}

    for element in elements:
        if element.element_type == HEADING:
            level = element.level or 1
            index = len(sections)
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent = stack[-1][1] if stack else None
            sections.append(
                ParsedSection(
                    index=index,
                    title=element.content.strip() or f"Section {index + 1}",
                    level=level,
                    order_index=index,
                    start_page=element.page_number,
                    end_page=element.page_number,
                    parent_index=parent,
                )
            )
            stack.append((level, index))
            buffers[index] = []
            current = index
            element.section_index = index
            continue

        if current is None:
            # Content before the first heading.
            current = len(sections)
            sections.append(
                ParsedSection(
                    index=current,
                    title=FRONT_MATTER_TITLE,
                    level=1,
                    order_index=current,
                    start_page=element.page_number,
                    end_page=element.page_number,
                    parent_index=None,
                )
            )
            buffers[current] = []
            stack.append((1, current))

        element.section_index = current
        section = sections[current]
        section.end_page = max(section.end_page, element.page_number)
        section.start_page = min(section.start_page, element.page_number)

        if element.element_type in _TEXTUAL and element.content.strip():
            buffers[current].append(element.content.strip())
        elif element.element_type == TABLE and element.content.strip():
            buffers[current].append(element.content.strip())
        elif element.element_type == FIGURE:
            label = element.meta.get("label") or "figure"
            buffers[current].append(f"[{label}]")

    for section in sections:
        section.text = "\n\n".join(buffers.get(section.index, []))

    _extend_end_pages(sections, page_count)
    return sections


def _extend_end_pages(sections: list[ParsedSection], page_count: int) -> None:
    """A section with no body content would otherwise end on its own heading
    page. Stretch each section up to just before the next same-or-higher level
    section begins, so page ranges stay contiguous and useful for the PDF
    viewer's jump-to-section behaviour."""
    for i, section in enumerate(sections):
        next_start: int | None = None
        for later in sections[i + 1 :]:
            if later.level <= section.level:
                next_start = later.start_page
                break

        if next_start is not None:
            # The next section may start on the same page this one ends on.
            candidate = max(next_start - 1, section.start_page)
        elif page_count:
            candidate = page_count
        else:
            candidate = section.end_page

        section.end_page = max(section.end_page, candidate, section.start_page)


def section_page_span(section: ParsedSection) -> tuple[int, int]:
    return section.start_page, max(section.end_page, section.start_page)
