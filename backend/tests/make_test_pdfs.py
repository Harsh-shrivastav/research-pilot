"""Generate a small corpus of realistic test PDFs for the Phase 2 pipeline.

Produces:
  1. paper_text.pdf    - multi-page research paper: headings, numbered sections,
                         a table, a figure box + captions, and a [n] bibliography
  2. paper_plain.pdf   - simple prose report with canonical (unnumbered) headings
  3. scanned.pdf       - image-only PDF (no text layer at all) -> exercises OCR
  4. broken.pdf        - not a real PDF -> exercises the FAILED state
"""
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas as pdfcanvas
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testpdfs")
os.makedirs(OUT, exist_ok=True)

BODY = "Helvetica"
BOLD = "Helvetica-Bold"
M = inch


class Writer:
    def __init__(self, path, title=None, author=None):
        self.c = pdfcanvas.Canvas(path, pagesize=LETTER)
        if title:
            self.c.setTitle(title)
        if author:
            self.c.setAuthor(author)
        self.w, self.h = LETTER
        self.y = self.h - M
        self.page = 1

    def _room(self, need):
        if self.y - need < M:
            self.footer()
            self.c.showPage()
            self.page += 1
            self.y = self.h - M

    def footer(self):
        self.c.setFont(BODY, 9)
        self.c.drawCentredString(self.w / 2, M * 0.55, str(self.page))
        self.c.setFont(BODY, 9)
        self.c.drawString(M, self.h - M * 0.6, "ResearchPilot Test Corpus")

    def heading(self, text, size=13):
        self._room(size + 16)
        self.y -= size + 8
        self.c.setFont(BOLD, size)
        self.c.drawString(M, self.y, text)
        self.y -= 6

    def para(self, text, size=10.5, leading=14):
        words = text.split()
        line = ""
        maxw = self.w - 2 * M
        self.c.setFont(BODY, size)
        lines = []
        for word in words:
            trial = (line + " " + word).strip()
            if self.c.stringWidth(trial, BODY, size) > maxw:
                lines.append(line)
                line = word
            else:
                line = trial
        if line:
            lines.append(line)
        for ln in lines:
            self._room(leading)
            self.c.setFont(BODY, size)
            self.y -= leading
            self.c.drawString(M, self.y, ln)
        self.y -= 4

    def caption(self, text, size=9.5):
        self._room(20)
        self.y -= 14
        self.c.setFont(BODY, size)
        self.c.drawString(M, self.y, text)
        self.y -= 6

    def table(self, rows, col_w=None):
        n = len(rows[0])
        col_w = col_w or [(self.w - 2 * M) / n] * n
        rh = 18
        self._room(rh * len(rows) + 10)
        top = self.y
        for r, row in enumerate(rows):
            x = M
            yy = top - (r + 1) * rh
            for ci, cell in enumerate(row):
                self.c.setStrokeColor(colors.black)
                self.c.setLineWidth(0.6)
                self.c.rect(x, yy, col_w[ci], rh, stroke=1, fill=0)
                self.c.setFont(BOLD if r == 0 else BODY, 9.5)
                self.c.drawString(x + 4, yy + 5, str(cell))
                x += col_w[ci]
        self.y = top - rh * len(rows) - 8

    def figure_image(self, w=3.6 * inch, h=2.0 * inch):
        """Draw a real raster image so it registers as an embedded image."""
        self._room(h + 14)
        img = Image.new("RGB", (int(w * 2), int(h * 2)), (245, 247, 252))
        d = ImageDraw.Draw(img)
        for i, val in enumerate([40, 95, 150, 122, 190, 210]):
            x0 = 30 + i * 55
            d.rectangle([x0, img.height - 30 - val, x0 + 38, img.height - 30],
                        fill=(59, 92, 173))
        d.line([20, img.height - 30, img.width - 20, img.height - 30], fill=(0, 0, 0), width=3)
        tmp = os.path.join(OUT, "_fig_tmp.png")
        img.save(tmp)
        self.y -= h
        self.c.drawImage(tmp, M, self.y, width=w, height=h)
        self.y -= 6

    def save(self):
        self.footer()
        self.c.save()


def paper_text():
    w = Writer(os.path.join(OUT, "paper_text.pdf"),
               title="Retrieval-Augmented Structuring of Scientific PDFs",
               author="A. Mehta, R. Okonkwo, L. Fischer")
    w.c.setFont(BOLD, 16)
    w.y -= 20
    w.c.drawString(M, w.y, "Retrieval-Augmented Structuring of Scientific PDFs")
    w.y -= 22
    w.c.setFont(BODY, 11)
    w.c.drawString(M, w.y, "A. Mehta, R. Okonkwo, L. Fischer")
    w.y -= 14
    w.c.drawString(M, w.y, "Institute for Document Intelligence")
    w.y -= 10

    w.heading("Abstract", 12)
    w.para("We present a pipeline that converts unstructured research PDFs into a "
           "queryable structured representation. Our system recovers the heading "
           "hierarchy, table regions, figure regions and bibliography entries "
           "without relying on publisher-specific templates. Evaluated on a corpus "
           "of 1,200 documents, the pipeline recovers 94.1 percent of section "
           "boundaries and 88.7 percent of table regions.")

    w.heading("1. Introduction")
    w.para("Scientific communication remains dominated by PDF, a format optimised "
           "for fixed-layout rendering rather than semantic access. Downstream "
           "tasks such as retrieval, summarisation and citation analysis all "
           "require structure that the format does not preserve. Recovering that "
           "structure is therefore a prerequisite rather than an enhancement.")
    w.para("Prior approaches fall into two families. Rule-based extractors exploit "
           "font metrics and whitespace, while learned layout models predict "
           "region classes directly from rendered pages. We combine both.")

    w.heading("2. Related Work")
    w.para("Early systems such as GROBID applied conditional random fields to "
           "header and reference parsing. More recent layout models operate on "
           "page images and achieve stronger generalisation across publishers, at "
           "the cost of substantially higher inference latency.")

    w.heading("3. Method")
    w.para("Our pipeline proceeds in four stages: page decoding, element "
           "detection, hierarchy assembly and reference resolution. Each stage "
           "emits provenance so that any extracted span can be traced back to its "
           "originating page and bounding box.")

    w.heading("3.1 Element Detection")
    w.para("Word-level geometry is clustered into lines, then lines into blocks. "
           "A block is promoted to a heading when its median font size exceeds "
           "the modal body size by at least twelve percent, or when it carries a "
           "hierarchical numeric prefix.")

    w.heading("3.2 Hierarchy Assembly")
    w.para("Detected headings are pushed onto a level stack. Each non-heading "
           "element is attributed to the section currently on top of the stack, "
           "which yields a tree without requiring a separate parsing pass.")

    w.heading("4. Experiments")
    w.para("We evaluate on a held-out set of 300 documents spanning eight "
           "publishers. The table below reports section-boundary recovery by "
           "document class.")
    w.table([
        ["Document class", "Documents", "Boundary F1", "Table IoU"],
        ["Two-column", "140", "0.941", "0.887"],
        ["Single-column", "96", "0.962", "0.901"],
        ["Scanned", "64", "0.788", "0.712"],
    ])
    w.caption("Table 1: Section boundary and table region recovery by class.")

    w.para("Figure 1 shows throughput as a function of page count. Latency grows "
           "linearly, dominated by rasterisation on scanned inputs.")
    w.figure_image()
    w.caption("Figure 1: Processing throughput across the evaluation corpus.")

    w.heading("5. Discussion")
    w.para("The residual error on scanned documents is dominated by OCR noise in "
           "multi-column layouts, where line merging across the column gutter "
           "produces spurious paragraph boundaries.")

    w.heading("6. Limitations")
    w.para("We do not attempt formula recognition beyond region detection, and "
           "our reference parser assumes a Latin-script bibliography.")

    w.heading("7. Conclusion")
    w.para("Structured extraction from PDF is tractable with a hybrid of "
           "geometric heuristics and learned layout priors. We release our "
           "pipeline to support reproducible document intelligence research.")

    w.heading("References")
    refs = [
        "[1] Lopez, P. GROBID: Combining Automatic Bibliographic Data Recognition "
        "and Term Extraction. Proceedings of ECDL, 2009. doi:10.1007/978-3-642-04346-8_62",
        "[2] Zhong, X., Tang, J., and Jimeno-Yepes, A. PubLayNet: Largest Dataset "
        "Ever for Document Layout Analysis. ICDAR, 2019.",
        "[3] Auer, C., Nassar, A., and Livathinos, N. Docling Technical Report. "
        "arXiv preprint, 2024. https://arxiv.org/abs/2408.09869",
        "[4] Smith, R. An Overview of the Tesseract OCR Engine. ICDAR, 2007.",
        "[5] Shen, Z., Zhang, R., and Dell, M. LayoutParser: A Unified Toolkit for "
        "Deep Learning Based Document Image Analysis. ICDAR, 2021.",
    ]
    for r in refs:
        w.para(r, size=9.5, leading=12)
    w.save()


def paper_plain():
    w = Writer(os.path.join(OUT, "paper_plain.pdf"), title="Quarterly Research Report")
    w.c.setFont(BOLD, 15)
    w.y -= 18
    w.c.drawString(M, w.y, "Quarterly Research Report")
    w.y -= 16

    for title, body in [
        ("Introduction",
         "This report summarises activity for the quarter across the three active "
         "research programmes and highlights deviations from the original plan."),
        ("Background",
         "The programme was chartered to evaluate document understanding methods "
         "for internal archives containing several million scanned pages."),
        ("Methods",
         "We sampled two thousand pages stratified by decade of origin and "
         "annotated them for layout class, then measured extraction quality "
         "against those annotations."),
        ("Results",
         "Extraction quality on born-digital pages exceeded the target threshold. "
         "Scanned pages from before 1985 remain substantially below target, "
         "largely because of degraded typography and skew."),
        ("Discussion",
         "The gap between born-digital and legacy scanned material argues for a "
         "dedicated preprocessing stage rather than a single unified model."),
        ("Conclusion",
         "We recommend proceeding to a limited production pilot restricted to "
         "born-digital material while preprocessing work continues."),
    ]:
        w.heading(title, 12.5)
        w.para(body)
        w.para("Additional supporting detail is retained in the programme "
               "appendix and is available on request from the research office.")
    w.save()


def scanned():
    """Render text to raster images and wrap them in a PDF: no text layer."""
    pages = [
        ["ANNUAL SURVEY OF DOCUMENT ARCHIVES", "",
         "1. Overview", "",
         "This page exists only as a raster image. There is no",
         "embedded text layer, so any text recovered from it",
         "must come from optical character recognition.", "",
         "The survey covered forty seven institutional archives",
         "between January and September."],
        ["2. Findings", "",
         "Most archives reported that scanned holdings exceed",
         "born digital holdings by a wide margin. Only nine",
         "institutions had completed full text indexing.", "",
         "3. Conclusion", "",
         "Optical character recognition remains essential for",
         "access to the majority of archival material."],
    ]
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
    except Exception:
        font = ImageFont.load_default()

    images = []
    for lines in pages:
        img = Image.new("RGB", (1700, 2200), "white")
        d = ImageDraw.Draw(img)
        y = 180
        for ln in lines:
            d.text((160, y), ln, fill=(15, 15, 15), font=font)
            y += 52
        images.append(img.convert("RGB"))

    images[0].save(os.path.join(OUT, "scanned.pdf"), save_all=True,
                   append_images=images[1:], resolution=200.0)


def broken():
    with open(os.path.join(OUT, "broken.pdf"), "wb") as f:
        f.write(b"%PDF-1.4\nthis file is deliberately corrupt and cannot be parsed\n")


if __name__ == "__main__":
    paper_text()
    paper_plain()
    scanned()
    broken()
    tmp = os.path.join(OUT, "_fig_tmp.png")
    if os.path.exists(tmp):
        os.remove(tmp)
    for name in sorted(os.listdir(OUT)):
        print(name, os.path.getsize(os.path.join(OUT, name)), "bytes")
