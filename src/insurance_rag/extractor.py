"""
extractor.py
------------
Phase 1 of the pipeline: open the PDF and pull out every text block
with its font metadata and bounding-box coordinates.

Responsibility: raw extraction ONLY.
  - No cleaning (that's cleaner.py)
  - No section logic (that's reconstructor.py)
  - No chunking (that's chunker.py)
"""

import fitz  # PyMuPDF
from pathlib import Path

from .models import TextBlock


# ---------------------------------------------------------------------------
# Font-size / layout thresholds
# These are the values to tune when adapting to a new insurer's PDF.
# ---------------------------------------------------------------------------

HEADING_FONT_MIN_PT = 10.5      # blocks at or above this size → candidate heading
FOOTER_ZONE_PT      = 50        # pts from page bottom → footer zone
HEADER_ZONE_PT      = 60        # pts from page top    → running header zone
HEADER_MAX_CHARS    = 80        # short blocks in header zone only


class PDFExtractor:
    """
    Opens a PDF with PyMuPDF and extracts TextBlock objects page by page.

    Usage
    -----
    with PDFExtractor("policy.pdf") as ext:
        blocks   = ext.extract_blocks()
        metadata = ext.get_document_metadata()
        toc      = ext.get_toc()
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self._doc: fitz.Document | None = None

    # ── context manager ───────────────────────────────────────────────────

    def __enter__(self):
        self._doc = fitz.open(str(self.pdf_path))
        return self

    def __exit__(self, *_):
        if self._doc:
            self._doc.close()

    # ── public methods ────────────────────────────────────────────────────

    def extract_blocks(self) -> list[TextBlock]:
        """
        Iterate every page and return a flat list of TextBlock objects.

        PyMuPDF's get_text("dict") returns:
            page → blocks → lines → spans
        Each span has: text, font name, font size, bounding box.
        We aggregate spans up to block level, keeping the max font size
        and OR-ing the bold flag across all spans.
        """
        self._require_open()
        blocks: list[TextBlock] = []

        for page_idx in range(len(self._doc)):
            page        = self._doc[page_idx]
            page_height = page.rect.height
            page_dict   = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

            for b_idx, raw_block in enumerate(page_dict["blocks"]):

                # block["type"] == 1 means an image block — skip entirely
                if raw_block["type"] != 0:
                    continue

                text, max_font_size, is_bold = self._parse_block_spans(raw_block)

                if not text.strip():
                    continue

                bbox       = tuple(raw_block["bbox"])
                block_type = self._classify_block(
                    text, max_font_size, is_bold,
                    y0=bbox[1], y1=bbox[3],
                    page_height=page_height,
                )

                blocks.append(TextBlock(
                    page_num   = page_idx + 1,   # 1-indexed for human readability
                    block_num  = b_idx,
                    bbox       = bbox,
                    text       = text,
                    block_type = block_type,
                    font_size  = max_font_size,
                    is_bold    = is_bold,
                ))

        return blocks

    def get_document_metadata(self) -> dict:
        """Return PDF-level metadata (title, author, page count, etc.)."""
        self._require_open()
        meta = self._doc.metadata or {}
        return {
            "title"      : meta.get("title",   ""),
            "author"     : meta.get("author",  ""),
            "subject"    : meta.get("subject", ""),
            "creator"    : meta.get("creator", ""),
            "page_count" : len(self._doc),
            "source_file": self.pdf_path.name,
        }

    def get_toc(self) -> list[dict]:
        """
        Return the PDF's built-in table of contents (bookmarks).
        Each entry: {"level": int, "title": str, "page": int}
        Empty list if the PDF has no bookmarks.
        """
        self._require_open()
        return [
            {"level": lvl, "title": title, "page": page}
            for lvl, title, page in self._doc.get_toc()
        ]

    # ── private helpers ───────────────────────────────────────────────────

    def _require_open(self):
        """Ensure the document is loaded before performing operations."""
        if self._doc is None:
            raise RuntimeError(
                f"PDFExtractor is not open. Use 'with PDFExtractor(\"{self.pdf_path}\") as ext:'"
            )

    @staticmethod
    def _parse_block_spans(raw_block: dict) -> tuple[str, float, bool]:
        """
        Walk block → lines → spans and return:
          (concatenated_text, max_font_size, any_span_is_bold)
        """
        lines_text   = []
        max_font_size = 0.0
        is_bold       = False

        for line in raw_block["lines"]:
            line_text = ""
            for span in line["spans"]:
                line_text    += span["text"]
                max_font_size = max(max_font_size, span["size"])
                if "bold" in span["font"].lower():
                    is_bold = True
            lines_text.append(line_text)

        return "\n".join(lines_text), max_font_size, is_bold

    @staticmethod
    def _classify_block(
        text: str,
        font_size: float,
        is_bold: bool,
        y0: float,
        y1: float,
        page_height: float,
    ) -> str:
        """
        Assign one of four labels to a block using positional + font heuristics.

        "footer"  — bottom FOOTER_ZONE_PT of the page
        "header"  — top HEADER_ZONE_PT of the page + short text
        "heading" — large font OR bold + short + all-caps
        "text"    — everything else (body copy)

        Tuning guide
        ------------
        If real headings are being classified as "text":
            → lower HEADING_FONT_MIN_PT
        If body copy is being classified as "heading":
            → raise HEADING_FONT_MIN_PT, or tighten the len/uppercase check
        If footers are leaking into body:
            → raise FOOTER_ZONE_PT
        """

        # ── footer zone ────────────────────────────────────────────────
        if y1 > (page_height - FOOTER_ZONE_PT):
            return "footer"

        # ── running header zone ────────────────────────────────────────
        if y0 < HEADER_ZONE_PT and len(text) < HEADER_MAX_CHARS:
            return "header"

        # ── heading detection ──────────────────────────────────────────
        large_font   = font_size >= HEADING_FONT_MIN_PT
        caps_bold    = is_bold and len(text) < 120 and text.strip().upper() == text.strip()

        if large_font or caps_bold:
            return "heading"

        return "text"
