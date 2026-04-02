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
from .patterns import _MAJOR_SECTION_RE, _CLAUSE_RE, _NOT_A_SECTION


# ---------------------------------------------------------------------------
# Font-size / layout thresholds
# These are the values to tune when adapting to a new insurer's PDF.
# ---------------------------------------------------------------------------

HEADING_FONT_MIN_PT = 12      # blocks at or above this size → candidate heading
FOOTER_ZONE_PT      = 50        # pts from page bottom → footer zone
HEADER_ZONE_PT      = 90        # pts from page top    → running header zone
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

                text, bold_text, max_font_size, is_bold, bold_ratio = self._parse_block_spans(raw_block)

                if not text.strip():
                    continue

                bbox = tuple(raw_block["bbox"])

                block_type = self._classify_block(
                    text, max_font_size, bold_ratio,
                    y0=bbox[1], y1=bbox[3],
                    page_height=page_height,
                )

                if block_type == "subheading":
                    blocks.append(TextBlock(
                        page_num   = page_idx + 1,   # 1-indexed for human readability
                        block_num  = b_idx,
                        bbox       = bbox,
                        text       = bold_text,
                        block_type = block_type,
                        font_size  = max_font_size,
                        is_bold    = is_bold,
                        bold_ratio = bold_ratio,
                    ))
                    block_type = "text"  # also add the full text as a regular block for chunking
                    text = text[len(bold_text):].strip()
                
                if not text.strip():
                    continue

                blocks.append(TextBlock(
                    page_num   = page_idx + 1,   # 1-indexed for human readability
                    block_num  = b_idx,
                    bbox       = bbox,
                    text       = text,
                    block_type = block_type,
                    font_size  = max_font_size,
                    is_bold    = is_bold,
                    bold_ratio = bold_ratio,
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
            {"level": entry[0], "title": entry[1], "page": entry[2]}
            for entry in self._doc.get_toc()
        ]

    # ── private helpers ───────────────────────────────────────────────────

    def _require_open(self):
        """Ensure the document is loaded before performing operations."""
        if self._doc is None:
            raise RuntimeError(
                f"PDFExtractor is not open. Use 'with PDFExtractor(\"{self.pdf_path}\") as ext:'"
            )

    @staticmethod
    def _parse_block_spans(raw_block: dict) -> tuple[str, float, float]:
        """
        Walk block → lines → spans and return:
          (concatenated_text, max_font_size, bold_ratio)
        """
        lines_text   = []
        max_font_size = 0.0
        is_bold = False
        total_chars = 0
        bold_chars = 0
        bold_text = ""

        for line in raw_block["lines"]:
            line_text = ""
            for span in line["spans"]:
                line_text    += span["text"]
                max_font_size = max(max_font_size, span["size"])
                char_count    = len(span["text"].strip())
                total_chars   += char_count
                if "bold" in span["font"].lower():
                    is_bold = True
                    bold_chars += char_count
                if bold_chars == total_chars and char_count > 0:
                    bold_text += span["text"].strip()
            lines_text.append(line_text)

        bold_ratio = float(bold_chars) / float(total_chars) if total_chars > 0 else 0.0
        return "\n".join(lines_text), bold_text, max_font_size, is_bold, bold_ratio
    

    @staticmethod
    def _classify_block(
        text: str,
        font_size: float,
        bold_ratio: float,
        y0: float,
        y1: float,
        page_height: float,
    ) -> str:
        # ── footer / header zones (unchanged) ─────────────────────────────
        if y1 > (page_height - FOOTER_ZONE_PT):
            return "footer"
        if y0 < HEADER_ZONE_PT:
            return "header"

        # Normalise newlines for pattern matching (e.g. "2.\nDefinitions")
        normalised = text.replace("\n", " ").strip()
        text_len   = len(normalised)

        # ── fully bold ─────────────────────────────────────────────────────
        if bold_ratio > 0.9:

            # Blocklist: bold but definitely not a section
            if _NOT_A_SECTION.match(normalised):
                return "text"

            # Matches a known major section pattern → L1 heading
            if _MAJOR_SECTION_RE.match(normalised) or font_size >= HEADING_FONT_MIN_PT:
                return "heading"

            # Matches a clause pattern → L2 subheading
            # (e.g. "2.1.1. Accidental Bodily Injury" or "(i) Pre-existing")
            if _CLAUSE_RE.match(normalised):
                return "subheading"

            # Fully bold, short, no pattern match
            # Could be a named section like "BASE BENEFITS" or "Standard Definitions:"
            # Use all-caps or ends-with-colon as tiebreakers
            # if normalised.upper() == normalised or normalised.endswith(":"):
            #     return "heading"

            # Fully bold, short, mixed case, no pattern — safest bet is subheading
            return "subheading"

        # ── partially bold ─────────────────────────────────────────────────
        if bold_ratio > 0.1:
            # Partially bold blocks are subheadings only if short
            # Long partially-bold blocks are just body text with inline emphasis
            return "subheading"

        return "text"
