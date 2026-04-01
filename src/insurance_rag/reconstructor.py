"""
reconstructor.py
----------------
Phase 2 of the pipeline: group raw TextBlocks into logical sections.

A "section" here means:  one heading  +  all the body text that follows it
until the next heading is detected.

Responsibility:
  - Detect heading boundaries (font-based classification + regex fallback)
  - Flush and start new sections at each heading
  - Track which pages each section spans
  - Parse section / clause labels from heading text
"""

import re

from .models import TextBlock
from .cleaner import clean_block_text


# ---------------------------------------------------------------------------
# Compiled regex patterns for insurance document structure
# ---------------------------------------------------------------------------

# Major section headings, e.g.:
#   "SECTION 3 – EXCLUSIONS"
#   "PART B – GENERAL CONDITIONS"
#   "2. DEFINITIONS"
#   "4. BENEFITS AND COVERAGE"
_SECTION_PATTERNS = [
    re.compile(
        r"^(SECTION|PART|CHAPTER)\s+[A-Z0-9]+[\s\-–:]+(.+)$", re.IGNORECASE
    ),
    re.compile(
        r"^(\d+\.?\s+)(DEFINITIONS|BENEFITS|COVERAGE|EXCLUSIONS|CONDITIONS|"
        r"CLAIMS|PREMIUM|RENEWABILITY|GENERAL|WAITING PERIOD|"
        r"HOSPITALIZATION|SPECIAL|OPTIONAL|SCHEDULE)",
        re.IGNORECASE,
    ),
]

# Numbered clause headings, e.g.:  "4.2.1 Waiting Period"
_CLAUSE_PATTERN = re.compile(r"^(\d+\.\d+[\.\d]*)\s+(.+)$")

# Keywords that identify benefit / schedule tables — used as metadata flags
_SCHEDULE_KEYWORDS = re.compile(
    r"(schedule of benefits|sum insured|deductible|co-payment|sub-limit)",
    re.IGNORECASE,
)

# List-item prefixes — used as metadata flags
_LIST_ITEM_PATTERN = re.compile(
    r"^(\s*[•\-\*◦▪➢→]|\s*\([a-z]\)|\s*[ivxIVX]+\.|\s*\d+\.)\s+"
)


# ---------------------------------------------------------------------------
# SectionReconstructor
# ---------------------------------------------------------------------------

class SectionReconstructor:
    """
    Consumes a list of TextBlock objects and emits a list of section dicts.

    Each section dict:
    {
        "section"    : str   — top-level label, e.g. "EXCLUSIONS"
        "sub_section": str   — clause number, e.g. "4.2.1"
        "heading"    : str   — full heading text
        "pages"      : set   — all page numbers this section spans
        "text"       : str   — concatenated cleaned body text
    }
    """

    def reconstruct(self, blocks: list[TextBlock]) -> list[dict]:
        """
        Main entry point.  Iterates blocks in reading order and
        accumulates body text under the most recently seen heading.
        """
        sections: list[dict] = []
        current = _new_section()

        for block in blocks:

            # Headers and footers were already classified — discard them
            if block.block_type in ("header", "footer"):
                continue

            text = clean_block_text(block.text)
            if not text:
                continue

            # ── is this block a heading? ───────────────────────────────
            if block.block_type == "heading" or self._is_major_heading(text, block):

                # Flush the current accumulator if it has content
                if current["text"].strip():
                    sections.append(current)

                label   = _extract_section_label(text)
                current = _new_section(
                    section     = label["section"],
                    sub_section = label["sub_section"],
                    heading     = text,
                    page        = block.page_num,
                )
                continue

            # ── body text: accumulate ──────────────────────────────────
            current["text"] += text + "\n"
            current["pages"].add(block.page_num)

        # Don't forget the final section — it never triggers a heading flush
        if current["text"].strip():
            sections.append(current)

        return sections

    # ── private ───────────────────────────────────────────────────────────

    @staticmethod
    def _is_major_heading(text: str, block: TextBlock) -> bool:
        """
        Second-chance heading detection via regex.

        A block may have missed the font-size threshold in _classify_block
        (e.g. the insurer uses 10pt bold for section headings) but still
        match a known structural pattern.
        """
        for pattern in _SECTION_PATTERNS:
            if pattern.match(text):
                return True

        # Numbered clause AND (bold OR large font)
        if _CLAUSE_PATTERN.match(text) and (
            block.is_bold or block.font_size >= 10.5
        ):
            return True

        return False


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _new_section(
    section: str = "PREAMBLE",
    sub_section: str = "",
    heading: str = "",
    page: int = 1,
) -> dict:
    """Create a fresh section accumulator dict."""
    return {
        "section"    : section,
        "sub_section": sub_section,
        "heading"    : heading,
        "pages"      : {page},   # set — deduplicates page numbers automatically
        "text"       : "",
    }


def _extract_section_label(text: str) -> dict:
    """
    Parse section and sub_section strings from a heading.

    Returns {"section": str, "sub_section": str}
    Falls back to using the full (truncated) heading as the section label.
    """
    for pattern in _SECTION_PATTERNS:
        m = pattern.match(text)
        if m:
            return {"section": text.strip(), "sub_section": ""}

    m = _CLAUSE_PATTERN.match(text)
    if m:
        return {
            "section"    : m.group(2).strip(),
            "sub_section": m.group(1).strip(),
        }

    return {"section": text[:60].strip(), "sub_section": ""}
