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

        # State tracking for hierarchy
        active_major_section = "PREAMBLE"
        
        # Initialize with the default starting section
        current = _new_section(section=active_major_section)

        for block in blocks:
            if block.block_type in ("header", "footer"):
                continue

            text = clean_block_text(block.text)
            if not text:
                continue

            # ── 1. MAJOR HEADING (L1) ──────────────────────────────────
            # Resets the "Parent" context
            if block.block_type == "heading":
                if current["text"].strip() or current["heading"]:
                    sections.append(current)

                label = _extract_section_label(text, "heading")
                # Only update active_major_section if this is truly a major heading
                if label["is_major"]:
                    active_major_section = label["section"]
                
                current = _new_section(
                    section     = active_major_section,
                    sub_section = "",
                    heading     = text,
                    page        = block.page_num,
                )
                continue

            # ── 2. SUBHEADING (L2) ─────────────────────────────────────
            # Starts a new chunk but INHERITS the active_major_section
            if block.block_type == "subheading":
                # Only flush if the current chunk has some content 
                # (Prevents empty sections when a Heading is followed by a Subheading)
                if current["text"].strip() or current["heading"]:
                    sections.append(current)

                label = _extract_section_label(text, "subheading")
                # If it's a subheading, 'section' in label usually contains the title
                # We move that title to sub_section or heading to keep the Parent intact
                current = _new_section(
                    section     = active_major_section, 
                    sub_section = label["sub_section"],
                    heading     = text,
                    page        = block.page_num,
                )
                continue

            # ── 3. BODY TEXT ACCUMULATION ──────────────────────────────
            current["text"] += text + "\n"
            current["pages"].add(block.page_num)

        # Final flush for the last block
        if current["text"].strip() or current["heading"]:
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

        if len(text) > 120: return False

        for pattern in _SECTION_PATTERNS:
            if pattern.match(text):
                return True

        # # Numbered clause AND (bold OR large font)
        # if _CLAUSE_PATTERN.match(text) and (
        #     block.is_bold or block.font_size >= 10.5
        # ):
        #     return True

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


def _extract_section_label(text: str, block_type: str) -> dict:
    """
    Parse section and sub_section from a heading string.

    Returns {"section": str, "sub_section": str, "is_major": bool}

    "is_major" tells the reconstructor whether to update active_major_section.
    If False, the caller should keep the existing parent context.
    """
    # Normalise embedded newlines before matching
    normalised = text.replace("\n", " ").strip()

    # 1. Major section pattern — e.g. "4. Exclusions", "SECTION 3 – BENEFITS"
    # for pattern in _SECTION_PATTERNS:
    #     m = pattern.match(normalised)
    #     if m:
    #         return {
    #             "section"   : normalised,
    #             "sub_section": "",
    #             "is_major"  : True,
    #         }

    if block_type == "heading":
        return {
            "section"   : normalised,
            "sub_section": "",
            "is_major"  : True,
        }

    # 2. Clause pattern — e.g. "2.1.1. Accidental Bodily Injury"
    # m = _CLAUSE_PATTERN.match(normalised)
    # if m and block_type == "subheading":
    #     return {
    #         "section"    : "",        # full text: "2.1.1. Accidental/Accident"
    #         "sub_section": normalised, # just the number: "2.1.1."
    #         "is_major"   : False,
    #     }

    # 3. No pattern matched — this is a bold phrase, not a structural heading
    #    Return is_major=False so active_major_section is preserved
    return {
        "section"   : "",
        "sub_section": normalised,  # put the full text here since we have no better info
        "is_major"  : False,
    }
