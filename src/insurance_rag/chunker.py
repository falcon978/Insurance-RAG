"""
chunker.py
----------
Phase 3 of the pipeline: split sections into RAG-ready PolicyChunk objects.

Chunking strategy (in priority order):
  1. If the section fits within chunk_size — emit as-is (no split needed)
  2. Split on paragraph boundaries (\n\n) in the second half of the window
  3. Fall back to sentence boundaries (. ) in the second half of the window
  4. Hard cut at chunk_size as last resort

Overlap: consecutive chunks share `overlap` characters so that a sentence
spanning a boundary isn't lost to either chunk's embedding.
"""

import hashlib
import re

from .models import PolicyChunk
from .cleaner import clean_section_text


class Chunker:
    """
    Splits a list of section dicts (from SectionReconstructor) into
    PolicyChunk objects ready for embedding.

    Parameters
    ----------
    chunk_size  : target character count per chunk  (default 800)
    overlap     : characters of overlap between consecutive chunks (default 100)
    source_file : PDF filename — stored on every chunk for provenance
    """

    def __init__(
        self,
        chunk_size: int  = 800,
        overlap: int     = 100,
        source_file: str = "",
    ):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size  = chunk_size
        self.overlap     = overlap
        self.source_file = source_file

    # ── public ────────────────────────────────────────────────────────────

    def chunk(self, sections: list[dict]) -> list[PolicyChunk]:
        """Convert all sections into a flat list of PolicyChunk objects."""
        all_chunks: list[PolicyChunk] = []

        for section in sections:
            text = clean_section_text(section["text"])
            if not text:
                continue

            pages      = sorted(section["pages"])
            sub_chunks = self._split(text)

            for idx, chunk_text in enumerate(sub_chunks):
                chunk_id = _make_chunk_id(section["heading"], idx, chunk_text)

                all_chunks.append(PolicyChunk(
                    chunk_id       = chunk_id,
                    source_file    = self.source_file,
                    page_start     = pages[0],
                    page_end       = pages[-1],
                    section        = section["section"],
                    sub_section    = section["sub_section"],
                    heading        = section["heading"],
                    text           = chunk_text,
                    token_estimate = len(chunk_text) // 4,
                    metadata       = {
                        "chunk_index"            : idx,
                        "total_chunks_in_section": len(sub_chunks),
                        "has_table"  : _has_table(chunk_text),
                        "has_list"   : _has_list(chunk_text),
                        "char_count" : len(chunk_text),
                    },
                ))

        return all_chunks

    # ── private ───────────────────────────────────────────────────────────

    def _split(self, text: str) -> list[str]:
        """
        Sliding-window splitter with graceful boundary detection.

        The window advances by (chunk_size - overlap) characters each step,
        but the actual cut point is nudged backwards to the nearest paragraph
        or sentence break in the second half of the window.

        Why "second half"?
            If we allowed cuts in the first half, a very early break could
            produce a tiny chunk.  Requiring the break to be past the midpoint
            ensures every chunk is at least chunk_size // 2 characters long.
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Last chunk — take everything remaining
            if end >= len(text):
                tail = text[start:].strip()
                if tail:
                    chunks.append(tail)
                break

            midpoint = start + self.chunk_size // 2

            # ── try paragraph break first ──────────────────────────────
            para_break = text.rfind("\n\n", start, end)
            if para_break > midpoint:
                end = para_break

            else:
                # ── fall back to sentence break ────────────────────────
                sent_break = max(
                    text.rfind(". ",  start, end),
                    text.rfind(".\n", start, end),
                )
                if sent_break > midpoint:
                    end = sent_break + 1   # include the period

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Slide forward, keeping `overlap` chars from the end
            start = end - self.overlap

        return chunks


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_SCHEDULE_RE = re.compile(
    r"(schedule of benefits|sum insured|deductible|co-payment|sub-limit)",
    re.IGNORECASE,
)
_LIST_RE = re.compile(
    r"^(\s*[•\-\*◦▪]|\s*\([a-z]\)|\s*[ivxIVX]+\.|\s*\d+\.)\s+",
    re.MULTILINE,
)


def _has_table(text: str) -> bool:
    """True if the chunk likely contains a benefit schedule or table."""
    return bool(_SCHEDULE_RE.search(text))


def _has_list(text: str) -> bool:
    """True if the chunk contains bullet / numbered list items."""
    return bool(_LIST_RE.search(text))


def _make_chunk_id(heading: str, idx: int, text: str) -> str:
    """
    Generate a stable, human-readable chunk ID.

    Format:  {heading_slug}_{idx}_{hash8}
    Example: exclusions_general_0_a3f2c1b4

    The hash is computed from heading + idx + first 40 chars of text,
    so re-running extraction on the same PDF produces identical IDs.
    This makes vector-store upserts idempotent.
    """
    digest = hashlib.md5(
        f"{heading}{idx}{text[:40]}".encode()
    ).hexdigest()[:8]

    slug = re.sub(r"[^a-z0-9]+", "_", heading[:30].lower()).strip("_")
    return f"{slug}_{idx}_{digest}"
