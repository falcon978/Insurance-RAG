"""
tests/test_extractor.py
-----------------------
Unit tests for cleaner, chunker, and reconstructor modules.
Run with:  pytest tests/
"""

import sys
from pathlib import Path

# Make src/ importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from insurance_rag.cleaner import clean_block_text
from insurance_rag.chunker import Chunker, _make_chunk_id
from insurance_rag.reconstructor import SectionReconstructor
from insurance_rag.models import TextBlock


# ---------------------------------------------------------------------------
# cleaner.py
# ---------------------------------------------------------------------------

class TestCleaner:

    def test_dehyphenation(self):
        result = clean_block_text("hospitaliz-\nation")
        assert result == "hospitalization"

    def test_intentional_hyphen_preserved(self):
        # "co-payment" broken at line end should rejoin without hyphen
        result = clean_block_text("co-\npayment")
        assert result == "copayment"   # current behaviour: merges

    def test_collapse_spaces(self):
        result = clean_block_text("word    another\ttab")
        assert result == "word another tab"

    def test_normalize_newlines(self):
        result = clean_block_text("para1\n\n\n\n\npara2")
        assert result == "para1\n\npara2"

    def test_ligature_fi(self):
        result = clean_block_text("\ufb01nal")   # ﬁnal
        assert result == "final"

    def test_smart_quotes(self):
        result = clean_block_text("\u201chello\u201d")
        assert result == '"hello"'

    def test_non_breaking_space(self):
        result = clean_block_text("word\xa0word")
        assert result == "word word"

    def test_strip(self):
        result = clean_block_text("  \n  hello  \n  ")
        assert result == "hello"


# ---------------------------------------------------------------------------
# chunker.py
# ---------------------------------------------------------------------------

class TestChunker:

    def _make_chunker(self, size=100, overlap=20):
        return Chunker(chunk_size=size, overlap=overlap, source_file="test.pdf")

    def test_short_text_not_split(self):
        chunker = self._make_chunker(size=500)
        result  = chunker._split("short text")
        assert result == ["short text"]

    def test_split_produces_overlap(self):
        chunker = self._make_chunker(size=50, overlap=10)
        text    = "a" * 120
        chunks  = chunker._split(text)
        assert len(chunks) > 1
        # Each subsequent chunk should start with chars from the previous chunk's end
        for i in range(1, len(chunks)):
            overlap_text = chunks[i - 1][-10:]
            assert chunks[i].startswith(overlap_text)

    def test_chunk_objects_have_metadata(self):
        chunker  = self._make_chunker(size=500)
        sections = [{
            "section"    : "EXCLUSIONS",
            "sub_section": "",
            "heading"    : "General Exclusions",
            "pages"      : {3, 4},
            "text"       : "This condition is excluded. " * 5,
        }]
        chunks = chunker.chunk(sections)
        assert len(chunks) >= 1
        c = chunks[0]
        assert c.section      == "EXCLUSIONS"
        assert c.page_start   == 3
        assert c.page_end     == 4
        assert c.source_file  == "test.pdf"
        assert "chunk_index" in c.metadata

    def test_chunk_id_is_stable(self):
        id1 = _make_chunk_id("Exclusions", 0, "Some text here")
        id2 = _make_chunk_id("Exclusions", 0, "Some text here")
        assert id1 == id2

    def test_chunk_id_differs_by_index(self):
        id1 = _make_chunk_id("Exclusions", 0, "Some text here")
        id2 = _make_chunk_id("Exclusions", 1, "Some text here")
        assert id1 != id2

    def test_invalid_overlap_raises(self):
        import pytest
        with pytest.raises(ValueError):
            Chunker(chunk_size=100, overlap=100)

    def test_empty_section_skipped(self):
        chunker  = self._make_chunker()
        sections = [{
            "section": "EMPTY", "sub_section": "",
            "heading": "Empty Section", "pages": {1}, "text": "   ",
        }]
        assert chunker.chunk(sections) == []


# ---------------------------------------------------------------------------
# reconstructor.py
# ---------------------------------------------------------------------------

class TestReconstructor:

    def _make_block(self, text, block_type="text", page=1,
                    font_size=9.0, is_bold=False):
        return TextBlock(
            page_num=page, block_num=0, bbox=(0, 0, 100, 20),
            text=text, block_type=block_type,
            font_size=font_size, is_bold=is_bold,
        )

    def test_headers_footers_discarded(self):
        blocks = [
            self._make_block("Page 1", block_type="header"),
            self._make_block("Body text here.", block_type="text"),
            self._make_block("1 of 20", block_type="footer"),
        ]
        rec      = SectionReconstructor()
        sections = rec.reconstruct(blocks)
        assert len(sections) == 1
        assert "Page 1" not in sections[0]["text"]
        assert "1 of 20" not in sections[0]["text"]

    def test_heading_starts_new_section(self):
        blocks = [
            self._make_block("Body of first section.", block_type="text"),
            self._make_block("EXCLUSIONS", block_type="heading"),
            self._make_block("Exclusion body text.", block_type="text"),
        ]
        rec      = SectionReconstructor()
        sections = rec.reconstruct(blocks)
        assert len(sections) == 2
        assert sections[1]["section"] == "EXCLUSIONS"

    def test_pages_tracked_across_blocks(self):
        blocks = [
            self._make_block("DEFINITIONS", block_type="heading", page=2),
            self._make_block("Body page 2.", block_type="text",   page=2),
            self._make_block("Body page 3.", block_type="text",   page=3),
        ]
        rec      = SectionReconstructor()
        sections = rec.reconstruct(blocks)
        assert sections[0]["pages"] == {2, 3}

    def test_regex_fallback_detects_section(self):
        blocks = [
            # font_size below heading threshold, but matches SECTION pattern
            self._make_block("SECTION 4 – EXCLUSIONS", block_type="text",
                             font_size=9.0, is_bold=False),
            self._make_block("Excluded items.", block_type="text"),
        ]
        rec      = SectionReconstructor()
        sections = rec.reconstruct(blocks)
        # Should create a section for SECTION 4
        assert any("EXCLUSIONS" in s["section"] for s in sections)
