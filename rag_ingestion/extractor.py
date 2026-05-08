"""
extractor.py
------------
Phase 1: PDF to Markdown Extraction.
Uses PyMuPDF4LLM's advanced layout engine to extract structurally perfect Markdown.
"""

import pymupdf
import pymupdf4llm
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Extracts Markdown and document metadata from a PDF.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)

    def extract_markdown_with_pages(self) -> str:
        """
        Extracts Markdown while injecting hidden page markers.
        """
        logger.info(f"Extracting markdown from {self.pdf_path.name}")

        # page_chunks=True returns a list of dictionaries (one per page)
        page_data = pymupdf4llm.to_markdown(self.pdf_path, page_chunks=True)

        full_md = ""
        for page in page_data:
            # PyMuPDF uses 1-based indexing
            display_page_num = page.get("metadata").get("page_number")

            text = page.get("text", "")

            # Inject a custom marker at the top of every page's text for later retrieval
            marker = f"\n\n[__RAG_PIPELINE_PAGE_{display_page_num}__]\n\n"
            full_md += marker + text

        return full_md

    def get_document_metadata(self) -> dict:
        """Standard PyMuPDF metadata extraction."""
        with pymupdf.open(str(self.pdf_path)) as doc:
            meta = doc.metadata or {}
            return {
                "title": meta.get("title", ""),
                "author": meta.get("author", ""),
                "subject": meta.get("subject", ""),
                "creator": meta.get("creator", ""),
                "page_count": len(doc),
                "source_file": self.pdf_path.name,
            }

    def get_toc(self) -> list[dict]:
        """Standard PyMuPDF TOC extraction."""
        with pymupdf.open(str(self.pdf_path)) as doc:
            return [
                {"level": entry[0], "title": entry[1], "page": entry[2]}
                for entry in doc.get_toc()
            ]
