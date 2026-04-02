"""
models.py
---------
Dataclasses that represent the two core units flowing through the pipeline:
  - TextBlock  : a raw extracted block from a single PDF page
  - PolicyChunk: a cleaned, chunked, RAG-ready unit with full metadata
"""

from dataclasses import dataclass, field


@dataclass
class TextBlock:
    """
    One contiguous rectangle of text on a PDF page, as returned by PyMuPDF.

    Attributes
    ----------
    page_num   : 1-indexed page number
    block_num  : block index within the page (PyMuPDF ordering)
    bbox       : (x0, y0, x1, y1) bounding box in points (1 pt = 1/72 inch)
    text       : raw text content of the block
    block_type : one of "text" | "heading" | "header" | "footer"
    font_size  : largest font size seen across all spans in this block
    is_bold    : True if any span in the block uses a bold font
    bold_ratio : ratio of bold characters to total characters in the block
    """

    page_num: int
    block_num: int
    bbox: tuple
    text: str
    block_type: str
    font_size: float = 0.0
    is_bold: bool = False
    bold_ratio: float = 0.0


@dataclass
class PolicyChunk:
    """
    A single RAG-ready chunk with provenance metadata.

    This is the unit that gets embedded and stored in a vector store.
    Rich metadata enables filtered retrieval (e.g. "only search EXCLUSIONS").

    Attributes
    ----------
    chunk_id        : stable unique ID — deterministic hash, safe to upsert
    source_file     : original PDF filename
    page_start      : first page of the parent section
    page_end        : last page of the parent section
    section         : top-level section label  e.g. "EXCLUSIONS"
    sub_section     : clause number if detected  e.g. "4.2.1"
    heading         : full heading text of the parent section
    text            : the chunk text to embed
    token_estimate  : rough token count (len(text) // 4)
    metadata        : extra flags — has_table, has_list, chunk_index, etc.
    """

    chunk_id: str
    source_file: str
    page_start: int
    page_end: int
    section: str
    sub_section: str
    heading: str
    text: str
    token_estimate: int
    metadata: dict = field(default_factory=dict)
