"""
models.py
---------
Dataclasses that represent the core units flowing through the pipeline.
"""

from dataclasses import dataclass, field


@dataclass
class PolicyChunk:
    """
    A single RAG-ready chunk with provenance metadata.
    This is the unit that gets embedded and stored in a vector store.

    Attributes
    ----------
    chunk_id        : stable unique ID — deterministic hash, safe to upsert
    source_file     : original PDF filename
    page_start      : first page where this chunk's text appears
    page_end        : last page where this chunk's text appears
    section         : top-level section label (e.g., "SECTION B. BENEFITS")
    sub_section     : clause number/title (e.g., "1.1. Hospitalization")
    heading         : full heading text of the parent section
    text            : the chunk text to embed (with context injected)
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
