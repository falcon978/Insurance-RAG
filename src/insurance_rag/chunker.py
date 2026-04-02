"""
chunker.py
----------
Phase 3: Hierarchical Document-Aware Chunking.

Strategy:
  1. Hard Boundaries: Never allow text from two different sections to share a chunk.
  2. Context Injection: Prepend the Major Section and Clause to every single chunk 
     so the vector database always knows the exact legal context.
  3. Semantic Splitting: Use LangChain's Recursive split for oversized sections.
"""

import hashlib
import re
from typing import List

from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from .models import PolicyChunk
from .cleaner import clean_section_text


class HierarchicalChunker:
    """
    Splits reconstructed sections into RAG-ready PolicyChunks.
    Guarantees contextual lineage is injected into the text payload.

    Parameters
    ----------
    chunk_size  : Target character count per chunk. 
                  (1200 chars ~ 300 tokens. Safe for BGE-Large-en 512 token limit).
    chunk_overlap: Overlap to prevent mid-sentence cutoff.
    source_file  : PDF filename for provenance.
    """

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
        source_file: str = "",
    ):
        self.source_file = source_file
        
        # We rely on LangChain's battle-tested semantic splitter for the heavy lifting
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )

    def chunk(self, sections: List[dict]) -> List[PolicyChunk]:
        all_chunks: List[PolicyChunk] = []

        for section in sections:
            text = clean_section_text(section.get("text", ""))
            if not text:
                continue

            # 1. Construct the Ancestry Prefix
            # Example: "[SECTION 3 - EXCLUSIONS] [3.1.1 Pre-existing Diseases] "
            major_label = section.get("section", "").strip()
            sub_label = section.get("sub_section", "").strip()
            
            # Format the prefix cleanly
            lineage_parts = []
            if major_label:
                lineage_parts.append(f"[{major_label}]")
            if sub_label:
                lineage_parts.append(f"[{sub_label}]")
                
            lineage_prefix = " ".join(lineage_parts) + "\n" if lineage_parts else ""
            
            # 2. Account for prefix length in the chunking math
            # We don't want the prefix + the chunk to exceed the embedding model's limit
            effective_chunk_size = self._text_splitter._chunk_size - len(lineage_prefix)
            
            # Temporarily adjust the splitter if the prefix is unusually long
            original_size = self._text_splitter._chunk_size
            if effective_chunk_size > 0:
                self._text_splitter._chunk_size = effective_chunk_size

            # 3. Split the body text using LangChain
            sub_chunks = self._text_splitter.split_text(text)
            
            # Reset the splitter size
            self._text_splitter._chunk_size = original_size

            # 4. Build the final RAG payloads
            pages = sorted(section.get("pages", [1]))
            
            for idx, raw_chunk_text in enumerate(sub_chunks):
                # INJECTION: Prepend the lineage to the raw text
                payload_text = lineage_prefix + raw_chunk_text.strip()
                
                chunk_id = _make_chunk_id(section.get("heading", "unknown"), idx, raw_chunk_text)

                # Ensure metadata dict exists and is populated
                metadata = section.get("metadata", {})
                metadata.update({
                    "chunk_index": idx,
                    "total_chunks_in_section": len(sub_chunks),
                    "char_count": len(payload_text),
                    "has_table": _has_table(raw_chunk_text),
                    "has_list": _has_list(raw_chunk_text),
                })

                all_chunks.append(PolicyChunk(
                    chunk_id       = chunk_id,
                    source_file    = self.source_file,
                    page_start     = pages[0],
                    page_end       = pages[-1],
                    section        = major_label,
                    sub_section    = sub_label,
                    heading        = section.get("heading", ""),
                    text           = payload_text,               # The enriched text to embed!
                    token_estimate = len(payload_text) // 4,
                    metadata       = metadata
                ))

        return all_chunks
    
    # ---------------------------------------------------------------------------
    # Private functions
    # ---------------------------------------------------------------------------

    def _merge_siblings(self, chunks: list[PolicyChunk], min_chars: int = 300) -> list[PolicyChunk]:
        """
        Merges small chunks ONLY if they belong to the same Major Section.
        This reduces DB rows without crossing dangerous legal boundaries.
        """
        merged = []
        carry = None
        
        for chunk in chunks:
            if carry is not None:
                combined_len = len(carry.text) + 1 + len(chunk.text)
                
                # Rule: Must share the same Major Section AND fit the limit
                if (len(carry.text) < min_chars and 
                    carry.section == chunk.section and 
                    combined_len <= self._text_splitter._chunk_size):
                    
                    # Combine text
                    chunk.text = carry.text + "\n" + chunk.text
                    
                    # Update metadata
                    chunk.token_estimate = len(chunk.text) // 4
                    chunk.metadata["char_count"] = len(chunk.text)
                    chunk.page_start = min(carry.page_start, chunk.page_start)
                    
                    # Note: We clear the sub_section since this chunk now holds multiple
                    chunk.sub_section = "Multiple Clauses" 
                    
                    carry = None
                else:
                    merged.append(carry)
                    carry = None
            carry = chunk
            
        if carry:
            merged.append(carry)
            
        return merged


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
    """Generate a stable, human-readable chunk ID for idempotent upserts."""
    digest = hashlib.md5(f"{heading}{idx}{text[:40]}".encode()).hexdigest()[:8]
    slug = re.sub(r"[^a-z0-9]+", "_", heading[:30].lower()).strip("_")
    if not slug:
        slug = "chunk"
    return f"{slug}_{idx}_{digest}"