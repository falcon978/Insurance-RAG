"""
chunker.py
----------
Phase 2: Semantic Chunking & Context Injection.

Takes raw Markdown from the PDF layout engine, splits it by structural headers, 
applies standard sizing, and injects the parent-child lineage directly into the text payload.
"""

import hashlib
import re
import logging
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from .models import PolicyChunk
from .cleaner import clean_section_text

logger = logging.getLogger(__name__)

class MarkdownHierarchicalChunker:
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 150):
        # 1. Define the Markdown hierarchy to track
        self.headers_to_split_on = [
            ("#", "major_section"),  # L1 (e.g., PART III - EXCLUSIONS)
            ("##", "clause"),        # L2 (e.g., 3.1 Pre-existing diseases)
            ("###", "sub_clause"),   # L3
        ]
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=True # Strip because we format and inject manually
        )
        
        # 2. Setup the size-based splitter for long sections
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk(self, md_text: str, source_file: str) -> list[PolicyChunk]:
        """
        Splits Markdown text and ensures page number inheritance across sub-chunks.
        Tracks both start and end pages for chunks that span across page breaks.
        Returns fully formed PolicyChunks with injected context.
        """
        # 1. Structural Split (By Headers)
        structural_docs = self.md_splitter.split_text(md_text)
        
        # 2. Sub-split by Character Limit
        final_docs = self.text_splitter.split_documents(structural_docs)

        policy_chunks = []
        current_page = 1  # Stateful tracker

        for i, doc in enumerate(final_docs):
            # Extract headers for context injection
            parent = doc.metadata.get("major_section", "General Conditions")
            clause = doc.metadata.get("clause", "")
            sub_clause = doc.metadata.get("sub_clause", "")
            
            # Combine clauses for the metadata field
            full_clause = " > ".join(filter(None, [clause, sub_clause]))

            # Find all markers in the current chunk
            page_matches = re.findall(r'\[__RAG_PIPELINE_PAGE_(\d+)__\]', doc.page_content)
            
            # page_start is either the first marker found OR the last seen 'current_page'
            page_start = int(page_matches[0]) if page_matches else current_page
            
            # page_end is the last marker found in this chunk OR the same as page_start
            page_end = int(page_matches[-1]) if page_matches else page_start
            
            # Update the stateful tracker for the NEXT chunk
            current_page = page_end
            # ---------------------------

            # Clean markers for the LLM
            clean_payload = re.sub(r'\[__RAG_PIPELINE_PAGE_\d+__\]', '', doc.page_content)
            clean_payload = re.sub(r'\n+', '\n', clean_payload).strip()

            # Context Injection
            context_prefix = f"DOCUMENT SECTION: {parent}\n"
            if full_clause:
                context_prefix += f"CLAUSE: {full_clause}\n"
            injected_payload = f"{context_prefix}---\n{clean_payload}"

            # Assess table/list presence for metadata (simple heuristics)
            has_table = "|" in clean_payload and "-|-" in clean_payload
            has_list = bool(re.search(r"^\s*[-*+]\s", clean_payload, re.MULTILINE))
            char_count = len(injected_payload)

            # Generate stable ID
            digest_str = f"{source_file}_{parent}_{full_clause}_{i}".encode()
            chunk_id = hashlib.md5(digest_str).hexdigest()[:12]

            policy_chunks.append(
                PolicyChunk(
                    chunk_id=chunk_id,
                    source_file=source_file,
                    page_start=page_start,
                    page_end=page_end,
                    section=parent,
                    sub_section=full_clause,
                    heading=full_clause or parent,
                    text=injected_payload,
                    token_estimate=len(injected_payload) // 4,
                    metadata={
                        "has_table": has_table,
                        "has_list": has_list,
                        "chunk_index": i,
                        "char_count" : char_count,
                    }
                )
            )

        return policy_chunks