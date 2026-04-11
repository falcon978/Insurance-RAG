"""
pipeline.py
-----------
Orchestrator: wires PDFExtractor → SectionReconstructor → HierarchicalChunker → PolicyVectorStore.

This is the only file you need to import in your notebook or application.
The individual modules remain independently testable and importable.

Usage
-----
from insurance_rag.pipeline import ExtractionPipeline

result = ExtractionPipeline("policy.pdf").run()
# result.chunks  — list[PolicyChunk]
# result.stats   — extraction and indexing statistics
# result.doc_meta, result.toc
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .models import PolicyChunk
from .extractor import PDFExtractor
from .reconstructor import SectionReconstructor
from .chunker import HierarchicalChunker
from .indexer import PolicyVectorStore


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """Everything produced by one pipeline run."""

    doc_meta  : dict
    toc       : list[dict]
    stats     : dict
    chunks    : list[PolicyChunk] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "document": self.doc_meta,
            "toc"     : self.toc,
            "stats"   : self.stats,
            "chunks"  : [asdict(c) for c in self.chunks],
        }

    def save(self, path: str) -> Path:
        """Serialise to JSON and return the output path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ExtractionPipeline:
    """
    Full extraction and indexing pipeline for a single insurance policy PDF.

    Parameters
    ----------
    pdf_path        : path to the PDF file
    chunk_size      : target characters per RAG chunk   (default 1200)
    chunk_overlap   : overlap characters between chunks (default 150)
    persist_dir     : folder to save the vector DB      (default "./chroma_data")
    collection_name : name of the ChromaDB collection   (default "insurance_policies")
    verbose         : print progress to stdout          (default True)
    """

    def __init__(
        self,
        pdf_path        : str,
        chunk_size      : int  = 1200,
        chunk_overlap   : int  = 150,
        persist_dir     : str  = "./chroma_data",
        collection_name : str  = "insurance_policies",
        verbose         : bool = True,
        device          : str  = "cpu",
    ):
        self.pdf_path        = Path(pdf_path)
        self.chunk_size      = chunk_size
        self.chunk_overlap   = chunk_overlap
        self.persist_dir     = persist_dir
        self.collection_name = collection_name
        self.verbose         = verbose
        self.device          = device

    def run(self) -> ExtractionResult:
        """Execute all four phases and return an ExtractionResult."""
        t0 = time.time()

        # ── Phase 1: Extract ──────────────────────────────────────────
        self._log("Phase 1/4 — Extracting raw blocks …")
        with PDFExtractor(str(self.pdf_path)) as ext:
            blocks   = ext.extract_blocks()
            doc_meta = ext.get_document_metadata()
            toc      = ext.get_toc()

        self._log(
            f"           {doc_meta.get('page_count', 0)} pages | "
            f"{len(blocks)} blocks | "
            f"{len(toc) if toc else 0} TOC entries"
        )

        # ── Phase 2: Reconstruct sections ─────────────────────────────
        self._log("Phase 2/4 — Reconstructing sections …")
        reconstructor = SectionReconstructor()
        sections      = reconstructor.reconstruct(blocks)
        self._log(f"           {len(sections)} sections found")

        # ── Phase 3: Chunk ────────────────────────────────────────────
        self._log("Phase 3/4 — Chunking for RAG (Hierarchical) …")
        chunker = HierarchicalChunker(
            chunk_size    = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            source_file   = self.pdf_path.name,
        )
        chunks = chunker.chunk(sections)
        self._log(f"           {len(chunks)} RAG chunks created")

        # ── Phase 4: Index ────────────────────────────────────────────
        self._log(f"Phase 4/4 — Indexing into ChromaDB ({self.persist_dir}) …")
        store = PolicyVectorStore(
            persist_directory=self.persist_dir,
            collection_name=self.collection_name,
            device=self.device,
        )
        store.index_chunks(chunks)

        elapsed  = round(time.time() - t0, 2)
        avg_chars = int(sum(len(c.text) for c in chunks) / max(len(chunks), 1))

        stats = {
            "total_blocks"      : len(blocks),
            "total_sections"    : len(sections),
            "total_chunks"      : len(chunks),
            "avg_chunk_chars"   : avg_chars,
            "avg_token_estimate": avg_chars // 4,
            "indexed_to_db"     : True,
            "database_path"     : self.persist_dir,
            "elapsed_seconds"   : elapsed,
        }

        self._log(
            f"\n✅  Pipeline Complete in {elapsed}s — "
            f"{len(chunks)} chunks successfully indexed!"
        )

        return ExtractionResult(
            doc_meta = doc_meta,
            toc      = toc,
            stats    = stats,
            chunks   = chunks,
        )

    # ── private ───────────────────────────────────────────────────────────

    def _log(self, msg: str):
        if self.verbose:
            print(msg)
