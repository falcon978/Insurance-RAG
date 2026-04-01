"""
pipeline.py
-----------
Orchestrator: wires PDFExtractor → SectionReconstructor → Chunker together.

This is the only file you need to import in your notebook or application.
The individual modules (extractor, reconstructor, chunker) remain independently
testable and importable.

Usage
-----
from insurance_rag.pipeline import ExtractionPipeline

result = ExtractionPipeline("policy.pdf").run()
# result.chunks  — list[PolicyChunk]
# result.stats   — extraction statistics
# result.doc_meta, result.toc
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .models import PolicyChunk
from .extractor import PDFExtractor
from .reconstructor import SectionReconstructor
from .chunker import Chunker


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
    Full extraction pipeline for a single insurance policy PDF.

    Parameters
    ----------
    pdf_path   : path to the PDF file
    chunk_size : target characters per RAG chunk   (default 800)
    overlap    : overlap characters between chunks  (default 100)
    verbose    : print progress to stdout           (default True)
    """

    def __init__(
        self,
        pdf_path   : str,
        chunk_size : int  = 800,
        overlap    : int  = 100,
        verbose    : bool = True,
    ):
        self.pdf_path   = Path(pdf_path)
        self.chunk_size = chunk_size
        self.overlap    = overlap
        self.verbose    = verbose

    def run(self) -> ExtractionResult:
        """Execute all three phases and return an ExtractionResult."""
        t0 = time.time()

        # ── Phase 1: Extract ──────────────────────────────────────────
        self._log("Phase 1/3 — Extracting raw blocks …")
        with PDFExtractor(str(self.pdf_path)) as ext:
            blocks   = ext.extract_blocks()
            doc_meta = ext.get_document_metadata()
            toc      = ext.get_toc()

        self._log(
            f"           {doc_meta['page_count']} pages | "
            f"{len(blocks)} blocks | "
            f"{len(toc)} TOC entries"
        )

        # ── Phase 2: Reconstruct sections ─────────────────────────────
        self._log("Phase 2/3 — Reconstructing sections …")
        reconstructor = SectionReconstructor()
        sections      = reconstructor.reconstruct(blocks)
        self._log(f"           {len(sections)} sections found")

        # ── Phase 3: Chunk ────────────────────────────────────────────
        self._log("Phase 3/3 — Chunking for RAG …")
        chunker = Chunker(
            chunk_size  = self.chunk_size,
            overlap     = self.overlap,
            source_file = self.pdf_path.name,
        )
        chunks = chunker.chunk(sections)

        elapsed  = round(time.time() - t0, 2)
        avg_chars = int(sum(len(c.text) for c in chunks) / max(len(chunks), 1))

        stats = {
            "total_blocks"     : len(blocks),
            "total_sections"   : len(sections),
            "total_chunks"     : len(chunks),
            "avg_chunk_chars"  : avg_chars,
            "avg_token_estimate": avg_chars // 4,
            "elapsed_seconds"  : elapsed,
        }

        self._log(
            f"\n✅  Done in {elapsed}s — "
            f"{len(chunks)} chunks | "
            f"~{avg_chars} chars avg | "
            f"~{avg_chars // 4} tokens avg"
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
