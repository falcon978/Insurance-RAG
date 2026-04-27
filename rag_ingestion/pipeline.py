"""
pipeline.py
-----------
Orchestrator: wires PDFExtractor → MarkdownHierarchicalChunker → PolicyVectorStore.
"""

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
import sys

from config import settings # Added centralized settings
from .models import PolicyChunk
from .extractor import PDFExtractor
from .cleaner import clean_markdown_layout
from .chunker import MarkdownHierarchicalChunker
from .indexer import PolicyVectorStore

# 1. Set up a basic console handler so logs actually have a place to print
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# 2. Mute specific noisy libraries (just in case)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# 3. WAKE UP your specific modules! (Assuming they use logger = logging.getLogger(__name__))
logging.getLogger("extractor").setLevel(logging.INFO)
logging.getLogger("chunker").setLevel(logging.INFO)
logging.getLogger("indexer").setLevel(logging.INFO)
logging.getLogger("pipeline").setLevel(logging.INFO)

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


class ExtractionPipeline:
    def __init__(
        self,
        pdf_path: str,
        collection_name: str = "insurance_policies",
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
        device: str = settings.hf_device, # Pulls from config
        verbose: bool = True,
    ):
        self.pdf_path = Path(pdf_path)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = device
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def run(self) -> ExtractionResult:
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"Cannot find {self.pdf_path}")

        t0 = time.time()
        self._log(f"🚀 Starting Extraction Pipeline for: {self.pdf_path.name}\n")

        # ── Phase 1: Markdown Extraction ──────────────────────────────
        self._log("Phase 1/3 — Extracting Markdown via Layout Engine …")
        extractor = PDFExtractor(str(self.pdf_path))
        doc_meta  = extractor.get_document_metadata()
        toc       = extractor.get_toc()
        md_text   = extractor.extract_markdown_with_pages()

        # ── Phase 1.5: Markdown Cleaning ──────────────────────────────
        clean_md = clean_markdown_layout(md_text)

        # ── Phase 2: Chunk & Inject ───────────────────────────────────
        self._log("Phase 2/3 — Semantic Chunking & Context Injection …")
        chunker = MarkdownHierarchicalChunker(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        chunks = chunker.chunk(clean_md, self.pdf_path.name)
        self._log(f"           {len(chunks)} RAG chunks created")

        # ── Phase 3: Index ────────────────────────────────────────────
        # Dynamically log the correct database type
        self._log(f"Phase 3/3 — Indexing into {settings.vector_db_type.upper()} …")
        
        store = PolicyVectorStore(
            collection_name=self.collection_name,
            device=self.device,
        ) # Removed the invalid persist_directory argument!
        
        store.index_chunks(chunks)

        elapsed  = round(time.time() - t0, 2)
        avg_chars = int(sum(len(c.text) for c in chunks) / max(len(chunks), 1))

        stats = {
            "total_chunks"      : len(chunks),
            "avg_chunk_chars"   : avg_chars,
            "avg_token_estimate": avg_chars // 4,
            "indexed_to_db"     : True,
            "database_type"     : settings.vector_db_type, # Reflects Pinecone or Chroma
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