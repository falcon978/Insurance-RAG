# Insurance Policy Wording Extractor

PyMuPDF-based pipeline that extracts structured, RAG-ready chunks from insurance policy PDFs.

## Project Structure

```
insurance_rag/
├── src/
│   └── insurance_rag/
│       ├── __init__.py          # public API surface
│       ├── models.py            # TextBlock, PolicyChunk dataclasses
│       ├── cleaner.py           # text cleaning (de-hyphenation, unicode, etc.)
│       ├── extractor.py         # Phase 1 — raw PDF block extraction
│       ├── reconstructor.py     # Phase 2 — section reconstruction
│       ├── chunker.py           # Phase 3 — RAG chunking with overlap
│       └── pipeline.py          # orchestrator — wires all phases together
├── notebooks/
│   └── 01_extraction_walkthrough.ipynb   # Colab-ready demo
├── data/
│   ├── raw/                     # put your PDF files here
│   └── chunks/                  # extracted JSON output lands here
├── tests/
│   └── test_extractor.py        # unit tests
├── requirements.txt
└── setup.py
```

## Quick Start

```bash
pip install -e .
```

```python
from insurance_rag.pipeline import ExtractionPipeline

result = ExtractionPipeline("data/raw/policy.pdf").run()
result.save("data/chunks/policy_chunks.json")

print(result.stats)
# {'total_blocks': 892, 'total_sections': 34, 'total_chunks': 178, ...}
```

## Notebook (Google Colab)

Open `notebooks/01_extraction_walkthrough.ipynb` in Colab. It covers:
- Upload your own PDF or download the sample HDFC Ergo policy
- Step-by-step pipeline walkthrough (each phase separately)
- Chunk quality inspection with charts
- RAG query demo using ChromaDB + sentence-transformers (no API key needed)

## Tuning for a Different Insurer's PDF

The heuristics that may need adjusting are all in `src/insurance_rag/extractor.py`:

| Constant | Default | What it controls |
|---|---|---|
| `HEADING_FONT_MIN_PT` | 10.5 | Minimum font size to classify as heading |
| `FOOTER_ZONE_PT` | 50 | Points from page bottom → footer zone |
| `HEADER_ZONE_PT` | 60 | Points from page top → running header zone |
| `HEADER_MAX_CHARS` | 80 | Max chars for a top-zone block to be called a header |

Section regex patterns live in `src/insurance_rag/reconstructor.py` (`_SECTION_PATTERNS`).

## Running Tests

```bash
pytest tests/ -v
```
