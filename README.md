# Insurance Policy Wording Extractor

PyMuPDF-based pipeline that extracts structured, RAG-ready chunks from insurance policy PDFs.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Notebook Demo](#notebook-demo)
- [Configuration & Tuning](#configuration--tuning)
- [Testing](#testing)
- [License](#license)

## Overview

This project automates the extraction and structuring of insurance policy documents into RAG-ready chunks. It processes PDFs through three phases:

1. **Extraction** — Extract raw text blocks from PDF
2. **Reconstruction** — Group blocks into logical sections
3. **Chunking** — Create overlapping chunks optimized for retrieval

## Project Structure

```
insurance_rag/
├── src/
│   └── insurance_rag/
│       ├── __init__.py                # Public API surface
│       ├── models.py                  # TextBlock, PolicyChunk dataclasses
│       ├── cleaner.py                 # Text cleaning (de-hyphenation, unicode, etc.)
│       ├── extractor.py               # Phase 1 — raw PDF block extraction
│       ├── reconstructor.py           # Phase 2 — section reconstruction
│       ├── chunker.py                 # Phase 3 — RAG chunking with overlap
│       └── pipeline.py                # Orchestrator — wires all phases together
├── notebooks/
│   └── 01_extraction_walkthrough.ipynb # Colab-ready demo
├── data/
│   ├── raw/                           # Place your PDF files here
│   └── chunks/                        # Extracted JSON output directory
├── tests/
│   └── test_extractor.py              # Unit tests
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package configuration
└── README.md                          # This file
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/falcon978/Insurance-RAG.git
   cd Insurance-RAG
   ```

2. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

3. **Verify installation:**
   ```bash
   python -c "from insurance_rag.pipeline import ExtractionPipeline; print('Installation successful!')"
   ```

## Quick Start

### Basic Usage

Extract chunks from a single PDF file:

```python
from insurance_rag.pipeline import ExtractionPipeline

# Initialize and run the pipeline
pipeline = ExtractionPipeline("data/raw/policy.pdf")
result = pipeline.run()

# Save extracted chunks
result.save("data/chunks/policy_chunks.json")

# View statistics
print(result.stats)
# Output: {'total_blocks': 892, 'total_sections': 34, 'total_chunks': 178, ...}
```

### Output Format

The extracted chunks are saved as JSON with the following structure:

```json
{
  "chunks": [
    {
      "id": "chunk_001",
      "text": "...",
      "section": "...",
      "page_range": [1, 2],
      "overlap_with_next": true
    }
  ],
  "metadata": {
    "source_file": "policy.pdf",
    "extraction_date": "2026-05-11",
    "total_chunks": 178
  }
}
```

## Usage Guide

### Processing Multiple PDFs

```python
from insurance_rag.pipeline import ExtractionPipeline
import os
from pathlib import Path

pdf_dir = Path("data/raw")
output_dir = Path("data/chunks")
output_dir.mkdir(exist_ok=True)

for pdf_file in pdf_dir.glob("*.pdf"):
    print(f"Processing {pdf_file.name}...")
    pipeline = ExtractionPipeline(str(pdf_file))
    result = pipeline.run()
    
    output_file = output_dir / f"{pdf_file.stem}_chunks.json"
    result.save(str(output_file))
    print(f"Saved: {output_file}")
```

### Inspecting Chunks

```python
import json

with open("data/chunks/policy_chunks.json") as f:
    data = json.load(f)

# View first 5 chunks
for chunk in data["chunks"][:5]:
    print(f"Section: {chunk['section']}")
    print(f"Text: {chunk['text'][:100]}...\n")
```

## Notebook Demo

### Google Colab Setup

Open `notebooks/01_extraction_walkthrough.ipynb` in [Google Colab](https://colab.research.google.com/). The notebook includes:

- ✅ Upload your own PDF or download the sample HDFC Ergo policy
- ✅ Step-by-step pipeline walkthrough (each phase separately)
- ✅ Chunk quality inspection with visualizations
- ✅ RAG query demo using ChromaDB + sentence-transformers (no API key needed)

**Direct Colab Link:**
```
https://colab.research.google.com/github/falcon978/Insurance-RAG/blob/main/notebooks/01_extraction_walkthrough.ipynb
```

## Configuration & Tuning

### Adjusting Extraction Parameters

The heuristics used for PDF extraction are configurable in `src/insurance_rag/extractor.py`:

| Constant | Default | Purpose |
|---|---|---|
| `HEADING_FONT_MIN_PT` | 10.5 | Minimum font size to classify text as a heading |
| `FOOTER_ZONE_PT` | 50 | Points from page bottom to define footer zone |
| `HEADER_ZONE_PT` | 60 | Points from page top to define header zone |
| `HEADER_MAX_CHARS` | 80 | Maximum characters for top-zone block to be classified as header |

### Customizing Section Patterns

Section detection patterns are defined in `src/insurance_rag/reconstructor.py` (`_SECTION_PATTERNS`). Modify regex patterns to match your insurer's section naming conventions:

```python
_SECTION_PATTERNS = {
    "DEFINITIONS": r"definitions?",
    "COVERAGE": r"coverage|benefits?",
    "EXCLUSIONS": r"exclusions?",
    # Add more patterns for your document structure
}
```

### Example: Tuning for Different Insurers

1. Extract a sample PDF with default settings
2. Review the output and identify misclassified sections
3. Adjust `HEADING_FONT_MIN_PT` or section patterns
4. Re-run and compare statistics
5. Iterate until quality meets requirements

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test

```bash
pytest tests/test_extractor.py::test_heading_detection -v
```

### Test Coverage

```bash
pytest tests/ --cov=src/insurance_rag --cov-report=html
```

## Requirements

See `requirements.txt` for all dependencies:

- PyMuPDF (fitz)
- sentence-transformers
- chromadb
- pytest (for testing)

Install with:
```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Submit a Pull Request

## License

This project is licensed under the MIT License — see the LICENSE file for details.

## Support

For issues, questions, or suggestions:
- Open an [issue](https://github.com/falcon978/Insurance-RAG/issues)
- Check existing documentation in the notebooks folder

---

**Last Updated:** 2026-05-11
