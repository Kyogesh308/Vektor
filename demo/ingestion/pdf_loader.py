"""
demo/ingestion/pdf_loader.py
------------------------------
Extract clean text from a PDF for downstream chunking.
"""

from __future__ import annotations

from pathlib import Path


def extract_text(pdf_path: str | Path) -> str:
    """
    Extract text from a PDF file.

    Args:
        pdf_path: Path to the PDF.

    Returns:
        Extracted text as a single string, pages joined with double newlines.
    """
    import pypdf

    reader = pypdf.PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)
