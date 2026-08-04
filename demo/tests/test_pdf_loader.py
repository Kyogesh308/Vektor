"""
Tests for demo/ingestion/pdf_loader.py

Mocks pypdf.PdfReader — no real PDF file required for unit tests.
"""

from unittest.mock import patch, MagicMock

import pytest

from demo.ingestion.pdf_loader import extract_text


class TestExtractText:

    def test_joins_multiple_pages_with_double_newline(self):
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page one content"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page two content"

        with patch("pypdf.PdfReader") as mock_reader_cls:
            mock_reader_cls.return_value.pages = [mock_page1, mock_page2]
            result = extract_text("fake.pdf")

        assert result == "Page one content\n\nPage two content"

    def test_handles_page_with_no_extractable_text(self):
        """A page.extract_text() returning None must not crash the join."""
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = None
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Real content"

        with patch("pypdf.PdfReader") as mock_reader_cls:
            mock_reader_cls.return_value.pages = [mock_page1, mock_page2]
            result = extract_text("fake.pdf")

        assert "Real content" in result
        # Must not raise TypeError from joining None

    def test_single_page_document(self):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Only page"

        with patch("pypdf.PdfReader") as mock_reader_cls:
            mock_reader_cls.return_value.pages = [mock_page]
            result = extract_text("fake.pdf")

        assert result == "Only page"

    def test_accepts_path_object(self):
        from pathlib import Path
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "content"

        with patch("pypdf.PdfReader") as mock_reader_cls:
            mock_reader_cls.return_value.pages = [mock_page]
            result = extract_text(Path("fake.pdf"))

        assert result == "content"
