"""
Tests for demo/ingestion/chunker.py

The infinite-loop bug (overlap >= chunk_size) is the most important test
here — everything else is standard boundary coverage.
"""

import pytest

from demo.ingestion.chunker import chunk_text, Chunk


class TestChunkText:

    def test_basic_chunking_produces_multiple_chunks(self):
        text = "a" * 1000
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) > 1

    def test_overlap_equal_to_chunk_size_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            chunk_text("some text", chunk_size=100, overlap=100)

    def test_overlap_greater_than_chunk_size_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            chunk_text("some text", chunk_size=100, overlap=150)

    def test_does_not_hang_with_valid_overlap(self):
        """Regression test for the infinite-loop bug — must complete quickly."""
        import time
        text = "word " * 5000
        t0 = time.perf_counter()
        chunks = chunk_text(text, chunk_size=500, overlap=75)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, "Chunking took too long — possible loop regression"
        assert len(chunks) > 0

    def test_zero_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size"):
            chunk_text("text", chunk_size=0, overlap=0)

    def test_negative_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size"):
            chunk_text("text", chunk_size=-10, overlap=0)

    def test_empty_text_returns_empty_list(self):
        assert chunk_text("", chunk_size=100, overlap=10) == []

    def test_text_shorter_than_chunk_size_returns_one_chunk(self):
        text = "short text"
        chunks = chunk_text(text, chunk_size=500, overlap=75)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_chunk_ids_are_unique_and_sequential(self):
        text = "a" * 2000
        chunks = chunk_text(text, chunk_size=200, overlap=20)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"
        assert ids == [f"chunk_{i}" for i in range(len(chunks))]

    def test_overlap_actually_overlaps_content(self):
        text = "0123456789" * 20  # 200 chars
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        # End of chunk N should share content with start of chunk N+1
        if len(chunks) >= 2:
            end_of_first = chunks[0].text[-10:]
            start_of_second = text[chunks[1].start_char:chunks[1].start_char + 10]
            assert end_of_first in text  # sanity check content is real
            assert chunks[1].start_char < chunks[0].start_char + 50, (
                "Second chunk should start before first chunk ends, "
                "confirming overlap is applied"
            )

    def test_no_overlap_chunks_are_contiguous(self):
        text = "a" * 300
        chunks = chunk_text(text, chunk_size=100, overlap=0)
        for i in range(len(chunks) - 1):
            assert chunks[i + 1].start_char == chunks[i].start_char + 100

    def test_start_char_matches_actual_position_in_source_text(self):
        text = "ABCDEFGHIJ" * 30
        chunks = chunk_text(text, chunk_size=50, overlap=5)
        for chunk in chunks:
            expected = text[chunk.start_char:chunk.start_char + 50].strip()
            assert chunk.text == expected

    def test_whitespace_only_segments_are_skipped(self):
        text = "real content" + " " * 500 + "more content"
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert all(c.text.strip() != "" for c in chunks)
