"""
Tests for demo/eval/golden_set.py

Validates the hand-authored fixture is well-formed — cheap insurance
against a typo silently breaking the one part of this branch that's
actual evidence.
"""

from demo.eval.golden_set import GOLDEN_SET


class TestGoldenSetStructure:

    def test_golden_set_is_non_empty(self):
        assert len(GOLDEN_SET) > 0, (
            "GOLDEN_SET is empty — this is the evidentiary component, "
            "it must contain hand-authored entries before any review."
        )

    def test_every_entry_has_required_keys(self):
        for item in GOLDEN_SET:
            assert "query" in item
            assert "expected_chunk_ids" in item

    def test_every_query_is_non_empty_string(self):
        for item in GOLDEN_SET:
            assert isinstance(item["query"], str)
            assert len(item["query"].strip()) > 0

    def test_every_entry_has_at_least_one_expected_chunk(self):
        for item in GOLDEN_SET:
            assert len(item["expected_chunk_ids"]) > 0, (
                f"Query '{item['query']}' has no expected_chunk_ids — "
                f"an entry with zero expected answers can't be scored."
            )

    def test_no_duplicate_queries(self):
        queries = [item["query"] for item in GOLDEN_SET]
        assert len(queries) == len(set(queries)), "Duplicate queries found in golden set"

    def test_minimum_size_for_credible_evaluation(self):
        assert len(GOLDEN_SET) >= 10, (
            f"Golden set has only {len(GOLDEN_SET)} entries — fewer than 10 "
            f"is thin evidence for a semantic accuracy claim in a review."
        )
