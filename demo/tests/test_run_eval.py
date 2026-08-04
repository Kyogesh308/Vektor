"""
Tests for demo/eval/run_eval.py

Most important tests here: the two schema-separation tests, which
architecturally enforce (not just document) that golden-set and
auto-questionnaire results can never be conflated.
"""

from unittest.mock import MagicMock

import pytest

from demo.eval.run_eval import run_golden_set_eval, run_auto_questionnaire_eval, save_report


def make_search_result(id, score, text):
    r = MagicMock()
    r.id = id
    r.score = score
    r.metadata = {"text": text}
    return r


class TestRunGoldenSetEval:

    def test_precision_at_k_perfect_match(self, monkeypatch):
        from demo.eval import golden_set
        monkeypatch.setattr(golden_set, "GOLDEN_SET", [
            {"query": "test query", "expected_chunk_ids": ["chunk_0"]},
        ])
        # Reimport-safe: patch the imported reference used inside run_eval
        import demo.eval.run_eval as run_eval_module
        monkeypatch.setattr(run_eval_module, "GOLDEN_SET", golden_set.GOLDEN_SET)

        mock_collection = MagicMock()
        mock_collection.search.return_value = [make_search_result("chunk_0", 0.95, "matched text")]

        mock_embedder = MagicMock()
        mock_embedder.embed_one.return_value = [0.1] * 384

        report = run_golden_set_eval(mock_collection, mock_embedder, k=1)

        assert report["source"] == "golden_set"
        assert report["results"][0]["precision_at_k"] == 1.0

    def test_precision_at_k_zero_match(self, monkeypatch):
        import demo.eval.run_eval as run_eval_module
        monkeypatch.setattr(run_eval_module, "GOLDEN_SET", [
            {"query": "test query", "expected_chunk_ids": ["chunk_99"]},
        ])

        mock_collection = MagicMock()
        mock_collection.search.return_value = [make_search_result("chunk_0", 0.5, "wrong chunk")]

        mock_embedder = MagicMock()
        mock_embedder.embed_one.return_value = [0.1] * 384

        report = run_golden_set_eval(mock_collection, mock_embedder, k=1)
        assert report["results"][0]["precision_at_k"] == 0.0

    def test_methodology_field_present(self, monkeypatch):
        import demo.eval.run_eval as run_eval_module
        monkeypatch.setattr(run_eval_module, "GOLDEN_SET", [
            {"query": "q", "expected_chunk_ids": ["c1"]},
        ])
        mock_collection = MagicMock()
        mock_collection.search.return_value = []
        mock_embedder = MagicMock()
        mock_embedder.embed_one.return_value = [0.1] * 384

        report = run_golden_set_eval(mock_collection, mock_embedder, k=5)
        assert "methodology" in report
        assert "evidentiary" in report["methodology"].lower()


class TestRunAutoQuestionnaireEval:

    def test_auto_report_has_no_precision_field(self):
        """
        Architectural enforcement: auto-questionnaire results must NEVER
        contain a precision_at_k or expected_chunk_ids field anywhere —
        their presence would imply false evidentiary weight.
        """
        mock_collection = MagicMock()
        mock_collection.search.return_value = [make_search_result("chunk_0", 0.8, "some text")]
        mock_embedder = MagicMock()
        mock_embedder.embed_one.return_value = [0.1] * 384

        report = run_auto_questionnaire_eval(
            mock_collection, mock_embedder, questions=["What is this about?"], k=3
        )

        report_str = str(report)
        assert "precision_at_k" not in report_str, (
            "Auto-questionnaire report must never contain precision_at_k — "
            "there is no ground truth to compute it against."
        )
        assert "expected_chunk_ids" not in report_str

    def test_auto_report_methodology_labels_as_non_evidentiary(self):
        mock_collection = MagicMock()
        mock_collection.search.return_value = []
        mock_embedder = MagicMock()
        mock_embedder.embed_one.return_value = [0.1] * 384

        report = run_auto_questionnaire_eval(mock_collection, mock_embedder, questions=["q"])
        assert "not" in report["methodology"].lower()
        assert "evidence" in report["methodology"].lower()


class TestSaveReport:

    def test_output_keeps_sources_as_separate_top_level_keys(self, tmp_path, monkeypatch):
        import demo.eval.run_eval as run_eval_module
        monkeypatch.setattr(run_eval_module, "RESULTS_DIR", tmp_path)

        golden = {"source": "golden_set", "results": []}
        auto = {"source": "auto_generated_questionnaire", "results": []}

        path = save_report(golden, auto)

        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "golden_set_evaluation" in data
        assert "auto_questionnaire_demo" in data
        assert data["golden_set_evaluation"]["source"] == "golden_set"
        assert data["auto_questionnaire_demo"]["source"] == "auto_generated_questionnaire"
