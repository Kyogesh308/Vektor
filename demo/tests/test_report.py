"""
Tests for demo/eval/report.py

Regression tests for the disclaimer text specifically — it's easy for
someone editing this file later to accidentally delete the "not evidence"
label while reformatting.
"""

from demo.eval.report import generate_markdown_report


def make_sample_json_report():
    return {
        "timestamp": "20250101_120000",
        "golden_set_evaluation": {
            "methodology": "Hand-authored, evidentiary.",
            "mean_precision_at_k": 0.85,
            "n_queries": 10,
            "results": [
                {"query": "test query", "precision_at_k": 0.8,
                 "retrieved": [{"text_preview": "some retrieved text"}]},
            ],
        },
        "auto_questionnaire_demo": {
            "methodology": "LLM-generated, NOT evidence, supplementary only.",
            "n_queries": 5,
            "results": [
                {"query": "auto question",
                 "retrieved": [{"text_preview": "auto retrieved text"}]},
            ],
        },
    }


class TestGenerateMarkdownReport:

    def test_contains_evidentiary_label_for_golden_set(self):
        md = generate_markdown_report(make_sample_json_report())
        assert "Evidentiary" in md

    def test_contains_not_evidence_label_for_auto_questionnaire(self):
        md = generate_markdown_report(make_sample_json_report())
        assert "NOT Evidence" in md or "not evidence" in md.lower()

    def test_contains_mean_precision_number(self):
        md = generate_markdown_report(make_sample_json_report())
        assert "0.850" in md or "0.85" in md

    def test_golden_section_appears_before_auto_section(self):
        """Ordering matters for how a reviewer reads the report top-down."""
        md = generate_markdown_report(make_sample_json_report())
        golden_pos = md.find("Golden Set Results")
        auto_pos = md.find("Auto-Generated Questionnaire")
        assert golden_pos < auto_pos
        assert golden_pos != -1 and auto_pos != -1

    def test_handles_empty_results_gracefully(self):
        report = make_sample_json_report()
        report["golden_set_evaluation"]["results"] = []
        report["auto_questionnaire_demo"]["results"] = []
        md = generate_markdown_report(report)  # must not raise
        assert isinstance(md, str)
