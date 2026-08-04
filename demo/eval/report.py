"""
demo/eval/report.py
----------------------
Converts a JSON eval run into a readable Markdown report,
with the two sources visually and explicitly separated.
"""

from __future__ import annotations


def generate_markdown_report(json_report: dict) -> str:
    golden = json_report["golden_set_evaluation"]
    auto = json_report["auto_questionnaire_demo"]

    lines = [
        f"# Vektor Semantic Evaluation Report — {json_report['timestamp']}",
        "",
        "## Golden Set Results (Evidentiary)",
        "",
        f"> {golden['methodology']}",
        "",
        f"**Mean Precision@k: {golden['mean_precision_at_k']:.3f}** "
        f"across {golden['n_queries']} hand-authored queries",
        "",
        "| Query | Precision@k | Top Match |",
        "|---|---|---|",
    ]
    for r in golden["results"]:
        top = r["retrieved"][0] if r["retrieved"] else {}
        lines.append(
            f"| {r['query'][:50]} | {r['precision_at_k']:.2f} | "
            f"{top.get('text_preview', '')[:50]}... |"
        )

    lines += [
        "",
        "---",
        "",
        "## Auto-Generated Questionnaire (Demo Only — NOT Evidence)",
        "",
        f"> {auto['methodology']}",
        "",
        f"{auto['n_queries']} questions run for live demonstration purposes. "
        f"No ground truth exists for these — shown for coverage/breadth only.",
        "",
    ]
    for r in auto["results"]:
        lines.append(f"- **Q:** {r['query']}")
        if r["retrieved"]:
            lines.append(f"  - Top result: {r['retrieved'][0]['text_preview'][:80]}...")

    return "\n".join(lines)
