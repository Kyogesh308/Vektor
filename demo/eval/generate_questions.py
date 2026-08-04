"""
demo/eval/generate_questions.py
----------------------------------
LLM-based question generation from the source PDF.

LABELED CLEARLY: this exists for live-demo breadth, not accuracy evidence.
Do not present results from this module as proof of semantic correctness —
see the methodology note attached to every report this produces.
"""

from __future__ import annotations


def generate_questions(text: str, n_questions: int = 10) -> list[str]:
    """
    Generate plausible questions from document text using an LLM.

    Implementation left to your existing LLM access pattern (API call or
    local model) — plug in whichever you already use elsewhere in your
    other projects. Not implemented here since it depends on your
    specific API/key setup.

    Returns:
        List of question strings.
    """
    raise NotImplementedError(
        "Wire up your LLM provider of choice here. Keep this module isolated "
        "from golden_set.py — never let auto-generated questions leak into "
        "the evidentiary precision@k calculation."
    )
