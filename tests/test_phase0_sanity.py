"""
Phase 0 sentinel tests.

These two tests have exactly one job: prove the test runner works.
- test_passes_intentionally must always be green.
- test_fails_intentionally must always be red.

Do not delete these. Do not modify them to make both green.
If test_fails_intentionally ever shows as PASSED, your test configuration is broken.
"""

import src.vektor


def test_passes_intentionally():
    """This test must always pass. If it fails, pytest itself is broken."""
    assert 1 + 1 == 2


def test_fails_intentionally():
    """
    This test must always fail. It is not a bug.
    Its purpose is to prove that pytest correctly reports failures.
    GitHub Actions must show this test as FAILED — that is correct behavior.
    """
    assert 1 + 1 == 3, "Sentinel: this failure is intentional and expected."