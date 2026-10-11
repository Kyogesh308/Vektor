"""
demo/tests/test_llm.py
----------------------
Unit tests for the multi-provider LLM module (Ollama and Cloud LLM APIs).
"""

import pytest
from unittest.mock import patch, MagicMock
from demo.llm import (
    LLMConfig,
    LLMResponse,
    generate_llm_answer,
    _call_ollama,
    _call_openai_compatible,
    _call_anthropic,
    _call_gemini,
)


def test_empty_chunks_returns_default():
    resp = generate_llm_answer("What is this?", [])
    assert resp.success is True
    assert "No relevant passages" in resp.text


@patch("requests.post")
def test_ollama_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "This is an answer from Ollama."}
    mock_post.return_value = mock_resp

    cfg = LLMConfig(provider="ollama", model="llama3:8b", base_url="http://test-ollama:11434")
    resp = generate_llm_answer("What is HNSW?", ["HNSW is a graph index."], cfg)

    assert resp.success is True
    assert resp.text == "This is an answer from Ollama."
    assert "Ollama" in resp.provider
    assert mock_post.called
    assert mock_post.call_args[0][0] == "http://test-ollama:11434/api/generate"


@patch("requests.post")
def test_openai_compatible_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "This is an answer from GPT-4o."}}]
    }
    mock_post.return_value = mock_resp

    cfg = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="sk-test-key",
        base_url="https://api.openai.com/v1",
    )
    resp = generate_llm_answer("Explain Cosine Similarity", ["Cosine is dot product divided by norms."], cfg)

    assert resp.success is True
    assert resp.text == "This is an answer from GPT-4o."
    assert resp.provider == "OpenAI"
    assert mock_post.called
    headers = mock_post.call_args[1]["headers"]
    assert headers["Authorization"] == "Bearer sk-test-key"


@patch("requests.post")
def test_groq_missing_key_fails_gracefully(mock_post):
    cfg = LLMConfig(provider="groq", api_key="", model="llama-3.3-70b-versatile")
    resp = generate_llm_answer("Test query", ["Some text"], cfg)

    assert resp.success is False
    assert "Missing Groq API Key" in resp.error
    assert not mock_post.called


@patch("requests.post")
def test_anthropic_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "content": [{"type": "text", "text": "Answer from Claude."}]
    }
    mock_post.return_value = mock_resp

    cfg = LLMConfig(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        api_key="anthropic-key-123",
    )
    resp = generate_llm_answer("What is precision?", ["Precision is true positives over positives."], cfg)

    assert resp.success is True
    assert resp.text == "Answer from Claude."
    assert resp.provider == "Anthropic"
    assert mock_post.call_args[1]["headers"]["x-api-key"] == "anthropic-key-123"


@patch("requests.post")
def test_gemini_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Answer from Gemini."}]}}]
    }
    mock_post.return_value = mock_resp

    cfg = LLMConfig(
        provider="gemini",
        model="gemini-1.5-flash",
        api_key="gemini-key-123",
    )
    resp = generate_llm_answer("What is recall?", ["Recall is true positives over actual positives."], cfg)

    assert resp.success is True
    assert resp.text == "Answer from Gemini."
    assert resp.provider == "Google Gemini"


@patch("requests.post")
def test_auto_mode_falls_back_to_openai_when_ollama_unreachable(mock_post):
    def side_effect(url, **kwargs):
        if "api/generate" in url:
            raise ConnectionError("Ollama connection refused")
        m = MagicMock()
        m.status_code = 200
        m.json.return_value = {
            "choices": [{"message": {"content": "Fallback answer from OpenAI."}}]
        }
        return m

    mock_post.side_effect = side_effect

    cfg = LLMConfig(
        provider="auto",
        api_key="sk-test-openai",
        base_url="http://localhost:11434",
    )
    resp = generate_llm_answer("What is vector search?", ["Passage 1"], cfg)

    assert resp.success is True
    assert resp.text == "Fallback answer from OpenAI."
    assert "Fallback" in resp.provider
