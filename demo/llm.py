"""
demo/llm.py
-----------
Unified, multi-provider LLM client for answering user queries.
Supports both local inference (Ollama) and cloud LLM APIs:
- Ollama (local server / container)
- OpenAI & OpenAI-compatible APIs (OpenAI, Groq, OpenRouter, DeepSeek, Together, vLLM, LMStudio)
- Anthropic Claude
- Google Gemini
- Auto mode (tries Ollama, automatically falls back to available cloud API keys)

Designed to ensure 100% demo resilience during project evaluation.
"""

from __future__ import annotations

import os
import time
import requests
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class LLMConfig:
    provider: str = "auto"  # auto | ollama | openai | groq | anthropic | gemini
    model: str = "llama3:8b"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 45.0
    temperature: float = 0.2


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    success: bool
    latency_ms: float = 0.0
    error: Optional[str] = None


def _format_context(retrieved_chunks: List[str]) -> str:
    return "\n\n---\n\n".join(
        [f"Passage {i+1}:\n{text}" for i, text in enumerate(retrieved_chunks)]
    )


def _build_system_prompt() -> str:
    return (
        "You are an AI assistant answering questions about a document using only the provided passages.\n"
        "Provide a clear, direct, and concise answer based strictly on the provided passages.\n"
        "If the passages do not contain the answer, state that the document does not explicitly state this information."
    )


def _call_ollama(
    query: str,
    retrieved_chunks: List[str],
    model: str = "llama3:8b",
    base_url: Optional[str] = None,
    timeout: float = 45.0,
) -> LLMResponse:
    t0 = time.perf_counter()
    endpoint_base = (
        base_url
        or os.environ.get("OLLAMA_HOST")
        or "http://localhost:11434"
    ).rstrip("/")
    url = f"{endpoint_base}/api/generate"

    context_str = _format_context(retrieved_chunks)
    prompt = (
        f"{_build_system_prompt()}\n\n"
        f"DOCUMENT PASSAGES:\n{context_str}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER:"
    )

    models_to_try = [model, "llama3:8b", "llama3", "llama3:latest", "llama3.2"]
    unique_models = []
    for m in models_to_try:
        if m and m not in unique_models:
            unique_models.append(m)

    last_error = None
    for m in unique_models:
        try:
            res = requests.post(
                url,
                json={"model": m, "prompt": prompt, "stream": False},
                timeout=timeout,
            )
            if res.status_code == 200:
                answer = res.json().get("response", "").strip()
                if answer:
                    latency = (time.perf_counter() - t0) * 1000
                    return LLMResponse(
                        text=answer,
                        provider="Ollama (Local / Docker)",
                        model=m,
                        success=True,
                        latency_ms=latency,
                    )
            else:
                last_error = f"Ollama HTTP {res.status_code}: {res.text[:200]}"
        except Exception as e:
            last_error = str(e)
            continue

    latency = (time.perf_counter() - t0) * 1000
    return LLMResponse(
        text="",
        provider="Ollama",
        model=model,
        success=False,
        latency_ms=latency,
        error=last_error or f"Could not connect to Ollama at {endpoint_base}",
    )


def _call_openai_compatible(
    query: str,
    retrieved_chunks: List[str],
    api_key: str,
    model: str = "gpt-4o-mini",
    base_url: str = "https://api.openai.com/v1",
    provider_name: str = "OpenAI",
    timeout: float = 45.0,
    temperature: float = 0.2,
) -> LLMResponse:
    t0 = time.perf_counter()
    url = f"{base_url.rstrip('/')}/chat/completions"
    context_str = _format_context(retrieved_chunks)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": _build_system_prompt()},
            {
                "role": "user",
                "content": f"DOCUMENT PASSAGES:\n{context_str}\n\nQUESTION: {query}",
            },
        ],
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=timeout)
        latency = (time.perf_counter() - t0) * 1000
        if res.status_code == 200:
            data = res.json()
            content = data["choices"][0]["message"]["content"].strip()
            return LLMResponse(
                text=content,
                provider=provider_name,
                model=model,
                success=True,
                latency_ms=latency,
            )
        else:
            return LLMResponse(
                text="",
                provider=provider_name,
                model=model,
                success=False,
                latency_ms=latency,
                error=f"{provider_name} API Error {res.status_code}: {res.text[:300]}",
            )
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        return LLMResponse(
            text="",
            provider=provider_name,
            model=model,
            success=False,
            latency_ms=latency,
            error=str(e),
        )


def _call_anthropic(
    query: str,
    retrieved_chunks: List[str],
    api_key: str,
    model: str = "claude-3-5-sonnet-20241022",
    timeout: float = 45.0,
    temperature: float = 0.2,
) -> LLMResponse:
    t0 = time.perf_counter()
    url = "https://api.anthropic.com/v1/messages"
    context_str = _format_context(retrieved_chunks)

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 1024,
        "temperature": temperature,
        "system": _build_system_prompt(),
        "messages": [
            {
                "role": "user",
                "content": f"DOCUMENT PASSAGES:\n{context_str}\n\nQUESTION: {query}",
            }
        ],
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=timeout)
        latency = (time.perf_counter() - t0) * 1000
        if res.status_code == 200:
            data = res.json()
            parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
            content = "".join(parts).strip()
            return LLMResponse(
                text=content,
                provider="Anthropic",
                model=model,
                success=True,
                latency_ms=latency,
            )
        else:
            return LLMResponse(
                text="",
                provider="Anthropic",
                model=model,
                success=False,
                latency_ms=latency,
                error=f"Anthropic API Error {res.status_code}: {res.text[:300]}",
            )
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        return LLMResponse(
            text="",
            provider="Anthropic",
            model=model,
            success=False,
            latency_ms=latency,
            error=str(e),
        )


def _call_gemini(
    query: str,
    retrieved_chunks: List[str],
    api_key: str,
    model: str = "gemini-1.5-flash",
    timeout: float = 45.0,
) -> LLMResponse:
    t0 = time.perf_counter()
    clean_model = model if model.startswith("models/") else f"models/{model}"
    url = f"https://generativelanguage.googleapis.com/v1beta/{clean_model}:generateContent?key={api_key}"
    context_str = _format_context(retrieved_chunks)

    prompt = (
        f"{_build_system_prompt()}\n\n"
        f"DOCUMENT PASSAGES:\n{context_str}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER:"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    try:
        res = requests.post(url, json=payload, timeout=timeout)
        latency = (time.perf_counter() - t0) * 1000
        if res.status_code == 200:
            data = res.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                content = "".join([p.get("text", "") for p in parts]).strip()
                return LLMResponse(
                    text=content,
                    provider="Google Gemini",
                    model=model,
                    success=True,
                    latency_ms=latency,
                )
            return LLMResponse(
                text="",
                provider="Google Gemini",
                model=model,
                success=False,
                latency_ms=latency,
                error="No text candidate returned by Gemini API.",
            )
        else:
            return LLMResponse(
                text="",
                provider="Google Gemini",
                model=model,
                success=False,
                latency_ms=latency,
                error=f"Gemini API Error {res.status_code}: {res.text[:300]}",
            )
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        return LLMResponse(
            text="",
            provider="Google Gemini",
            model=model,
            success=False,
            latency_ms=latency,
            error=str(e),
        )


def generate_llm_answer(
    query: str,
    retrieved_chunks: List[str],
    config: Optional[LLMConfig] = None,
) -> LLMResponse:
    """
    Primary entrypoint for document Q&A. Supports Ollama and Cloud LLM APIs.
    """
    if not retrieved_chunks:
        return LLMResponse(
            text="No relevant passages found in document.",
            provider="System",
            model="None",
            success=True,
        )

    if config is None:
        config = LLMConfig()

    provider = (config.provider or "auto").lower()

    # 1. Direct provider selection
    if provider == "ollama":
        return _call_ollama(
            query,
            retrieved_chunks,
            model=config.model or "llama3:8b",
            base_url=config.base_url,
            timeout=config.timeout,
        )

    if provider == "openai":
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return LLMResponse(
                text="",
                provider="OpenAI",
                model=config.model or "gpt-4o-mini",
                success=False,
                error="Missing OpenAI API Key. Please provide one in the sidebar or set OPENAI_API_KEY.",
            )
        base_url = config.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return _call_openai_compatible(
            query,
            retrieved_chunks,
            api_key=api_key,
            model=config.model or "gpt-4o-mini",
            base_url=base_url,
            provider_name="OpenAI",
            timeout=config.timeout,
            temperature=config.temperature,
        )

    if provider == "groq":
        api_key = config.api_key or os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return LLMResponse(
                text="",
                provider="Groq",
                model=config.model or "llama-3.3-70b-versatile",
                success=False,
                error="Missing Groq API Key. Please provide one in the sidebar or set GROQ_API_KEY.",
            )
        base_url = config.base_url or "https://api.groq.com/openai/v1"
        return _call_openai_compatible(
            query,
            retrieved_chunks,
            api_key=api_key,
            model=config.model or "llama-3.3-70b-versatile",
            base_url=base_url,
            provider_name="Groq",
            timeout=config.timeout,
            temperature=config.temperature,
        )

    if provider == "anthropic":
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return LLMResponse(
                text="",
                provider="Anthropic",
                model=config.model or "claude-3-5-sonnet-20241022",
                success=False,
                error="Missing Anthropic API Key. Please provide one in the sidebar or set ANTHROPIC_API_KEY.",
            )
        return _call_anthropic(
            query,
            retrieved_chunks,
            api_key=api_key,
            model=config.model or "claude-3-5-sonnet-20241022",
            timeout=config.timeout,
            temperature=config.temperature,
        )

    if provider == "gemini":
        api_key = (
            config.api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY", "")
        )
        if not api_key:
            return LLMResponse(
                text="",
                provider="Google Gemini",
                model=config.model or "gemini-1.5-flash",
                success=False,
                error="Missing Gemini API Key. Please provide one in the sidebar or set GEMINI_API_KEY.",
            )
        return _call_gemini(
            query,
            retrieved_chunks,
            api_key=api_key,
            model=config.model or "gemini-1.5-flash",
            timeout=config.timeout,
        )

    # 2. "auto" fallback mode: Try Ollama first, then any available Cloud API Key!
    attempted_providers = []

    # Try Ollama first
    ollama_res = _call_ollama(
        query,
        retrieved_chunks,
        model=config.model or "llama3:8b",
        base_url=config.base_url,
        timeout=10.0,  # Fast timeout for auto-detection
    )
    if ollama_res.success:
        return ollama_res
    attempted_providers.append(f"Ollama ({ollama_res.error or 'unreachable'})")

    # If user provided an API key in config or env, check which one to try:
    openai_key = config.api_key or os.environ.get("OPENAI_API_KEY", "")
    groq_key = os.environ.get("GROQ_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if groq_key:
        groq_res = _call_openai_compatible(
            query,
            retrieved_chunks,
            api_key=groq_key,
            model="llama-3.3-70b-versatile",
            base_url="https://api.groq.com/openai/v1",
            provider_name="Groq (Cloud Fallback)",
            timeout=config.timeout,
        )
        if groq_res.success:
            return groq_res
        attempted_providers.append(f"Groq ({groq_res.error})")

    if openai_key:
        base_url = config.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        openai_res = _call_openai_compatible(
            query,
            retrieved_chunks,
            api_key=openai_key,
            model="gpt-4o-mini",
            base_url=base_url,
            provider_name="OpenAI (Cloud Fallback)",
            timeout=config.timeout,
        )
        if openai_res.success:
            return openai_res
        attempted_providers.append(f"OpenAI ({openai_res.error})")

    if gemini_key:
        gemini_res = _call_gemini(
            query,
            retrieved_chunks,
            api_key=gemini_key,
            model="gemini-1.5-flash",
            timeout=config.timeout,
        )
        if gemini_res.success:
            return gemini_res
        attempted_providers.append(f"Gemini ({gemini_res.error})")

    if anthropic_key:
        anthropic_res = _call_anthropic(
            query,
            retrieved_chunks,
            api_key=anthropic_key,
            model="claude-3-5-sonnet-20241022",
            timeout=config.timeout,
        )
        if anthropic_res.success:
            return anthropic_res
        attempted_providers.append(f"Anthropic ({anthropic_res.error})")

    # If everything failed:
    ollama_host = (
        config.base_url
        or os.environ.get("OLLAMA_HOST")
        or "http://localhost:11434"
    )
    diagnostics = "\n- " + "\n- ".join(attempted_providers)
    help_message = (
        f"⚠️ **Could not connect to Ollama or Cloud LLM APIs**\n\n"
        f"**Diagnostic Details:**{diagnostics}\n\n"
        "**How to resolve for demonstration:**\n"
        f"1. **Option A (Local Ollama):** Run `ollama serve` and pull a model `ollama pull llama3:8b`, "
        f"or launch with `docker compose --profile llm up`.\n"
        f"2. **Option B (Cloud LLM API):** In the sidebar, select a provider (OpenAI, Groq, Claude, Gemini) "
        f"and paste your API key (or set `OPENAI_API_KEY` / `GROQ_API_KEY` in `.env`)."
    )
    return LLMResponse(
        text=help_message,
        provider="Auto (Fallback)",
        model=config.model,
        success=False,
        error="No reachable LLM backend found.",
    )
