"""Thin LLM wrapper — one place to swap providers.

Everything downstream calls only `call_llm(prompt) -> str`. The provider
(OpenRouter, OpenAI-API-compatible) and model live in config.py / .env, so
switching providers means editing this file alone.
"""

from __future__ import annotations

from agents import config


class LLMNotConfigured(RuntimeError):
    """Raised when call_llm is invoked without an API key configured."""


def call_llm(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> str:
    """Send a single prompt to the configured LLM and return its text.

    Args:
        prompt: User prompt.
        system: Optional system instruction.
        temperature: Sampling temperature.
        max_tokens: Response cap.

    Returns:
        The model's text response.

    Raises:
        LLMNotConfigured: If OPENROUTER_API_KEY is missing.
    """
    if not config.llm_is_configured():
        raise LLMNotConfigured(
            "OPENROUTER_API_KEY is not set. Add it to the project-root .env "
            "so the reasoning agent can call the LLM."
        )

    # Lazy import so the whole pipeline (and Phase 1) runs without `openai`.
    from openai import OpenAI

    client = OpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
    )

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=config.OPENROUTER_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()
