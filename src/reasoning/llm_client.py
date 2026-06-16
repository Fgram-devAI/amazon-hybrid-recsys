"""LLM provider adapter and dry-run-aware ``call_llm`` orchestration.

Only Groq (OpenAI-compatible /chat/completions) is implemented in v1. The
adapter Protocol is provider-neutral: any future provider with the same shape
plugs in by config alone.

The live-mode response is validated against ``LLMRecommendationResponse`` from
``src/reasoning/schemas.py``. On validation success ``parsed_json`` is the
validated dump; on failure ``parsed_json`` is None and ``validation_error``
is a non-empty string.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

import requests
from pydantic import ValidationError

from src.reasoning.schemas import LLMRecommendationResponse


logger = logging.getLogger("reasoning.llm_client")

_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


class MissingApiKeyError(RuntimeError):
    """Raised when a live LLM call is requested without an API key."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str | None
    base_url: str
    temperature: float
    max_tokens: int
    timeout_seconds: int


class ChatCompletionAdapter(Protocol):
    def chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
    ) -> str: ...


@dataclass
class GroqAdapter:
    api_key: str
    model: str
    base_url: str = "https://api.groq.com/openai/v1"

    def chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
    ) -> str:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        response = requests.post(url, headers=headers, json=body, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        return str(payload["choices"][0]["message"]["content"])


def _try_validate_response(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Validate `text` against LLMRecommendationResponse; return (dump, error)."""
    candidate = _strip_json_fence(text)
    try:
        parsed = LLMRecommendationResponse.model_validate_json(candidate)
    except ValidationError as exc:
        return None, str(exc)
    return parsed.model_dump(mode="json"), None


def _strip_json_fence(text: str) -> str:
    match = _JSON_FENCE_RE.match(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def call_llm(
    *,
    adapter: ChatCompletionAdapter,
    prompt: dict[str, str],
    config: LLMConfig,
    dry_run: bool,
) -> dict[str, Any]:
    """Run an LLM chat call or skip it if ``dry_run`` is true."""
    if dry_run:
        return {
            "mode": "dry_run",
            "text": None,
            "parsed_json": None,
            "validation_error": None,
        }
    if not config.api_key:
        raise MissingApiKeyError(
            f"LLM provider {config.provider!r} requires an API key but none was provided "
            "(set the env variable named in config['llm']['api_key_env'])."
        )
    try:
        text = adapter.chat(
            system=prompt["system"],
            user=prompt["user"],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_seconds=config.timeout_seconds,
        )
    except requests.RequestException as exc:
        return {
            "mode": "live",
            "text": None,
            "parsed_json": None,
            "validation_error": f"LLM provider request failed: {exc}",
        }
    parsed_json, validation_error = _try_validate_response(text)
    return {
        "mode": "live",
        "text": text,
        "parsed_json": parsed_json,
        "validation_error": validation_error,
    }


def build_adapter_from_config(config: LLMConfig) -> ChatCompletionAdapter:
    """Construct the right adapter for ``config.provider``."""
    if config.provider == "groq":
        if not config.api_key:
            raise MissingApiKeyError(
                "Groq provider requires GROQ_API_KEY (or whatever api_key_env points at)."
            )
        return GroqAdapter(api_key=config.api_key, model=config.model, base_url=config.base_url)
    raise ValueError(f"Unsupported LLM provider: {config.provider!r}")
