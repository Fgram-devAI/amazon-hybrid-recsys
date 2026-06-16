"""LLM client tests — no real network call. Uses a fake adapter / monkeypatched requests."""
from __future__ import annotations

import json

import pytest
import requests

from src.reasoning.llm_client import (
    GroqAdapter,
    LLMConfig,
    MissingApiKeyError,
    call_llm,
)


class _FakeAdapter:
    def __init__(self, response):
        self.calls: list[dict] = []
        self._response = response

    def chat(self, *, system, user, temperature, max_tokens, timeout_seconds):
        self.calls.append(
            {
                "system": system,
                "user": user,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout_seconds": timeout_seconds,
            }
        )
        return self._response


def _valid_response_text():
    return json.dumps(
        {
            "summary": "ok",
            "recommendations": [
                {
                    "parent_asin": "C1",
                    "title": "Game C1",
                    "why": "matches",
                    "evidence_used": ["lightgcn_score"],
                    "confidence": "medium",
                }
            ],
            "caveats": [],
        }
    )


def test_call_llm_dry_run_does_not_invoke_adapter():
    adapter = _FakeAdapter(response="should not be returned")
    result = call_llm(
        adapter=adapter,
        prompt={"system": "S", "user": "U"},
        config=LLMConfig(
            provider="groq",
            model="x",
            api_key="key",
            base_url="https://example/v1",
            temperature=0.2,
            max_tokens=10,
            timeout_seconds=5,
        ),
        dry_run=True,
    )
    assert adapter.calls == []
    assert result["mode"] == "dry_run"
    assert result["text"] is None
    assert result["parsed_json"] is None
    assert result["validation_error"] is None


def test_call_llm_real_call_returns_validated_parsed_json():
    adapter = _FakeAdapter(response=_valid_response_text())
    result = call_llm(
        adapter=adapter,
        prompt={"system": "S", "user": "U"},
        config=LLMConfig(
            provider="groq",
            model="x",
            api_key="key",
            base_url="https://example/v1",
            temperature=0.2,
            max_tokens=10,
            timeout_seconds=5,
        ),
        dry_run=False,
    )
    assert adapter.calls and adapter.calls[0]["system"] == "S"
    assert result["mode"] == "live"
    assert result["parsed_json"]["summary"] == "ok"
    assert result["parsed_json"]["recommendations"][0]["confidence"] == "medium"
    assert result["validation_error"] is None


def test_call_llm_accepts_json_wrapped_in_markdown_fence():
    adapter = _FakeAdapter(response=f"```json\n{_valid_response_text()}\n```")
    result = call_llm(
        adapter=adapter,
        prompt={"system": "S", "user": "U"},
        config=LLMConfig(
            provider="groq",
            model="x",
            api_key="key",
            base_url="https://example/v1",
            temperature=0.2,
            max_tokens=10,
            timeout_seconds=5,
        ),
        dry_run=False,
    )
    assert result["text"].startswith("```json")
    assert result["parsed_json"]["summary"] == "ok"
    assert result["validation_error"] is None


def test_call_llm_handles_invalid_response_text():
    adapter = _FakeAdapter(response="hello, not json")
    result = call_llm(
        adapter=adapter,
        prompt={"system": "S", "user": "U"},
        config=LLMConfig(
            provider="groq",
            model="x",
            api_key="key",
            base_url="https://example/v1",
            temperature=0.2,
            max_tokens=10,
            timeout_seconds=5,
        ),
        dry_run=False,
    )
    assert result["text"] == "hello, not json"
    assert result["parsed_json"] is None
    assert isinstance(result["validation_error"], str)
    assert result["validation_error"]


def test_call_llm_handles_response_with_wrong_shape():
    """Valid JSON but wrong shape (missing required `summary`) -> validation error."""
    bad_shape = json.dumps({"recommendations": [], "caveats": []})
    adapter = _FakeAdapter(response=bad_shape)
    result = call_llm(
        adapter=adapter,
        prompt={"system": "S", "user": "U"},
        config=LLMConfig(
            provider="groq",
            model="x",
            api_key="key",
            base_url="https://example/v1",
            temperature=0.2,
            max_tokens=10,
            timeout_seconds=5,
        ),
        dry_run=False,
    )
    assert result["text"] == bad_shape
    assert result["parsed_json"] is None
    assert isinstance(result["validation_error"], str)
    assert "summary" in result["validation_error"].lower()


def test_call_llm_handles_provider_http_error():
    class _FailingAdapter:
        def chat(self, **kwargs):  # noqa: ARG002
            raise requests.HTTPError("413 Client Error: Payload Too Large")

    result = call_llm(
        adapter=_FailingAdapter(),
        prompt={"system": "S", "user": "U"},
        config=LLMConfig(
            provider="groq",
            model="x",
            api_key="key",
            base_url="https://example/v1",
            temperature=0.2,
            max_tokens=10,
            timeout_seconds=5,
        ),
        dry_run=False,
    )

    assert result["mode"] == "live"
    assert result["text"] is None
    assert result["parsed_json"] is None
    assert "Payload Too Large" in result["validation_error"]


def test_groq_adapter_calls_chat_completions(monkeypatch):
    captured = {}

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {"message": {"content": '{"summary": "ok"}'}}
                ]
            }

        def raise_for_status(self):
            return None

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr("src.reasoning.llm_client.requests.post", _fake_post)

    adapter = GroqAdapter(api_key="abc", model="llama-x", base_url="https://api.groq.com/openai/v1")
    text = adapter.chat(
        system="S", user="U", temperature=0.2, max_tokens=100, timeout_seconds=10
    )

    assert text == '{"summary": "ok"}'
    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer abc"
    assert captured["json"]["model"] == "llama-x"
    msgs = captured["json"]["messages"]
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == "S"
    assert msgs[1]["role"] == "user" and msgs[1]["content"] == "U"
    assert captured["timeout"] == 10


def test_missing_api_key_raises_when_calling_live():
    cfg = LLMConfig(
        provider="groq",
        model="x",
        api_key=None,
        base_url="https://example/v1",
        temperature=0.2,
        max_tokens=10,
        timeout_seconds=5,
    )
    with pytest.raises(MissingApiKeyError):
        call_llm(
            adapter=_FakeAdapter("x"),
            prompt={"system": "", "user": ""},
            config=cfg,
            dry_run=False,
        )


def test_missing_api_key_is_fine_for_dry_run():
    cfg = LLMConfig(
        provider="groq",
        model="x",
        api_key=None,
        base_url="https://example/v1",
        temperature=0.2,
        max_tokens=10,
        timeout_seconds=5,
    )
    result = call_llm(
        adapter=_FakeAdapter("x"), prompt={"system": "", "user": ""}, config=cfg, dry_run=True
    )
    assert result["mode"] == "dry_run"
