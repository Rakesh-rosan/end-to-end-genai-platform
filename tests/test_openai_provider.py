"""Exercises OpenAIProvider against a fake HTTP server - no key or network needed."""

import asyncio
import json

import httpx
import openai
import pytest

from app.config import Settings
from app.llm.gateway import build_llm_gateway
from app.llm.providers import OpenAIProvider, ProviderUnavailable


def make_provider(handler) -> OpenAIProvider:
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._openai = openai
    provider._client = openai.AsyncOpenAI(
        api_key="test", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    return provider


def completion(content: str, finish_reason: str = "stop") -> dict:
    return {
        "id": "c1", "object": "chat.completion", "created": 0, "model": "gpt-test",
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
    }


def test_openai_provider_sends_system_prompt_and_parses_usage():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=completion("  30 days.  "))

    response = asyncio.run(make_provider(handler).complete(
        "answer", "SYS", [{"role": "user", "content": "hi"}], "gpt-test", 100))
    assert seen["messages"][0] == {"role": "system", "content": "SYS"}
    assert seen["max_completion_tokens"] == 100
    assert (response.text, response.input_tokens, response.output_tokens) == ("30 days.", 12, 5)


def test_openai_provider_content_filter_is_refusal():
    provider = make_provider(lambda r: httpx.Response(200, json=completion("", "content_filter")))
    assert asyncio.run(provider.complete("answer", "S", [{"role": "user", "content": "x"}], "m", 10)).refused


def test_openai_provider_maps_429_to_unavailable():
    provider = make_provider(lambda r: httpx.Response(429, json={"error": {"message": "slow down"}}))
    with pytest.raises(ProviderUnavailable):
        asyncio.run(provider.complete("answer", "S", [{"role": "user", "content": "x"}], "m", 10))


def test_openai_requires_openai_model_names():
    with pytest.raises(ValueError, match="LLM_MODEL"):
        build_llm_gateway(Settings(llm_provider="openai", llm_model="claude-opus-5-5"))
