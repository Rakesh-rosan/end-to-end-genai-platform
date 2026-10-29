"""LLM providers behind one interface, so the rest of the system never imports a vendor SDK."""

import json
import re
from dataclasses import dataclass
from typing import Protocol

from app.config import Settings
from app.llm.embeddings import tokenize


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    refused: bool = False


class ProviderUnavailable(Exception):
    """Transient failure (rate limit, 5xx, network). The gateway may try the next provider."""


class LLMProvider(Protocol):
    name: str

    async def complete(
        self, task: str, system: str, messages: list[dict], model: str, max_tokens: int
    ) -> LLMResponse: ...


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        import anthropic

        self._anthropic = anthropic
        # The SDK already retries 408/409/429/5xx with backoff.
        # api_key=None falls back to the SDK's own lookup (ANTHROPIC_API_KEY env var, `ant auth login`).
        key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        self._client = anthropic.AsyncAnthropic(api_key=key, max_retries=2, timeout=60.0)
        self._effort = settings.llm_effort

    async def complete(self, task, system, messages, model, max_tokens) -> LLMResponse:
        anthropic = self._anthropic
        try:
            # Server-side fallbacks: if the model declines on safety grounds, the API
            # re-runs the request on a fallback model inside the same call.
            response = await self._client.beta.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                output_config={"effort": self._effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.BadRequestError:
            raise  # our bug (bad params) - retrying elsewhere will not help
        except (anthropic.RateLimitError, anthropic.APIConnectionError) as exc:
            raise ProviderUnavailable(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise ProviderUnavailable(str(exc)) from exc
            raise

        if response.stop_reason == "refusal":
            return LLMResponse("", response.model, self.name, refused=True)
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(
            text=text.strip(),
            model=response.model,
            provider=self.name,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


class OpenAIProvider:
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        import openai

        self._openai = openai
        # Retries 408/409/429/5xx with backoff. api_key=None falls back to the OPENAI_API_KEY env var.
        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        self._client = openai.AsyncOpenAI(api_key=key, max_retries=2, timeout=60.0)

    async def complete(self, task, system, messages, model, max_tokens) -> LLMResponse:
        openai = self._openai
        try:
            response = await self._client.chat.completions.create(
                model=model,
                max_completion_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, *messages],
            )
        except openai.BadRequestError:
            raise
        except (openai.RateLimitError, openai.APIConnectionError) as exc:
            raise ProviderUnavailable(str(exc)) from exc
        except openai.APIStatusError as exc:
            if exc.status_code >= 500:
                raise ProviderUnavailable(str(exc)) from exc
            raise

        choice = response.choices[0]
        usage = response.usage
        refused = choice.finish_reason == "content_filter" or bool(getattr(choice.message, "refusal", None))
        return LLMResponse(
            text="" if refused else (choice.message.content or "").strip(),
            model=response.model,
            provider=self.name,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            refused=refused,
        )


class MockProvider:
    """Deterministic offline stand-in. It answers by *extracting* from the prompt,
    so the whole pipeline (RAG, guardrails, cache, evals) is testable for free."""

    name = "mock"

    async def complete(self, task, system, messages, model, max_tokens) -> LLMResponse:
        prompt = messages[-1]["content"]
        if task == "route":
            text = "policy_faq"
        elif "<context>" in prompt:
            text = self._answer_from_context(prompt)
        elif "<facts>" in prompt:
            text = self._answer_from_facts(prompt)
        else:
            text = "I can help with orders, products, and store policies."
        return LLMResponse(text, "mock", self.name, len(prompt) // 4, len(text) // 4)

    @staticmethod
    def _between(prompt: str, tag: str) -> str:
        match = re.search(rf"<{tag}>\n?(.*?)\n?</{tag}>", prompt, re.S)
        return match.group(1) if match else ""

    def _answer_from_context(self, prompt: str) -> str:
        context = self._between(prompt, "context")
        question = prompt.rsplit("Shopper question:", 1)[-1]
        q_tokens = set(tokenize(question))
        scored, source = [], None
        for line in context.splitlines():
            if m := re.match(r"\[source: (.+)\]", line):
                source = m.group(1)
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", line.strip(" -")):
                overlap = len(q_tokens & set(tokenize(sentence)))
                if overlap and sentence.endswith((".", "!", "?")):
                    scored.append((overlap, sentence, source))
        if not scored:
            return "I don't know based on our policies. Would you like me to connect you with a support agent?"
        scored.sort(key=lambda item: -item[0])
        top = scored[:2]
        return " ".join(s for _, s, _ in top) + f" [source: {top[0][2]}]"

    def _answer_from_facts(self, prompt: str) -> str:
        facts = json.loads(self._between(prompt, "facts"))
        lines = []
        for item in facts if isinstance(facts, list) else [facts]:
            parts = []
            for key, value in item.items():
                if isinstance(value, dict):
                    parts += [f"{key} {k}: {v}" for k, v in value.items()]
                elif not isinstance(value, list):
                    parts.append(f"{key.replace('_', ' ')}: {value}")
            lines.append("; ".join(parts))
        return "Here's what I found:\n- " + "\n- ".join(lines)
