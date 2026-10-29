"""LLM gateway: the single choke point for every model call.

Owns provider failover, token + cost accounting and metrics. Teams later add
per-tenant budgets, prompt logging, or route through a hosted gateway
(LiteLLM, Portkey, AWS Bedrock) without touching business logic.
"""

import logging

from app.config import Settings
from app.llm.providers import (
    AnthropicProvider,
    LLMProvider,
    LLMResponse,
    MockProvider,
    OpenAIProvider,
    ProviderUnavailable,
)
from app.observability import log_event, metrics

logger = logging.getLogger(__name__)

# USD per 1M tokens (input, output). Keep in config/DB in a real system.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


class LLMUnavailable(Exception):
    pass


class LLMGateway:
    def __init__(self, providers: list[LLMProvider], max_tokens: int = 2048) -> None:
        self.providers = providers
        self.max_tokens = max_tokens

    @property
    def is_mock(self) -> bool:
        return all(p.name == "mock" for p in self.providers)

    async def complete(self, task: str, system: str, messages: list[dict], model: str) -> LLMResponse:
        last_error: Exception | None = None
        for provider in self.providers:
            try:
                response = await provider.complete(task, system, messages, model, self.max_tokens)
            except ProviderUnavailable as exc:
                metrics.inc("llm_errors_total", provider=provider.name, task=task)
                logger.warning("provider %s unavailable: %s", provider.name, exc)
                last_error = exc
                continue
            self._record(task, response)
            return response
        raise LLMUnavailable(str(last_error))

    def _record(self, task: str, r: LLMResponse) -> None:
        in_price, out_price = PRICES.get(r.model, (0.0, 0.0))
        cost = (r.input_tokens * in_price + r.output_tokens * out_price) / 1_000_000
        labels = {"provider": r.provider, "model": r.model, "task": task}
        metrics.inc("llm_requests_total", **labels)
        metrics.inc("llm_input_tokens_total", r.input_tokens, **labels)
        metrics.inc("llm_output_tokens_total", r.output_tokens, **labels)
        metrics.inc("llm_cost_usd_total", cost, **labels)
        if r.refused:
            metrics.inc("llm_refusals_total", **labels)
        log_event(logger, "llm_call", task=task, model=r.model, in_tokens=r.input_tokens,
                  out_tokens=r.output_tokens, cost_usd=round(cost, 6), refused=r.refused)


def build_llm_gateway(settings: Settings) -> LLMGateway:
    if settings.llm_provider == "anthropic":
        providers: list[LLMProvider] = [AnthropicProvider(settings)]
    elif settings.llm_provider == "openai":
        for field in ("llm_model", "llm_router_model"):
            if getattr(settings, field).startswith("claude"):
                raise ValueError(f"LLM_PROVIDER=openai but {field.upper()} is a Claude model; set it to an OpenAI model in .env")
        providers = [OpenAIProvider(settings)]
    else:
        providers = [MockProvider()]
    return LLMGateway(providers, settings.llm_max_tokens)
