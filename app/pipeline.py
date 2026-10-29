"""The request pipeline, in the same order as the architecture diagram:

    input guardrails -> semantic cache -> router/orchestrator -> output guardrails -> answer
"""

import re
from dataclasses import dataclass, field

from app.cache.semantic_cache import SemanticCache
from app.gateway.auth import User
from app.guardrails.input_guard import InputGuard
from app.guardrails.output_guard import OutputGuard
from app.memory import ConversationMemory
from app.observability import Trace, metrics
from app.orchestrator.orchestrator import Orchestrator

# Questions that are obviously about the caller's own data skip the cache entirely.
PERSONAL_RE = re.compile(r"\bORD-\d+\b|\bmy (orders?|account|refund|package|payment|delivery)\b", re.I)

BLOCKED_ANSWER = "I can't help with that request. I can help with orders, products, and store policies."
UNSAFE_OUTPUT_ANSWER = ("I'm not confident I can answer that accurately. "
                        "Would you like me to connect you with a support agent?")


@dataclass
class ChatResult:
    answer: str
    intent: str = ""
    sources: list[str] = field(default_factory=list)
    cached: bool = False
    escalated: bool = False
    ticket_id: int | None = None
    blocked: bool = False
    trace: list[dict] = field(default_factory=list)


class ChatPipeline:
    def __init__(self, input_guard: InputGuard, output_guard: OutputGuard, cache: SemanticCache,
                 orchestrator: Orchestrator, memory: ConversationMemory) -> None:
        self.input_guard = input_guard
        self.output_guard = output_guard
        self.cache = cache
        self.orchestrator = orchestrator
        self.memory = memory

    async def run(self, message: str, user: User, session_id: str) -> ChatResult:
        trace = Trace()

        # 1. Input guardrails
        with trace.span("input_guardrails") as span:
            guard = self.input_guard.check(message)
            span.update(allowed=guard.allowed, reasons=guard.reasons, pii=guard.pii)
        if not guard.allowed:
            metrics.inc("guardrail_blocks_total", stage="input", reason=guard.reasons[0])
            return ChatResult(BLOCKED_ANSWER, intent="blocked", blocked=True, trace=trace.spans)
        query = guard.text  # PII-redacted from here on

        # 2. Semantic cache (public questions only)
        personal = bool(PERSONAL_RE.search(query))
        if not personal:
            with trace.span("semantic_cache") as span:
                hit = self.cache.lookup(query)
                span.update(hit=hit is not None, score=round(hit[1], 3) if hit else None)
            metrics.inc("cache_requests_total", result="hit" if hit else "miss")
            if hit:
                entry = hit[0]
                self.memory.append_turn(user.sub, session_id, query, entry.answer)
                return ChatResult(entry.answer, intent=entry.meta["intent"], sources=entry.meta["sources"],
                                  cached=True, trace=trace.spans)

        # 3. Router + orchestrator (RAG / DB / API / HITL / LLM)
        history = self.memory.history(user.sub, session_id)
        result = await self.orchestrator.handle(query, user, session_id, history, trace)

        # 4. Output guardrails
        with trace.span("output_guardrails") as span:
            out = self.output_guard.check(result.answer, context=result.context)
            span.update(allowed=out.allowed, reasons=out.reasons)
        answer = out.text
        if not out.allowed:
            metrics.inc("guardrail_blocks_total", stage="output", reason=out.reasons[-1].split(":")[0])
            answer = UNSAFE_OUTPUT_ANSWER

        # 5. Populate cache only with safe, public, non-escalated answers
        if out.allowed and result.cacheable and not personal and not result.escalated:
            self.cache.store(query, answer, result.cache_tags, {"intent": result.intent, "sources": result.sources})

        self.memory.append_turn(user.sub, session_id, query, answer)
        metrics.inc("chat_requests_total", intent=result.intent)
        return ChatResult(answer, intent=result.intent, sources=result.sources, escalated=result.escalated,
                          ticket_id=result.ticket_id, trace=trace.spans)
