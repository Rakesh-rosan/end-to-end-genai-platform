"""Orchestrator: routes each request to the right handler and assembles the answer.

    policy_faq / unknown -> RAG over knowledge docs        (cacheable, public)
    product_info         -> commerce DB product search     (cacheable, public)
    order_status         -> commerce DB + carrier API      (personal, never cached)
    refund_request       -> policy check, HITL above threshold
    human_handoff        -> HITL ticket
    chitchat             -> canned reply (no LLM call = no cost)

This is a *deterministic workflow*: code decides which tool runs. The
alternative is an *agent* where the LLM picks tools in a loop. Start with a
workflow; move individual paths to an agent only when the logic is too open-ended to code.
"""

import json
import logging
import re
from dataclasses import dataclass, field

from app.commerce.db import CommerceDB
from app.config import Settings
from app.gateway.auth import User
from app.llm.gateway import LLMGateway, LLMUnavailable
from app.llm.prompts import ASSISTANT_SYSTEM
from app.observability import Trace, metrics
from app.orchestrator.router import IntentRouter
from app.orchestrator.tools import CommerceTools
from app.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)

ORDER_ID_RE = re.compile(r"\bORD-\d+\b", re.I)


@dataclass
class OrchestratorResult:
    answer: str
    intent: str = ""
    route_method: str = ""
    sources: list[str] = field(default_factory=list)
    context: str | None = None  # retrieved text, used by the output grounding check
    cacheable: bool = False
    cache_tags: set[str] = field(default_factory=set)
    escalated: bool = False
    ticket_id: int | None = None


class Orchestrator:
    def __init__(self, router: IntentRouter, retriever: HybridRetriever, tools: CommerceTools,
                 db: CommerceDB, llm: LLMGateway, settings: Settings) -> None:
        self.router = router
        self.retriever = retriever
        self.tools = tools
        self.db = db
        self.llm = llm
        self.settings = settings

    async def handle(self, query: str, user: User, session_id: str, history: list[dict], trace: Trace) -> OrchestratorResult:
        with trace.span("router") as span:
            decision = await self.router.classify(query)
            span.update(intent=decision.intent, method=decision.method)

        handler = {
            "order_status": self._order_status,
            "refund_request": self._refund,
            "product_info": self._product_info,
            "human_handoff": self._handoff,
            "chitchat": self._chitchat,
        }.get(decision.intent, self._rag)

        try:
            result = await handler(query, user, session_id, history, trace)
        except LLMUnavailable:
            metrics.inc("llm_unavailable_total")
            ticket = self.db.create_ticket("handoff", "open", user.customer_id, session_id, query, "llm_unavailable")
            result = OrchestratorResult(
                "Sorry, I'm having trouble right now. I've passed your question to a support agent "
                f"(ticket #{ticket}).", escalated=True, ticket_id=ticket,
            )
        result.intent, result.route_method = decision.intent, decision.method
        return result

    async def _generate(self, query: str, history: list[dict], tag: str, payload: str, trace: Trace) -> str:
        content = f"<{tag}>\n{payload}\n</{tag}>\n\nShopper question: {query}"
        with trace.span("llm.generate") as span:
            response = await self.llm.complete("answer", ASSISTANT_SYSTEM, history + [{"role": "user", "content": content}],
                                               self.settings.llm_model)
            span.update(model=response.model, in_tokens=response.input_tokens, out_tokens=response.output_tokens)
        if response.refused:
            return "I'm sorry, I can't help with that request."
        return response.text

    # --- handlers ------------------------------------------------------------
    async def _rag(self, query, user, session_id, history, trace) -> OrchestratorResult:
        with trace.span("rag.retrieve") as span:
            hits = self.retriever.retrieve(query)
            span.update(hits=len(hits), top_score=hits[0].score if hits else 0.0)
        if not hits:
            return OrchestratorResult(
                "I couldn't find that in our store policies. Would you like me to connect you with a support agent?"
            )
        context = "\n\n".join(f"[source: {h.chunk.doc_id}]\n{h.chunk.text}" for h in hits)
        answer = await self._generate(query, history, "context", context, trace)
        sources = sorted({h.chunk.doc_id for h in hits})
        return OrchestratorResult(answer, sources=sources, context=context, cacheable=True,
                                  cache_tags={f"doc:{s}" for s in sources})

    async def _product_info(self, query, user, session_id, history, trace) -> OrchestratorResult:
        with trace.span("tool.search_products") as span:
            products = self.tools.search_products(query)
            span.update(results=len(products))
        if not products:
            return await self._rag(query, user, session_id, history, trace)
        facts = [{k: p[k] for k in ("id", "name", "price", "availability", "description")} for p in products]
        answer = await self._generate(query, history, "facts", json.dumps(facts, indent=1), trace)
        return OrchestratorResult(answer, sources=[f"product:{p['id']}" for p in products], cacheable=True,
                                  cache_tags={f"product:{p['id']}" for p in products})

    async def _order_status(self, query, user, session_id, history, trace) -> OrchestratorResult:
        order_ids = sorted({m.upper() for m in ORDER_ID_RE.findall(query)})
        with trace.span("tool.order_status") as span:
            facts = await self.tools.order_status(user.customer_id, order_ids)
            span.update(orders=len(facts))
        if not facts:
            return OrchestratorResult("I couldn't find any orders on your account.")
        answer = await self._generate(query, history, "facts", json.dumps(facts, indent=1), trace)
        return OrchestratorResult(answer, sources=[f"order:{f.get('id', f.get('order_id'))}" for f in facts])

    async def _refund(self, query, user, session_id, history, trace) -> OrchestratorResult:
        order_ids = [m.upper() for m in ORDER_ID_RE.findall(query)]
        if not order_ids:
            return OrchestratorResult("Which order would you like a refund for? Please include the order id, e.g. ORD-1001.")
        order = self.tools.get_owned_order(user.customer_id, order_ids[0])
        if order is None:
            return OrchestratorResult(f"I couldn't find order {order_ids[0]} on your account.")

        # Risky, irreversible actions are where HITL earns its keep: the bot may
        # approve small refunds itself, anything larger goes to a person.
        with trace.span("hitl.refund_policy") as span:
            needs_human = order["total"] > self.settings.refund_hitl_threshold
            span.update(total=order["total"], threshold=self.settings.refund_hitl_threshold, needs_human=needs_human)
        if needs_human:
            ticket = self.db.create_ticket("refund", "open", user.customer_id, session_id, query,
                                           "refund_above_threshold", order["id"])
            return OrchestratorResult(
                f"Refunds over ${self.settings.refund_hitl_threshold:.0f} are reviewed by our support team. "
                f"I've opened ticket #{ticket} for order {order['id']}; you'll hear back within 1 business day.",
                escalated=True, ticket_id=ticket,
            )
        ticket = self.db.create_ticket("refund", "auto_approved", user.customer_id, session_id, query,
                                       "refund_below_threshold", order["id"])
        return OrchestratorResult(
            f"Your refund of ${order['total']:.2f} for order {order['id']} is approved (reference #{ticket}). "
            "It will appear on your original payment method within 5-7 business days.",
            ticket_id=ticket,
        )

    async def _handoff(self, query, user, session_id, history, trace) -> OrchestratorResult:
        ticket = self.db.create_ticket("handoff", "open", user.customer_id, session_id, query, "user_requested")
        return OrchestratorResult(
            f"I've connected you with a support agent (ticket #{ticket}). They usually reply within 1 business day.",
            escalated=True, ticket_id=ticket,
        )

    async def _chitchat(self, query, user, session_id, history, trace) -> OrchestratorResult:
        return OrchestratorResult("Hi! I can help you track orders, answer product questions, and explain our store policies.")
