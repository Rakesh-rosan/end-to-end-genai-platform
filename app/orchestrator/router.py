"""Intent router: rules first, LLM only when rules are unsure.

Rules are free, instant and predictable; they handle the bulk of traffic. The
LLM classifier handles the long tail. Measure both with evals/run_eval.py.
"""

import re
from dataclasses import dataclass

from app.llm.gateway import LLMGateway, LLMUnavailable
from app.llm.prompts import ROUTER_SYSTEM

INTENTS = ("order_status", "refund_request", "product_info", "policy_faq", "human_handoff", "chitchat")

# Order matters: the first matching rule wins.
RULES = [
    ("human_handoff", r"\b(human|real person|representative|live agent|speak to (someone|an agent)|talk to (someone|an agent))\b"),
    ("refund_request", r"\b(refund|money back|chargeback)\b"),
    ("order_status", r"\bORD-\d+\b|\bwhere is my\b|\btrack(ing)?\b|\border status\b|\bstatus of my\b|\bmy orders?\b"),
    ("chitchat", r"^\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening))\b[\s!.]*$"),
    ("policy_faq", r"\b(polic(y|ies)|returns?|exchanges?|warranty|shipping|ship|payments?|pay|international|cancel)\b"),
    ("product_info", r"\b(price|cost|how much|in stock|stock|available|availability|shoes?|earbuds|backpack|jacket|products?)\b"),
]
_COMPILED = [(intent, re.compile(pattern, re.I)) for intent, pattern in RULES]


@dataclass
class RouteDecision:
    intent: str
    method: str  # rules | llm | default


class IntentRouter:
    def __init__(self, llm: LLMGateway, model: str) -> None:
        self.llm = llm
        self.model = model

    async def classify(self, query: str) -> RouteDecision:
        for intent, pattern in _COMPILED:
            if pattern.search(query):
                return RouteDecision(intent, "rules")
        if not self.llm.is_mock:
            try:
                response = await self.llm.complete("route", ROUTER_SYSTEM, [{"role": "user", "content": query}], self.model)
                label = response.text.strip().lower()
                if label in INTENTS:
                    return RouteDecision(label, "llm")
            except LLMUnavailable:
                pass
        return RouteDecision("policy_faq", "default")
