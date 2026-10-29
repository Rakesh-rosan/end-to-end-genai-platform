"""Input guardrails: run *before* anything is cached, retrieved or sent to an LLM.

Layers (cheap -> expensive): size limits, prompt-injection heuristics, abuse
terms, PII redaction. In production add a classifier model (Llama Guard,
Bedrock Guardrails, Azure Content Safety) as the last layer.
"""

import re
from dataclasses import dataclass, field

from app.guardrails.pii import redact_pii

INJECTION_PATTERNS = [
    re.compile(p, re.I)
    for p in [
        r"ignore (all |any )?(the )?(previous|prior|above) (instructions|rules|prompts?)",
        r"disregard (your|the|all) (instructions|rules|system prompt)",
        r"(reveal|show|print|repeat|leak) (me )?(your|the) (system |hidden )?(prompt|instructions)",
        r"\bsystem prompt\b",
        r"you are now (a|an|in)\b",
        r"\b(developer|dan|jailbreak) mode\b",
        r"act as (an? )?(unfiltered|unrestricted|uncensored)",
    ]
]

ABUSE_PATTERNS = [re.compile(p, re.I) for p in [r"\bkill (yo)?urself\b", r"\bhow to (make|build) a bomb\b"]]


@dataclass
class GuardResult:
    allowed: bool
    text: str
    reasons: list[str] = field(default_factory=list)
    pii: list[str] = field(default_factory=list)


class InputGuard:
    def __init__(self, max_chars: int = 2000) -> None:
        self.max_chars = max_chars

    def check(self, text: str) -> GuardResult:
        text = text.strip()
        if not text:
            return GuardResult(False, text, ["empty"])
        if len(text) > self.max_chars:
            return GuardResult(False, text, ["too_long"])
        if any(p.search(text) for p in INJECTION_PATTERNS):
            return GuardResult(False, text, ["prompt_injection"])
        if any(p.search(text) for p in ABUSE_PATTERNS):
            return GuardResult(False, text, ["abuse"])
        # PII is redacted, not blocked: the shopper may legitimately paste an email.
        # Redacting here means it never reaches the cache, logs or the LLM provider.
        redacted, pii = redact_pii(text)
        return GuardResult(True, redacted, ["pii_redacted"] if pii else [], pii)
