"""Output guardrails: the last check before an answer leaves the system.

1. Leak check   - the model must never echo its instructions.
2. PII redaction - catches anything the model pulled from data it should not repeat.
3. Grounding    - a RAG answer must be supported by retrieved context.
                  Lexical overlap is a cheap proxy; production uses an
                  LLM-as-judge or NLI model for faithfulness.
"""

from app.guardrails.input_guard import GuardResult
from app.guardrails.pii import redact_pii
from app.llm.embeddings import tokenize

LEAK_MARKERS = ("you are shopassist", "begin system prompt", "<system>")


def grounding_score(answer: str, context: str) -> float:
    answer_tokens = set(tokenize(answer))
    if not answer_tokens:
        return 1.0
    return len(answer_tokens & set(tokenize(context))) / len(answer_tokens)


class OutputGuard:
    def __init__(self, grounding_threshold: float = 0.35, pii_allowlist: list[str] | None = None) -> None:
        self.grounding_threshold = grounding_threshold
        self.pii_allowlist = tuple(pii_allowlist or ())

    def check(self, answer: str, context: str | None = None) -> GuardResult:
        lowered = answer.lower()
        if any(marker in lowered for marker in LEAK_MARKERS):
            return GuardResult(False, answer, ["system_prompt_leak"])

        redacted, pii = redact_pii(answer, self.pii_allowlist)
        reasons = ["pii_redacted"] if pii else []

        if context is not None:
            score = grounding_score(redacted, context)
            if score < self.grounding_threshold:
                return GuardResult(False, redacted, reasons + [f"ungrounded:{score:.2f}"], pii)
        return GuardResult(True, redacted, reasons, pii)
