"""Regex PII detection + redaction.

Regexes are fast and explainable but miss names/addresses. Production systems
layer an NER model on top (Microsoft Presidio, AWS Comprehend, Bedrock Guardrails).
"""

import re

_PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){12,18}\d\b")),
    ("PHONE", re.compile(r"(?<![\w-])(?:\+?\d{1,2}[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b")),
]


def _luhn_ok(candidate: str) -> bool:
    """Card numbers carry a checksum; using it avoids redacting random long numbers."""
    digits = [int(c) for c in candidate if c.isdigit()][::-1]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact_pii(text: str, allowlist: tuple[str, ...] | list[str] = ()) -> tuple[str, list[str]]:
    """Return (redacted_text, sorted list of PII types found)."""
    found: list[str] = []

    def replacer(label: str):
        def sub(match: re.Match) -> str:
            value = match.group()
            if value in allowlist or (label == "CARD" and not _luhn_ok(value)):
                return value
            found.append(label)
            return f"[{label}]"

        return sub

    for label, pattern in _PATTERNS:
        text = pattern.sub(replacer(label), text)
    return text, sorted(set(found))
