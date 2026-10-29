"""Offline evaluation against a golden set.

    python -m evals.run_eval                          # uses .env (mock by default)
    LLM_PROVIDER=anthropic python -m evals.run_eval   # same questions against Claude

Run this before every prompt, model, threshold or chunking change and compare
the numbers. An LLM system without evals is a system you cannot change safely.
"""

import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

GOLDEN = Path(__file__).with_name("golden.jsonl")
PASSWORDS = {"alice": "alice123", "bob": "bob123", "agent": "agent123"}


def main() -> int:
    cases = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(db_path=Path(tmp) / "eval.db", cdc_enabled=False, rate_limit_per_minute=10_000)
        with TestClient(create_app(settings)) as client:
            tokens = {
                u: client.post("/auth/token", json={"username": u, "password": p}).json()["access_token"]
                for u, p in PASSWORDS.items()
            }
            rows, latencies = [], []
            for i, case in enumerate(cases):
                start = time.perf_counter()
                r = client.post("/v1/chat", json={"message": case["question"], "session_id": f"eval-{i}"},
                                headers={"Authorization": f"Bearer {tokens[case['user']]}"}).json()
                latencies.append((time.perf_counter() - start) * 1000)

                answer = r["answer"].lower()
                checks = {
                    "intent": r["intent"] == case["intent"],
                    "contains": all(s.lower() in answer for s in case.get("must_contain", [])),
                    "not_contains": not any(s.lower() in answer for s in case.get("must_not_contain", [])),
                    "escalation": r["escalated"] == case.get("expect_escalated", False),
                }
                rows.append((case, r, checks))

    passed = sum(all(c.values()) for _, _, c in rows)
    intent_acc = sum(c["intent"] for _, _, c in rows) / len(rows)
    for case, r, checks in rows:
        status = "PASS" if all(checks.values()) else "FAIL " + ",".join(k for k, v in checks.items() if not v)
        print(f"[{status:<20}] ({r['intent']:<14}) {case['question']}")
        if not all(checks.values()):
            print(f"    answer: {r['answer'][:200]!r}")
    print(f"\npassed {passed}/{len(rows)} | intent accuracy {intent_acc:.0%} | "
          f"p50 {statistics.median(latencies):.0f} ms | max {max(latencies):.0f} ms")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
