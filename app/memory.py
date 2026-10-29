"""Short-term conversation memory (last N turns per user+session).

Keyed by user AND session so one shopper can never read another's history.
In-memory here; use Redis with a TTL when running multiple replicas.
"""

from collections import defaultdict, deque


class ConversationMemory:
    def __init__(self, max_turns: int = 6) -> None:
        self._store: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=max_turns * 2))

    def history(self, user_id: str, session_id: str) -> list[dict]:
        return list(self._store[(user_id, session_id)])

    def append_turn(self, user_id: str, session_id: str, question: str, answer: str) -> None:
        turns = self._store[(user_id, session_id)]
        turns.append({"role": "user", "content": question})
        turns.append({"role": "assistant", "content": answer})
