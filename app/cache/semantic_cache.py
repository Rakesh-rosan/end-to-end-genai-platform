"""Semantic cache: reuse an answer when a *similar enough* question was answered before.

Design points worth knowing as an architect:
- Two tiers: exact match on the normalized text (free), then vector similarity.
- Threshold is a precision/recall dial. Too low = wrong answers served fast.
- Only *public* answers may be cached (policies, product info). "Where is my
  order?" must never be served from cache to a different shopper. The pipeline
  enforces this; the cache only stores what it is given.
- Every entry carries tags (doc:returns_policy.md, product:P-100) so CDC can
  invalidate exactly the answers whose source data changed.
- In-memory here; with several replicas use Redis (RediSearch vectors) or a
  vector DB so all replicas share hits and invalidations.
"""

import re
import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CacheEntry:
    question: str
    answer: str
    vector: np.ndarray
    tags: set[str]
    meta: dict
    created_at: float = field(default_factory=time.time)
    hits: int = 0


class SemanticCache:
    def __init__(self, embedder, threshold: float = 0.92, ttl_seconds: int = 3600, max_entries: int = 5000) -> None:
        self.embedder = embedder
        self.threshold = threshold
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._entries: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))

    def _evict_expired_locked(self) -> None:
        cutoff = time.time() - self.ttl
        for key in [k for k, e in self._entries.items() if e.created_at < cutoff]:
            del self._entries[key]

    def lookup(self, query: str) -> tuple[CacheEntry, float] | None:
        key = self._normalize(query)
        with self._lock:
            self._evict_expired_locked()
            if key in self._entries:
                entry = self._entries[key]
                entry.hits += 1
                return entry, 1.0
            if not self._entries:
                return None
            entries = list(self._entries.values())
            matrix = np.stack([e.vector for e in entries])

        scores = matrix @ self.embedder.embed([query])[0]
        best = int(np.argmax(scores))
        if scores[best] < self.threshold:
            return None
        entries[best].hits += 1
        return entries[best], float(scores[best])

    def store(self, query: str, answer: str, tags: set[str], meta: dict) -> None:
        vector = self.embedder.embed([query])[0]
        with self._lock:
            if len(self._entries) >= self.max_entries:
                oldest = min(self._entries, key=lambda k: self._entries[k].created_at)
                del self._entries[oldest]
            self._entries[self._normalize(query)] = CacheEntry(query, answer, vector, set(tags), meta)

    def invalidate_tags(self, tags: set[str]) -> int:
        with self._lock:
            doomed = [k for k, e in self._entries.items() if e.tags & tags]
            for key in doomed:
                del self._entries[key]
        return len(doomed)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._entries),
                "total_hits": sum(e.hits for e in self._entries.values()),
                "items": [
                    {"question": e.question, "hits": e.hits, "tags": sorted(e.tags)}
                    for e in self._entries.values()
                ],
            }
