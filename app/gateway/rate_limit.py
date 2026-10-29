"""Per-user rate limiting.

Why two implementations? An in-memory limiter is per-process: behind a load
balancer with N replicas a user effectively gets N x the limit. Redis gives one
shared counter for all replicas. (nginx also rate-limits per IP at the edge -
see deploy/nginx.conf - which protects against unauthenticated floods.)
"""

import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Request

from app.config import Settings
from app.gateway.auth import User, current_user
from app.observability import metrics


class InMemoryRateLimiter:
    """Sliding-window log: exact, but memory grows with the limit."""

    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)

    async def hit(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and hits[0] <= now - self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return False, int(self.window - (now - hits[0])) + 1
        hits.append(now)
        return True, 0


class RedisRateLimiter:
    """Fixed-window counter shared by every replica."""

    def __init__(self, url: str, limit: int, window_seconds: int = 60) -> None:
        import redis.asyncio as redis

        self._redis = redis.from_url(url)
        self.limit = limit
        self.window = window_seconds

    async def hit(self, key: str) -> tuple[bool, int]:
        now = time.time()
        bucket_key = f"rl:{key}:{int(now // self.window)}"
        count = await self._redis.incr(bucket_key)
        if count == 1:
            await self._redis.expire(bucket_key, self.window)
        return count <= self.limit, int(self.window - now % self.window) + 1


def build_rate_limiter(settings: Settings):
    if settings.redis_url:
        return RedisRateLimiter(settings.redis_url, settings.rate_limit_per_minute)
    return InMemoryRateLimiter(settings.rate_limit_per_minute)


async def rate_limited_user(request: Request, user: User = Depends(current_user)) -> User:
    allowed, retry_after = await request.app.state.services.rate_limiter.hit(user.sub)
    if not allowed:
        metrics.inc("rate_limited_total")
        raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(retry_after)})
    return user
