"""Tools the orchestrator can call: commerce DB reads and an external carrier API.

Security rule worth repeating: authentication happened at the gateway, but
*authorization* happens here, at the data. Every order lookup is checked
against the caller's customer_id, so neither a prompt injection nor a guessed
order id can leak another shopper's data. Never let the LLM decide who owns what.
"""

import asyncio
import hashlib
import time

from app.commerce.db import CommerceDB
from app.llm.embeddings import tokenize


class CircuitBreaker:
    """After `threshold` consecutive failures, stop calling the dependency for `reset_after` seconds."""

    def __init__(self, threshold: int = 3, reset_after: float = 30.0) -> None:
        self.threshold = threshold
        self.reset_after = reset_after
        self.failures = 0
        self.opened_at: float | None = None

    def allow(self) -> bool:
        if self.opened_at is None:
            return True
        if time.monotonic() - self.opened_at >= self.reset_after:
            self.opened_at = None  # half-open: let one call through
            return True
        return False

    def record_success(self) -> None:
        self.failures, self.opened_at = 0, None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = time.monotonic()


class CarrierAPI:
    """Stand-in for a carrier tracking API (UPS/FedEx). Replace `_fetch` with a real httpx call."""

    def __init__(self, timeout: float = 2.0) -> None:
        self.timeout = timeout
        self.breaker = CircuitBreaker()

    async def track(self, carrier: str, tracking_number: str) -> dict:
        if not self.breaker.allow():
            return {"status": "tracking temporarily unavailable"}
        try:
            result = await asyncio.wait_for(self._fetch(carrier, tracking_number), self.timeout)
        except (asyncio.TimeoutError, OSError):
            self.breaker.record_failure()
            return {"status": "tracking temporarily unavailable"}
        self.breaker.record_success()
        return result

    async def _fetch(self, carrier: str, tracking_number: str) -> dict:
        await asyncio.sleep(0.02)  # simulated network latency
        cities = ["Memphis, TN", "Louisville, KY", "Chicago, IL", "Dallas, TX"]
        city = cities[int(hashlib.md5(tracking_number.encode()).hexdigest(), 16) % len(cities)]
        return {"status": "in transit", "last_scan": city, "eta": "2 business days"}


class CommerceTools:
    def __init__(self, db: CommerceDB, carrier: CarrierAPI) -> None:
        self.db = db
        self.carrier = carrier

    def get_owned_order(self, customer_id: str | None, order_id: str) -> dict | None:
        order = self.db.get_order(order_id)
        # Same answer for "doesn't exist" and "not yours" - prevents order-id enumeration.
        if order is None or customer_id is None or order["customer_id"] != customer_id:
            return None
        return order

    async def order_status(self, customer_id: str | None, order_ids: list[str]) -> list[dict]:
        if order_ids:
            orders = [(oid, self.get_owned_order(customer_id, oid)) for oid in order_ids]
        else:
            orders = [(o["id"], o) for o in self.db.list_orders(customer_id)] if customer_id else []

        facts = []
        for order_id, order in orders:
            if order is None:
                facts.append({"order_id": order_id, "error": "order not found for this account"})
                continue
            fact = {k: order[k] for k in ("id", "status", "total", "created_at", "carrier", "tracking_number")}
            fact["items"] = order["items"]
            if order["status"] == "shipped" and order["tracking_number"]:
                fact["tracking"] = await self.carrier.track(order["carrier"], order["tracking_number"])
            facts.append(fact)
        return facts

    def search_products(self, query: str, limit: int = 3) -> list[dict]:
        q_tokens = set(tokenize(query))
        scored = []
        for product in self.db.all_products():
            overlap = len(q_tokens & set(tokenize(f"{product['name']} {product['description']}")))
            if overlap:
                scored.append((overlap, product))
        scored.sort(key=lambda item: -item[0])
        return [
            {**p, "availability": "in stock" if p["stock"] > 0 else "out of stock"}
            for _, p in scored[:limit]
        ]
