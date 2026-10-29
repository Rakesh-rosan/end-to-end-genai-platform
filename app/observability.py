"""Structured logging, Prometheus-style metrics and a per-request pipeline trace.

In production you would export these with OpenTelemetry (traces) and the
prometheus_client library (metrics). The shapes here are the same, so swapping
is mechanical.
"""

import contextvars
import json
import logging
import threading
import time
from collections import defaultdict
from contextlib import contextmanager

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def log_event(logger: logging.Logger, msg: str, **fields) -> None:
    logger.info(msg, extra={"fields": fields})


class Metrics:
    """Tiny in-process metrics registry rendered in Prometheus text format."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple, float] = defaultdict(float)
        self._sums: dict[tuple, float] = defaultdict(float)
        self._counts: dict[tuple, int] = defaultdict(int)

    @staticmethod
    def _key(name: str, labels: dict) -> tuple:
        return name, tuple(sorted((k, str(v)) for k, v in labels.items()))

    def inc(self, name: str, value: float = 1.0, **labels) -> None:
        with self._lock:
            self._counters[self._key(name, labels)] += value

    def observe(self, name: str, value: float, **labels) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._sums[key] += value
            self._counts[key] += 1

    def render(self) -> str:
        def fmt(labels: tuple) -> str:
            return "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}" if labels else ""

        lines = []
        with self._lock:
            for (name, labels), v in sorted(self._counters.items()):
                lines.append(f"{name}{fmt(labels)} {v}")
            for (name, labels), v in sorted(self._sums.items()):
                lines.append(f"{name}_sum{fmt(labels)} {v}")
                lines.append(f"{name}_count{fmt(labels)} {self._counts[(name, labels)]}")
        return "\n".join(lines) + "\n"


metrics = Metrics()


class Trace:
    """Records how long each pipeline stage took. Returned to the client so you can *see* the architecture work."""

    def __init__(self) -> None:
        self.spans: list[dict] = []

    @contextmanager
    def span(self, stage: str, **detail):
        record = {"stage": stage, **detail}
        start = time.perf_counter()
        try:
            yield record
        finally:
            record["ms"] = round((time.perf_counter() - start) * 1000, 2)
            self.spans.append(record)
            metrics.observe("stage_latency_ms", record["ms"], stage=stage)
