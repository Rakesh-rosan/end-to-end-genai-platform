"""Change Data Capture worker: keeps derived data (vector index, semantic cache) in sync with sources.

Two sources, one idea - detect a change, then invalidate/rebuild what depends on it:
- Commerce DB: reads change_log rows written by triggers -> invalidates cache tags product:<id>.
- Knowledge docs: compares file hashes -> re-ingests changed docs -> invalidates doc:<name>.

Without this, the semantic cache keeps serving yesterday's price after a price change.
In production: Debezium/Kafka (DB) and S3 event notifications (docs) push
events to this consumer instead of it polling.
"""

import asyncio
import hashlib
import logging

from app.cache.semantic_cache import SemanticCache
from app.commerce.db import CommerceDB
from app.observability import log_event, metrics
from app.rag.ingest import Ingestor

logger = logging.getLogger(__name__)


class CDCWorker:
    def __init__(self, db: CommerceDB, cache: SemanticCache, ingestor: Ingestor, poll_seconds: float = 2.0) -> None:
        self.db = db
        self.cache = cache
        self.ingestor = ingestor
        self.poll_seconds = poll_seconds
        self.cursor = db.max_change_seq()
        self.doc_hashes: dict[str, str] = self._hash_docs()

    def _hash_docs(self) -> dict[str, str]:
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.ingestor.iter_files()}

    def poll_once(self) -> dict:
        tags: set[str] = set()

        changes = self.db.changes_since(self.cursor)
        if changes:
            self.cursor = changes[-1]["seq"]
            tags |= {f"{c['entity']}:{c['entity_id']}" for c in changes}

        current = self._hash_docs()
        paths = {p.name: p for p in self.ingestor.iter_files()}
        for name, digest in current.items():
            if self.doc_hashes.get(name) != digest:
                self.ingestor.ingest_file(paths[name])
                tags.add(f"doc:{name}")
        for name in set(self.doc_hashes) - set(current):
            self.ingestor.store.delete_document(name)
            tags.add(f"doc:{name}")
        self.doc_hashes = current

        invalidated = self.cache.invalidate_tags(tags) if tags else 0
        if tags:
            metrics.inc("cdc_events_total", len(tags))
            metrics.inc("cache_invalidations_total", invalidated)
            log_event(logger, "cdc_applied", tags=sorted(tags), invalidated=invalidated)
        return {"tags": sorted(tags), "invalidated": invalidated}

    async def run(self) -> None:
        while True:
            try:
                await asyncio.to_thread(self.poll_once)
            except Exception:
                logger.exception("cdc poll failed")
            await asyncio.sleep(self.poll_seconds)
