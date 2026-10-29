"""In-memory vector store (brute-force cosine search with numpy).

Fine for thousands of chunks. Beyond that, use a vector DB with ANN indexes
(Qdrant, pgvector, OpenSearch, Pinecone). The interface - upsert by document,
delete by document, search - is the same one you will use there.
"""

import threading
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


class InMemoryVectorStore:
    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._lock = threading.Lock()

    def upsert_document(self, doc_id: str, chunks: list[Chunk]) -> None:
        """Replace every chunk of a document atomically (delete-then-insert)."""
        vectors = self.embedder.embed([c.text for c in chunks]) if chunks else None
        with self._lock:
            self._delete_locked(doc_id)
            if chunks:
                self._chunks.extend(chunks)
                self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])

    def delete_document(self, doc_id: str) -> None:
        with self._lock:
            self._delete_locked(doc_id)

    def _delete_locked(self, doc_id: str) -> None:
        keep = [i for i, c in enumerate(self._chunks) if c.doc_id != doc_id]
        if len(keep) == len(self._chunks):
            return
        self._chunks = [self._chunks[i] for i in keep]
        self._vectors = self._vectors[keep] if keep else None

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        q = self.embedder.embed([query])[0]
        with self._lock:
            if self._vectors is None:
                return []
            scores = self._vectors @ q
            chunks = list(self._chunks)
        order = np.argsort(-scores)[:k]
        return [(chunks[i], float(scores[i])) for i in order]

    def doc_ids(self) -> set[str]:
        with self._lock:
            return {c.doc_id for c in self._chunks}

    def __len__(self) -> int:
        return len(self._chunks)
