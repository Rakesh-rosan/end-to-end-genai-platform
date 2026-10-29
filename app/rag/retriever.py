"""Hybrid retrieval: vector recall, then keyword re-scoring.

Pure vector search misses exact terms (SKUs, order ids, "30 days"); pure
keyword search misses paraphrases. Blending both is the usual baseline. The
next upgrade is a cross-encoder re-ranker (e.g. Cohere Rerank, bge-reranker).
"""

from dataclasses import dataclass

from app.llm.embeddings import tokenize
from app.rag.vector_store import Chunk, InMemoryVectorStore


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class HybridRetriever:
    def __init__(self, store: InMemoryVectorStore, top_k: int = 4, min_score: float = 0.15, vector_weight: float = 0.7) -> None:
        self.store = store
        self.top_k = top_k
        self.min_score = min_score
        self.vector_weight = vector_weight

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        q_tokens = set(tokenize(query))
        results = []
        for chunk, vector_score in self.store.search(query, k=self.top_k * 3):
            keyword_score = len(q_tokens & set(tokenize(chunk.text))) / len(q_tokens) if q_tokens else 0.0
            score = self.vector_weight * vector_score + (1 - self.vector_weight) * keyword_score
            if score >= self.min_score:
                results.append(RetrievedChunk(chunk, round(score, 4)))
        results.sort(key=lambda r: -r.score)
        return results[: self.top_k]
