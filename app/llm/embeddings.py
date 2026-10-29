"""Embedders turn text into unit-length vectors so cosine similarity == dot product.

HashingEmbedder is lexical (shared words -> similar vectors). It needs no model
download, which keeps the project runnable anywhere, but it will not match
"refund" with "money back". Switch EMBEDDING_PROVIDER to sentence-transformers
(or a hosted embedding API such as Voyage) to see real semantic matching.
"""

import hashlib
import re

import numpy as np

STOPWORDS = {
    "s", "t", "a", "an", "and", "are", "as", "at", "be", "can", "do", "does", "for", "from", "have", "how", "i",
    "if", "in", "is", "it", "me", "my", "of", "on", "or", "our", "please", "so", "that", "the", "this",
    "to", "was", "we", "what", "when", "where", "which", "who", "why", "will", "with", "you", "your",
}


def tokenize(text: str) -> list[str]:
    tokens = []
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word in STOPWORDS:
            continue
        if len(word) > 3 and word.endswith("s"):
            word = word[:-1]  # crude stemming: "returns" -> "return"
        tokens.append(word)
    return tokens


class HashingEmbedder:
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for feature in features:
                h = int.from_bytes(hashlib.md5(feature.encode()).digest()[:8], "little")
                vectors[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)


class OpenAIEmbedder:
    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, max_retries=2, timeout=30.0)
        self._model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        response = self._client.embeddings.create(model=self._model, input=texts)
        vectors = np.asarray([d.embedding for d in response.data], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def build_embedder(provider: str, openai_api_key: str | None = None):
    if provider == "openai":
        return OpenAIEmbedder(openai_api_key)
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder()
    return HashingEmbedder()
