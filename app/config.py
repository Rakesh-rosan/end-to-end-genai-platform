from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All runtime configuration. Every field can be overridden by an env var of the same name."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "shopassist"

    # API gateway
    jwt_secret: str = "dev-only-secret-change-me-in-production-0001"
    jwt_issuer: str = "shopassist"
    jwt_ttl_minutes: int = 60
    rate_limit_per_minute: int = 20
    redis_url: str | None = None

    # LLM gateway
    llm_provider: str = "mock"  # mock | anthropic | openai
    # Read from .env here and passed to the SDK explicitly: values in .env are
    # NOT exported to os.environ, so the SDKs would not find them on their own.
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    llm_model: str = "claude-opus-5-5"
    llm_router_model: str = "claude-opus-5-5"
    llm_effort: str = "low"
    llm_max_tokens: int = 2048

    # Embeddings / RAG
    embedding_provider: str = "hashing"  # hashing | sentence-transformers | openai
    rag_top_k: int = 4
    rag_min_score: float = 0.15

    # Semantic cache
    cache_similarity_threshold: float = 0.92
    cache_ttl_seconds: int = 3600

    # Guardrails
    max_input_chars: int = 2000
    grounding_threshold: float = 0.35
    pii_allowlist: list[str] = ["support@shopassist.example", "1-800-555-0199"]

    # HITL
    refund_hitl_threshold: float = 200.0

    # Data + CDC
    db_path: Path = ROOT / "data" / "commerce.db"
    knowledge_dir: Path = ROOT / "data" / "knowledge"
    cdc_enabled: bool = True
    cdc_poll_seconds: float = 2.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
