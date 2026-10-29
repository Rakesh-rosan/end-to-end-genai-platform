"""Composition root: builds every component once and wires dependencies explicitly."""

from dataclasses import dataclass

from app.cache.semantic_cache import SemanticCache
from app.cdc.worker import CDCWorker
from app.commerce.db import CommerceDB
from app.config import Settings
from app.gateway.rate_limit import build_rate_limiter
from app.guardrails.input_guard import InputGuard
from app.guardrails.output_guard import OutputGuard
from app.llm.embeddings import build_embedder
from app.llm.gateway import LLMGateway, build_llm_gateway
from app.memory import ConversationMemory
from app.orchestrator.orchestrator import Orchestrator
from app.orchestrator.router import IntentRouter
from app.orchestrator.tools import CarrierAPI, CommerceTools
from app.pipeline import ChatPipeline
from app.rag.ingest import Ingestor
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import InMemoryVectorStore


@dataclass
class Services:
    db: CommerceDB
    store: InMemoryVectorStore
    ingestor: Ingestor
    cache: SemanticCache
    llm: LLMGateway
    pipeline: ChatPipeline
    rate_limiter: object
    cdc: CDCWorker


def build_services(settings: Settings) -> Services:
    db = CommerceDB(settings.db_path)
    openai_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    embedder = build_embedder(settings.embedding_provider, openai_key)

    store = InMemoryVectorStore(embedder)
    ingestor = Ingestor(store, settings.knowledge_dir)
    ingestor.ingest_dir()

    cache = SemanticCache(embedder, settings.cache_similarity_threshold, settings.cache_ttl_seconds)
    llm = build_llm_gateway(settings)
    orchestrator = Orchestrator(
        router=IntentRouter(llm, settings.llm_router_model),
        retriever=HybridRetriever(store, settings.rag_top_k, settings.rag_min_score),
        tools=CommerceTools(db, CarrierAPI()),
        db=db,
        llm=llm,
        settings=settings,
    )
    pipeline = ChatPipeline(
        InputGuard(settings.max_input_chars),
        OutputGuard(settings.grounding_threshold, settings.pii_allowlist),
        cache,
        orchestrator,
        ConversationMemory(),
    )
    return Services(
        db=db, store=store, ingestor=ingestor, cache=cache, llm=llm, pipeline=pipeline,
        rate_limiter=build_rate_limiter(settings),
        cdc=CDCWorker(db, cache, ingestor, settings.cdc_poll_seconds),
    )
