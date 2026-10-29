"""Knowledge ingestion: load -> chunk -> embed -> upsert.

Chunking strategy matters more than most people expect. Here: split on
markdown headings, pack paragraphs up to max_chars, and prefix each chunk with
its section heading so the chunk is self-describing when retrieved alone.
"""

import logging
import re
from pathlib import Path

from app.rag.vector_store import Chunk, InMemoryVectorStore

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")


def chunk_text(text: str, max_chars: int = 800, overlap: int = 120) -> list[tuple[str, str]]:
    """Return [(section_heading, chunk_text)]."""
    chunks: list[tuple[str, str]] = []
    section, buffer = "", ""

    def flush() -> None:
        nonlocal buffer
        if buffer:
            chunks.append((section, f"{section}\n{buffer}" if section else buffer))
            buffer = ""

    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if para.startswith("#"):
            flush()
            heading, _, para = para.partition("\n")
            section = heading.lstrip("#").strip()
            para = para.strip()
            if not para:
                continue
        if len(buffer) + len(para) + 1 > max_chars:
            flush()
        while len(para) > max_chars:  # one huge paragraph: sliding window with overlap
            buffer = para[:max_chars]
            flush()
            para = para[max_chars - overlap:]
        buffer = f"{buffer}\n{para}" if buffer else para
    flush()
    return chunks


class Ingestor:
    def __init__(self, store: InMemoryVectorStore, knowledge_dir: Path) -> None:
        self.store = store
        self.knowledge_dir = knowledge_dir

    def iter_files(self) -> list[Path]:
        if not self.knowledge_dir.exists():
            return []
        return sorted(p for p in self.knowledge_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)

    def ingest_file(self, path: Path) -> int:
        doc_id = path.name
        pieces = chunk_text(load_text(path))
        chunks = [
            Chunk(id=f"{doc_id}#{i}", doc_id=doc_id, text=text, metadata={"source": doc_id, "section": section})
            for i, (section, text) in enumerate(pieces)
        ]
        self.store.upsert_document(doc_id, chunks)
        logger.info("ingested %s (%d chunks)", doc_id, len(chunks))
        return len(chunks)

    def ingest_dir(self) -> dict[str, int]:
        return {p.name: self.ingest_file(p) for p in self.iter_files()}
