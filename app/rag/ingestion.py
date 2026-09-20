"""Ingestion: parse -> chunk -> embed -> store. Written as a plain service so that the API calls it
today and a Celery worker (roadmap week 5) can call it unchanged.

The document id is a hash of the file's bytes: uploading the same file twice (or a retried job)
is idempotent, while a changed file is a new document. Delete the old one explicitly if a new
version replaces it.
"""

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import PurePath

from app.core.embeddings import Embedder
from app.observability import metrics
from app.rag.chunking import Chunker
from app.rag.documents import DocumentRecord, DocumentRegistry
from app.rag.parsing import Section, parse_document
from app.rag.vector_store import QdrantChunkStore

log = logging.getLogger(__name__)

_H1 = re.compile(r"^#\s+(.+?)\s*#*$", re.MULTILINE)


class DocumentTooLargeError(ValueError):
    pass


@dataclass(frozen=True)
class DocumentUpload:
    filename: str
    content_type: str | None
    data: bytes
    title: str | None = None


def document_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:24]


def infer_title(sections: list[Section], filename: str) -> str:
    if m := _H1.search(sections[0].text[:2000]):
        return m.group(1).strip()
    return PurePath(filename).stem or "untitled"


class IngestionService:
    def __init__(
        self,
        store: QdrantChunkStore,
        registry: DocumentRegistry,
        embedder: Embedder,
        chunker: Chunker,
        *,
        chunker_name: str,
        max_bytes: int,
    ):
        self._store = store
        self._registry = registry
        self._embedder = embedder
        self._chunker = chunker
        self._chunker_name = chunker_name
        self.max_bytes = max_bytes

    async def ingest(self, tenant_id: str, upload: DocumentUpload) -> DocumentRecord:
        if len(upload.data) > self.max_bytes:
            raise DocumentTooLargeError(f"document exceeds {self.max_bytes} bytes")
        start = time.perf_counter()
        # PDF parsing is CPU-bound; keep it off the event loop.
        sections = await asyncio.to_thread(
            parse_document, upload.data, upload.filename, upload.content_type
        )
        title = (upload.title or "").strip() or infer_title(sections, upload.filename)
        doc_id = document_id(upload.data)
        chunks = self._chunker.chunk(sections, title)
        vectors = await self._embedder.embed_documents([c.text for c in chunks])
        await self._store.replace_document(tenant_id, doc_id, title, chunks, vectors)

        pages = [s.page for s in sections if s.page is not None]
        record = DocumentRecord(
            doc_id=doc_id,
            title=title,
            filename=upload.filename,
            content_type=upload.content_type,
            size_bytes=len(upload.data),
            pages=max(pages) if pages else None,
            chunks=len(chunks),
            chunker=self._chunker_name,
            created_at=datetime.now(UTC),
        )
        await self._registry.put(tenant_id, record)

        elapsed = time.perf_counter() - start
        metrics.RAG_DOCUMENTS_INGESTED.inc()
        metrics.RAG_CHUNKS_INGESTED.inc(len(chunks))
        metrics.RAG_INGEST_LATENCY.observe(elapsed)
        log.info(
            "document ingested",
            extra={
                "tenant_id": tenant_id,
                "doc_id": doc_id,
                "chunks": len(chunks),
                "bytes": len(upload.data),
                "duration_ms": round(elapsed * 1000, 1),
            },
        )
        return record

    async def delete(self, tenant_id: str, doc_id: str) -> bool:
        if await self._registry.get(tenant_id, doc_id) is None:
            return False
        await self._store.delete_document(tenant_id, doc_id)
        return await self._registry.delete(tenant_id, doc_id)
