"""Chunking, Ollama embeddings, and persistent ChromaDB document storage."""

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
import re
from typing import Protocol

import os

import chromadb
from chromadb.errors import ChromaError
import httpx
from ollama import Client, ResponseError

from .config import Settings
from .ingestion import ExtractedDocument

# ChromaDB's bundled PostHog telemetry client is incompatible with the
# `posthog` version resolved in this project (its `capture()` signature
# changed), which raises a TypeError on every collection operation that
# ChromaDB logs as "Failed to send telemetry event ...". Setting
# `anonymized_telemetry=False` alone does not prevent the crash (the
# mismatch happens at argument-binding time, before any enabled/disabled
# check), so swap in a no-op telemetry client via the documented
# `CHROMA_PRODUCT_TELEMETRY_IMPL` env var instead. Both settings are applied
# via env vars (rather than passed explicitly to PersistentClient) so any
# chromadb.PersistentClient created against a given persist directory --
# ours or a bare one, e.g. in tests -- resolves the same effective settings
# and avoids Chroma's "different settings" conflict for the same path.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_PRODUCT_TELEMETRY_IMPL", "rag_app.telemetry.NoopProductTelemetryClient")


class EmbeddingError(RuntimeError):
    """An embedding request failed or returned invalid vectors."""


class StorageError(RuntimeError):
    """A ChromaDB storage operation failed."""


class Embedder(Protocol):
    def __call__(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one embedding vector for each text."""


@dataclass(frozen=True)
class TextChunk:
    """A retrieval-sized text segment with its original page provenance."""

    text: str
    filename: str
    page: int | None
    media_type: str
    ingested_at: str
    chunk_index: int


@dataclass(frozen=True)
class StoredDocument:
    """A document represented in the persistent collection."""

    document_id: str
    filename: str
    ingested_at: str
    chunk_count: int


@dataclass(frozen=True)
class RetrievalResult:
    """A matched chunk with its distance and source provenance."""

    text: str
    distance: float
    similarity: float
    filename: str
    page: int | None
    media_type: str
    ingested_at: str
    chunk_index: int
    document_id: str = ""


@dataclass(frozen=True)
class RetrievalResponse:
    """Top-k search results and a status explaining whether evidence was found."""

    query: str
    results: tuple[RetrievalResult, ...]
    status: str


def split_text(text: str, *, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[str]:
    """Split text into overlapping character-sized chunks without dropping text."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = _last_whitespace(text, start + 1, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break

        next_start = max(start + 1, end - chunk_overlap)
        overlap_boundary = _last_whitespace(text, start + 1, next_start + 1)
        if overlap_boundary > start:
            next_start = overlap_boundary + 1
        start = next_start

    return chunks


def chunk_document(
    document: ExtractedDocument,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[TextChunk]:
    """Chunk each extracted page independently while retaining provenance."""

    chunks: list[TextChunk] = []
    for page in document.pages:
        for chunk_index, text in enumerate(
            split_text(page.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            chunks.append(
                TextChunk(
                    text=text,
                    filename=page.source.filename,
                    page=page.source.page,
                    media_type=page.source.media_type,
                    ingested_at=page.source.ingested_at.isoformat(),
                    chunk_index=chunk_index,
                )
            )
    return chunks


class OllamaEmbedder:
    """Generate vectors with the configured local Ollama embedding model."""

    def __init__(self, base_url: str, model: str) -> None:
        self._client = Client(host=base_url)
        self._model = model

    def __call__(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return []
        try:
            result = self._client.embed(model=self._model, input=list(texts))
        except (ResponseError, httpx.HTTPError) as error:
            raise EmbeddingError(
                f"Could not generate embeddings with Ollama model '{self._model}': {error}"
            ) from error

        vectors = result.embeddings
        if len(vectors) != len(texts) or any(not vector for vector in vectors):
            raise EmbeddingError(
                f"Ollama model '{self._model}' returned {len(vectors)} embeddings "
                f"for {len(texts)} text chunks."
            )
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise EmbeddingError(f"Ollama model '{self._model}' returned inconsistent embedding dimensions.")
        return vectors


class DocumentVectorStore:
    """Persistent chunk store; documents are replaced by normalized filename."""

    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
    ) -> None:
        self._settings = settings
        self._embedder = embedder or OllamaEmbedder(
            str(settings.ollama_base_url),
            settings.ollama_embedding_model,
        )
        try:
            self._client = chromadb.PersistentClient(path=str(settings.chroma_persist_directory))
            self._collection = self._client.get_or_create_collection(
                name=settings.chroma_collection_name,
                metadata={
                    "description": "Persisted source document text chunks",
                    "hnsw:space": "cosine",
                },
            )
            self._distance_space = str((self._collection.metadata or {}).get("hnsw:space", "l2"))
        except (ChromaError, OSError) as error:
            raise StorageError(f"Could not open ChromaDB collection: {error}") from error

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate validated embeddings using the configured embedding provider."""

        return self._validated_embeddings(texts)

    def get_or_create_collection(
        self,
        name: str,
        *,
        metadata: dict[str, str],
    ) -> chromadb.Collection:
        """Create/open a collection sharing the configured persistent ChromaDB client."""

        try:
            return self._client.get_or_create_collection(name=name, metadata=metadata)
        except (ChromaError, OSError) as error:
            raise StorageError(f"Could not open ChromaDB collection '{name}': {error}") from error

    def index_document(self, document: ExtractedDocument) -> StoredDocument:
        """Embed and persist a document, replacing any prior version of its filename."""

        chunks = chunk_document(
            document,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        if not chunks:
            raise StorageError(f"'{document.filename}' produced no indexable text chunks.")

        vectors = self._validated_embeddings([chunk.text for chunk in chunks])
        document_id = _document_id(document.filename)
        ids = [_chunk_id(document_id, chunk) for chunk in chunks]
        metadatas = [_chunk_metadata(document_id, chunk) for chunk in chunks]

        try:
            existing = self._collection.get(where={"document_id": document_id}, include=["metadatas"])
            old_ids = existing["ids"]
            new_ids = set(ids)
            self._collection.upsert(
                ids=ids,
                embeddings=vectors,
                documents=[chunk.text for chunk in chunks],
                metadatas=metadatas,
            )
            obsolete_ids = [stored_id for stored_id in old_ids if stored_id not in new_ids]
            if obsolete_ids:
                self._collection.delete(ids=obsolete_ids)
        except (ChromaError, OSError) as error:
            raise StorageError(f"Could not store '{document.filename}' in ChromaDB: {error}") from error

        return StoredDocument(
            document_id=document_id,
            filename=document.filename,
            ingested_at=document.ingested_at.isoformat(),
            chunk_count=len(chunks),
        )

    def list_documents(self) -> list[StoredDocument]:
        """List indexed source documents without loading their chunk text."""

        try:
            result = self._collection.get(include=["metadatas"])
        except (ChromaError, OSError) as error:
            raise StorageError(f"Could not list stored documents: {error}") from error

        documents: dict[str, StoredDocument] = {}
        for metadata in result["metadatas"] or []:
            if metadata is None:
                continue
            document_id = str(metadata["document_id"])
            current = documents.get(document_id)
            if current is None:
                documents[document_id] = StoredDocument(
                    document_id=document_id,
                    filename=str(metadata["filename"]),
                    ingested_at=str(metadata["ingested_at"]),
                    chunk_count=1,
                )
            else:
                documents[document_id] = StoredDocument(
                    document_id=current.document_id,
                    filename=current.filename,
                    ingested_at=current.ingested_at,
                    chunk_count=current.chunk_count + 1,
                )
        return sorted(documents.values(), key=lambda document: document.filename.casefold())

    def delete_document(self, document_id: str) -> bool:
        """Remove all chunks for a document; return whether it existed."""

        try:
            existing = self._collection.get(where={"document_id": document_id}, include=["metadatas"])
            if not existing["ids"]:
                return False
            self._collection.delete(where={"document_id": document_id})
            return True
        except (ChromaError, OSError) as error:
            raise StorageError(f"Could not remove document from ChromaDB: {error}") from error

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> RetrievalResponse:
        """Retrieve nearest chunks, filtering results below a cosine similarity threshold."""

        query = query.strip()
        if not query:
            raise ValueError("Search query cannot be empty.")

        requested_k = self._settings.retrieval_top_k if top_k is None else top_k
        if requested_k < 1:
            raise ValueError("top_k must be at least 1.")
        threshold = self._settings.retrieval_min_score if min_similarity is None else min_similarity
        if not 0 <= threshold <= 1:
            raise ValueError("min_similarity must be between 0 and 1.")

        try:
            collection_count = self._collection.count()
        except (ChromaError, OSError) as error:
            raise StorageError(f"Could not inspect the ChromaDB collection: {error}") from error
        if collection_count == 0:
            return RetrievalResponse(query=query, results=(), status="empty_collection")

        try:
            query_vectors = self._validated_embeddings([query])
            matches = self._collection.query(
                query_embeddings=query_vectors,
                n_results=min(requested_k, collection_count),
                include=["documents", "metadatas", "distances"],
            )
        except EmbeddingError:
            raise
        except (ChromaError, OSError) as error:
            raise StorageError(f"Could not search the ChromaDB collection: {error}") from error

        distances = (matches.get("distances") or [[]])[0]
        documents = (matches.get("documents") or [[]])[0]
        metadatas = (matches.get("metadatas") or [[]])[0]
        results: list[RetrievalResult] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            if text is None or metadata is None or distance is None:
                continue
            similarity = _distance_to_similarity(float(distance), self._distance_space)
            if similarity < threshold:
                continue
            raw_page = metadata.get("page")
            results.append(
                RetrievalResult(
                    text=text,
                    distance=float(distance),
                    similarity=similarity,
                    filename=str(metadata["filename"]),
                    page=int(raw_page) if raw_page is not None else None,
                    media_type=str(metadata["media_type"]),
                    ingested_at=str(metadata["ingested_at"]),
                    chunk_index=int(metadata["chunk_index"]),
                    document_id=str(metadata["document_id"]),
                )
            )

        status = "ok" if results else "no_relevant_results"
        return RetrievalResponse(query=query, results=tuple(results), status=status)

    def _validated_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            vectors = self._embedder(texts)
        except EmbeddingError:
            raise
        except (ResponseError, httpx.HTTPError) as error:
            raise EmbeddingError(f"Could not generate document embeddings: {error}") from error

        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"Embedding provider returned {len(vectors)} vectors for {len(texts)} chunks."
            )
        if not vectors or any(not vector for vector in vectors):
            raise EmbeddingError("Embedding provider returned an empty vector.")
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise EmbeddingError("Embedding provider returned vectors with inconsistent dimensions.")
        try:
            normalized_vectors = [[float(value) for value in vector] for vector in vectors]
        except (TypeError, ValueError) as error:
            raise EmbeddingError("Embedding provider returned a non-numeric vector value.") from error
        if any(not math.isfinite(value) for vector in normalized_vectors for value in vector):
            raise EmbeddingError("Embedding provider returned a non-finite vector value.")
        return normalized_vectors


def _document_id(filename: str) -> str:
    normalized_name = re.sub(r"\s+", " ", Path(filename).name.strip()).casefold()
    return sha256(normalized_name.encode("utf-8")).hexdigest()


def _last_whitespace(text: str, start: int, end: int) -> int:
    return next((index for index in range(end - 1, start - 1, -1) if text[index].isspace()), -1)


def _distance_to_similarity(distance: float, space: str) -> float:
    """Convert supported Chroma distances to a comparable 0-1 relevance score."""

    if space == "cosine":
        return max(0.0, min(1.0, 1.0 - distance))
    if space == "l2":
        return 1.0 / (1.0 + distance**0.5)
    if space == "ip":
        return max(0.0, min(1.0, 1.0 - distance))
    raise StorageError(f"Unsupported ChromaDB distance space '{space}'.")


def _chunk_id(document_id: str, chunk: TextChunk) -> str:
    page_id = chunk.page if chunk.page is not None else 0
    return f"{document_id}:p{page_id}:c{chunk.chunk_index}"


def _chunk_metadata(document_id: str, chunk: TextChunk) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {
        "document_id": document_id,
        "filename": chunk.filename,
        "media_type": chunk.media_type,
        "ingested_at": chunk.ingested_at,
        "chunk_index": chunk.chunk_index,
    }
    if chunk.page is not None:
        metadata["page"] = chunk.page
    return metadata
