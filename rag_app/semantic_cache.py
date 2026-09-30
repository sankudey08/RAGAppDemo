"""Persistent semantic cache for successful grounded answers."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from hashlib import sha256
import math

from chromadb import Collection
from chromadb.errors import ChromaError
import httpx

from .answering import Citation
from .config import Settings
from .vector_store import DocumentVectorStore, EmbeddingError, StorageError

logger = logging.getLogger(__name__)
CACHE_SCHEMA_VERSION = 1


class CacheError(RuntimeError):
    """A cache lookup or write could not be completed."""


@dataclass(frozen=True)
class CachedAnswer:
    """A successful answer and its still-valid source citations."""

    query: str
    answer: str
    citations: tuple[Citation, ...]
    similarity: float


class SemanticAnswerCache:
    """Store query embeddings and answer payloads in a dedicated Chroma collection."""

    def __init__(self, settings: Settings, vector_store: DocumentVectorStore) -> None:
        self._settings = settings
        self._vector_store = vector_store
        try:
            self._collection: Collection = vector_store.get_or_create_collection(
                settings.cache_collection_name,
                metadata={
                    "description": "Semantic cache of successful grounded answers",
                    "hnsw:space": "cosine",
                },
            )
        except StorageError as error:
            raise CacheError(str(error)) from error

    def lookup(self, query: str) -> CachedAnswer | None:
        """Return the closest valid answer above threshold, removing stale entries."""

        query = query.strip()
        if not query:
            raise ValueError("Cache query cannot be empty.")
        try:
            cache_count = self._collection.count()
            if not cache_count:
                return None
            query_embedding = self._vector_store.embed_texts([query])
            matches = self._collection.query(
                query_embeddings=query_embedding,
                n_results=min(cache_count, 10),
                include=["documents", "distances"],
            )
            stored_documents = self._vector_store.list_documents()
        except (EmbeddingError, StorageError) as error:
            raise CacheError(f"Could not check the semantic cache: {error}") from error
        except (ChromaError, OSError, httpx.HTTPError) as error:
            raise CacheError(f"Could not check the semantic cache: {error}") from error

        current_versions = {document.document_id: document.ingested_at for document in stored_documents}
        ids = (matches.get("ids") or [[]])[0]
        payloads = (matches.get("documents") or [[]])[0]
        distances = (matches.get("distances") or [[]])[0]
        stale_ids: list[str] = []
        for cache_id, payload_text, distance in zip(ids, payloads, distances):
            if payload_text is None or distance is None:
                stale_ids.append(cache_id)
                continue
            similarity = max(0.0, min(1.0, 1.0 - float(distance)))
            if similarity < self._settings.cache_similarity_threshold:
                break
            try:
                payload = json.loads(payload_text)
                if not self._is_valid(payload, current_versions):
                    stale_ids.append(cache_id)
                    continue
                citations = tuple(_citation_from_payload(item) for item in payload["citations"])
                answer = payload["answer"]
                cached_query = payload["query"]
                if not isinstance(answer, str) or not answer.strip() or not isinstance(cached_query, str):
                    raise ValueError("Cached answer payload has invalid text fields.")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                stale_ids.append(cache_id)
                logger.warning("Discarding malformed semantic cache entry %s: %s", cache_id, error)
                continue
            if stale_ids:
                self._delete_entries(stale_ids)
            return CachedAnswer(
                query=cached_query,
                answer=answer,
                citations=citations,
                similarity=similarity,
            )

        if stale_ids:
            self._delete_entries(stale_ids)
        return None

    def store(
        self,
        query: str,
        answer: str,
        citations: tuple[Citation, ...],
    ) -> None:
        """Persist a successful answer with its source-version snapshot."""

        query = query.strip()
        if not query or not answer.strip():
            raise ValueError("Only non-empty successful queries and answers can be cached.")
        if not citations:
            raise ValueError("Answers without source citations are not cacheable.")
        if any(not citation.document_id for citation in citations):
            raise ValueError("A citation is missing document provenance; the answer cannot be cached safely.")

        try:
            query_embedding = self._vector_store.embed_texts([query])[0]
            stored_documents = self._vector_store.list_documents()
            current_versions = {document.document_id: document.ingested_at for document in stored_documents}
            source_versions: dict[str, str] = {}
            for citation in citations:
                current_version = current_versions[citation.document_id]
                if citation.ingested_at != current_version:
                    raise ValueError(
                        f"Cited source '{citation.filename}' changed during answer generation; "
                        "the answer cannot be cached safely."
                    )
                source_versions[citation.document_id] = current_version
        except KeyError as error:
            raise CacheError("A cited source is no longer indexed; refusing to cache the answer.") from error
        except ValueError:
            raise
        except (EmbeddingError, StorageError) as error:
            raise CacheError(f"Could not prepare the semantic cache entry: {error}") from error
        except (ChromaError, OSError, httpx.HTTPError) as error:
            raise CacheError(f"Could not prepare the semantic cache entry: {error}") from error

        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "query": query,
            "answer": answer,
            "citations": [
                {
                    "reference": citation.reference,
                    "filename": citation.filename,
                    "page": citation.page,
                    "snippet": citation.snippet,
                    "similarity": citation.similarity,
                    "document_id": citation.document_id,
                    "ingested_at": citation.ingested_at,
                }
                for citation in citations
            ],
            "source_versions": source_versions,
            "chat_model": self._settings.ollama_chat_model,
            "embedding_model": self._settings.ollama_embedding_model,
            "retrieval_top_k": self._settings.retrieval_top_k,
            "retrieval_min_score": self._settings.retrieval_min_score,
        }
        cache_id = sha256(query.casefold().encode("utf-8")).hexdigest()
        try:
            self._collection.upsert(
                ids=[cache_id],
                embeddings=[query_embedding],
                documents=[json.dumps(payload, separators=(",", ":"), ensure_ascii=False)],
                metadatas=[{"created_at": datetime.now(timezone.utc).isoformat()}],
            )
        except (ChromaError, OSError) as error:
            raise CacheError(f"Could not store the semantic cache entry: {error}") from error

    def _is_valid(self, payload: object, current_versions: dict[str, str]) -> bool:
        if not isinstance(payload, dict):
            return False
        if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
            return False
        if payload.get("chat_model") != self._settings.ollama_chat_model:
            return False
        if payload.get("embedding_model") != self._settings.ollama_embedding_model:
            return False
        if payload.get("retrieval_top_k") != self._settings.retrieval_top_k:
            return False
        if payload.get("retrieval_min_score") != self._settings.retrieval_min_score:
            return False
        source_versions = payload.get("source_versions")
        if not isinstance(source_versions, dict) or not source_versions:
            return False
        return all(current_versions.get(document_id) == ingested_at for document_id, ingested_at in source_versions.items())

    def _delete_entries(self, cache_ids: list[str]) -> None:
        try:
            self._collection.delete(ids=cache_ids)
        except (ChromaError, OSError) as error:
            raise CacheError(f"Could not remove stale cache entries: {error}") from error


def _citation_from_payload(payload: object) -> Citation:
    if not isinstance(payload, dict):
        raise ValueError("Cached citation must be an object.")
    page = payload.get("page")
    similarity = float(payload["similarity"])
    if not math.isfinite(similarity):
        raise ValueError("Cached citation similarity is not finite.")
    return Citation(
        reference=str(payload["reference"]),
        filename=str(payload["filename"]),
        page=int(page) if page is not None else None,
        snippet=str(payload["snippet"]),
        similarity=similarity,
        document_id=str(payload["document_id"]),
        ingested_at=str(payload["ingested_at"]),
    )
