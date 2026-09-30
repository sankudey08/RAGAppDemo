"""Offline end-to-end validation across extraction, storage, retrieval, workflow, and cache."""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from rag_app.answering import OllamaAnswerGenerator
from rag_app.config import Settings
from rag_app.ingestion import IngestionError, extract_document
from rag_app.semantic_cache import SemanticAnswerCache
from rag_app.vector_store import DocumentVectorStore, EmbeddingError, OllamaEmbedder
from rag_app.workflow import RAGWorkflow


def local_embeddings(texts: list[str]) -> list[list[float]]:
    """A deterministic embedding boundary used instead of any live model service."""

    vectors = {
        "What are apples?": [1.0, 0.0],
        "Tell me about apples": [0.999, 0.02],
        "Apples are fruits.": [0.99, 0.1],
        "What is a carburetor?": [0.0, 1.0],
    }
    return [vectors.get(text, [0.99, 0.1]) for text in texts]


class LocalChatClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> list[SimpleNamespace]:
        self.calls.append(kwargs)
        assert kwargs["stream"] is True
        return [
            SimpleNamespace(message=SimpleNamespace(content="Apples are fruits ")),
            SimpleNamespace(message=SimpleNamespace(content="according to the source [S1].")),
        ]


def make_settings(path: Path, **overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        chroma_persist_directory=path,
        chroma_collection_name="e2e_documents",
        cache_collection_name="e2e_cache",
        chunk_size=100,
        chunk_overlap=0,
        retrieval_min_score=0.5,
        cache_enabled=True,
        cache_similarity_threshold=0.95,
        **overrides,
    )


def test_end_to_end_cache_hit_miss_and_persistence_without_external_services(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "chroma")
    store = DocumentVectorStore(settings, embedder=local_embeddings)
    extracted = extract_document("fruit.txt", b"Apples are fruits.")
    stored = store.index_document(extracted)
    cache = SemanticAnswerCache(settings, store)
    chat_client = LocalChatClient()
    generator = OllamaAnswerGenerator(
        settings,
        store,  # type: ignore[arg-type]
        chat_client=chat_client,  # type: ignore[arg-type]
    )
    workflow = RAGWorkflow(
        settings,
        store,
        answer_generator=generator,
        semantic_cache=cache,
    )

    miss_events = list(workflow.stream("What are apples?"))
    miss_result = miss_events[-1].result
    assert miss_result is not None
    assert miss_result.status == "answered"
    assert "".join(event.token for event in miss_events if event.kind == "token") == miss_result.answer
    assert miss_result.citations[0].filename == "fruit.txt"
    assert miss_result.citations[0].page is None
    assert miss_result.citations[0].document_id == stored.document_id
    assert len(chat_client.calls) == 1

    reopened_store = DocumentVectorStore(settings, embedder=local_embeddings)
    reopened_cache = SemanticAnswerCache(settings, reopened_store)
    reopened_workflow = RAGWorkflow(
        settings,
        reopened_store,
        answer_generator=OllamaAnswerGenerator(
            settings,
            reopened_store,
            chat_client=chat_client,  # type: ignore[arg-type]
        ),
        semantic_cache=reopened_cache,
    )
    hit_result = reopened_workflow.run("Tell me about apples")

    assert hit_result.status == "cache_hit"
    assert hit_result.cache_hit
    assert hit_result.answer == miss_result.answer
    assert hit_result.citations == miss_result.citations
    assert len(chat_client.calls) == 1
    assert reopened_store.list_documents()[0].filename == "fruit.txt"


def test_empty_index_completes_without_embedding_or_chat(tmp_path: Path) -> None:
    store = DocumentVectorStore(
        make_settings(tmp_path / "chroma"),
        embedder=lambda texts: pytest.fail("empty collection must not embed"),
    )
    client = LocalChatClient()
    workflow = RAGWorkflow(
        make_settings(tmp_path / "chroma"),
        store,
        answer_generator=OllamaAnswerGenerator(
            make_settings(tmp_path / "chroma"),
            store,
            chat_client=client,  # type: ignore[arg-type]
        ),
        semantic_cache=SemanticAnswerCache(make_settings(tmp_path / "chroma"), store),
    )

    result = workflow.run("What are apples?")

    assert result.status == "empty_collection"
    assert "do not contain sufficiently relevant" in result.answer
    assert result.citations == ()
    assert client.calls == []


def test_malformed_upload_is_rejected_without_affecting_indexed_sources(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "chroma")
    store = DocumentVectorStore(settings, embedder=local_embeddings)
    stored = store.index_document(extract_document("good.txt", b"Apples are fruits."))

    with pytest.raises(IngestionError, match="Malformed or unreadable PDF"):
        extract_document("bad.pdf", b"not a pdf")

    assert store.list_documents()[0].document_id == stored.document_id


def test_ollama_embedding_connection_failure_is_actionable() -> None:
    class OfflineClient:
        def embed(self, **kwargs: Any) -> None:
            raise httpx.ConnectError("connection refused")

    embedder = OllamaEmbedder("http://localhost:11434", "missing-model")
    embedder._client = OfflineClient()  # type: ignore[assignment]

    with pytest.raises(EmbeddingError, match="Could not generate embeddings.*connection refused"):
        embedder(["sample"])
