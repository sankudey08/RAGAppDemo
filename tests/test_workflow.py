from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from rag_app.answering import Citation, GenerationError, OllamaAnswerGenerator
from rag_app.config import Settings
from rag_app.semantic_cache import CacheError, CachedAnswer
from rag_app.vector_store import RetrievalResponse, RetrievalResult, StorageError
from rag_app.workflow import RAGWorkflow


def make_source() -> RetrievalResult:
    return RetrievalResult(
        text="Apples are fruits.",
        distance=0.1,
        similarity=0.9,
        filename="fruit.txt",
        page=1,
        media_type="txt",
        ingested_at=datetime.now(timezone.utc).isoformat(),
        chunk_index=0,
    )


class FakeStore:
    def __init__(self, response: RetrievalResponse | Exception) -> None:
        self.response = response

    def search(self, query: str) -> RetrievalResponse:
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeChatClient:
    def __init__(self, chunks: list[str] | None = None, error: Exception | None = None) -> None:
        self.chunks = chunks or []
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> list[SimpleNamespace]:
        self.calls.append(kwargs)
        if self.error:
            def fail() -> Any:
                raise self.error

            return fail()
        return [
            SimpleNamespace(message=SimpleNamespace(content=chunk))
            for chunk in self.chunks
        ]


def make_workflow(
    retrieval: RetrievalResponse | Exception,
    *,
    chat_client: FakeChatClient | None = None,
    semantic_cache: Any | None = None,
    cache_enabled: bool = False,
) -> tuple[RAGWorkflow, FakeChatClient]:
    client = chat_client or FakeChatClient(["Apples ", "are fruits [S1]."])
    settings = Settings(_env_file=None, ollama_chat_model="test-model", cache_enabled=cache_enabled)
    store = FakeStore(retrieval)
    answer_generator = OllamaAnswerGenerator(
        settings,
        store,  # type: ignore[arg-type]
        chat_client=client,  # type: ignore[arg-type]
    )
    workflow = RAGWorkflow(
        settings,
        store,  # type: ignore[arg-type]
        answer_generator=answer_generator,
        semantic_cache=semantic_cache,  # type: ignore[arg-type]
    )
    return workflow, client


def test_workflow_streams_tokens_and_final_citations() -> None:
    source = make_source()
    workflow, client = make_workflow(
        RetrievalResponse(query="What are apples?", results=(source,), status="ok")
    )

    events = list(workflow.stream("What are apples?"))

    assert [event.token for event in events if event.kind == "token"] == [
        "Apples ",
        "are fruits [S1].",
    ]
    final = events[-1].result
    assert events[-1].kind == "complete"
    assert final is not None
    assert final.status == "answered"
    assert final.answer == "Apples are fruits [S1]."
    assert final.citations[0].filename == "fruit.txt"
    assert final.citations[0].page == 1
    assert client.calls[0]["stream"] is True


@pytest.mark.parametrize("status", ["empty_collection", "no_relevant_results"])
def test_workflow_routes_empty_retrieval_without_calling_chat(status: str) -> None:
    workflow, client = make_workflow(RetrievalResponse(query="question", results=(), status=status))

    result = workflow.run("question")

    assert result.status == status
    assert "do not contain sufficiently relevant" in result.answer
    assert result.citations == ()
    assert client.calls == []


def test_workflow_returns_retrieval_error_in_final_state() -> None:
    workflow, client = make_workflow(StorageError("database unavailable"))

    result = workflow.run("question")

    assert result.status == "error"
    assert result.error == "database unavailable"
    assert client.calls == []


def test_workflow_returns_generation_error_and_keeps_citations() -> None:
    workflow, _ = make_workflow(
        RetrievalResponse(query="question", results=(make_source(),), status="ok"),
        chat_client=FakeChatClient(error=httpx.ConnectError("Ollama unavailable")),
    )

    result = workflow.run("question")

    assert result.status == "error"
    assert "Ollama unavailable" in (result.error or "")
    assert result.citations[0].filename == "fruit.txt"


def test_workflow_rejects_blank_question() -> None:
    workflow, _ = make_workflow(RetrievalResponse(query="", results=(), status="empty_collection"))

    with pytest.raises(ValueError, match="Question cannot be empty"):
        workflow.run("  ")


class FakeSemanticCache:
    def __init__(self, cached: CachedAnswer | None = None, store_error: Exception | None = None) -> None:
        self.cached = cached
        self.store_error = store_error
        self.lookups: list[str] = []
        self.stores: list[tuple[str, str, tuple[Citation, ...]]] = []

    def lookup(self, query: str) -> CachedAnswer | None:
        self.lookups.append(query)
        return self.cached

    def store(self, query: str, answer: str, citations: tuple[Citation, ...]) -> None:
        self.stores.append((query, answer, citations))
        if self.store_error:
            raise self.store_error


def test_workflow_returns_cache_hit_without_retrieval_or_generation() -> None:
    citation = Citation(
        reference="S1",
        filename="fruit.txt",
        page=1,
        snippet="Apples are fruits.",
        similarity=0.9,
        document_id="source-id",
        ingested_at="2026-09-30T00:00:00+00:00",
    )
    semantic_cache = FakeSemanticCache(
        CachedAnswer(
            query="what are apples",
            answer="Apples are fruits [S1].",
            citations=(citation,),
            similarity=0.98,
        )
    )
    workflow, client = make_workflow(
        StorageError("retrieval must not run for a cache hit"),
        semantic_cache=semantic_cache,
        cache_enabled=True,
    )

    result = workflow.run("what are apples?")

    assert semantic_cache.lookups == ["what are apples?"]
    assert result.status == "cache_hit"
    assert result.cache_hit
    assert result.answer == "Apples are fruits [S1]."
    assert result.citations == (citation,)
    assert client.calls == []


def test_workflow_stores_successful_answers_after_generation() -> None:
    semantic_cache = FakeSemanticCache()
    workflow, _ = make_workflow(
        RetrievalResponse(query="question", results=(make_source(),), status="ok"),
        semantic_cache=semantic_cache,
        cache_enabled=True,
    )

    result = workflow.run("question")

    assert result.status == "answered"
    assert len(semantic_cache.stores) == 1
    assert semantic_cache.stores[0][0:2] == ("question", result.answer)
    assert semantic_cache.stores[0][2] == result.citations


def test_workflow_does_not_store_failed_or_no_evidence_answers() -> None:
    cache_on_empty = FakeSemanticCache()
    no_evidence, _ = make_workflow(
        RetrievalResponse(query="question", results=(), status="empty_collection"),
        semantic_cache=cache_on_empty,
        cache_enabled=True,
    )
    no_evidence.run("question")
    assert cache_on_empty.stores == []

    cache_on_error = FakeSemanticCache()
    generation_failure, _ = make_workflow(
        RetrievalResponse(query="question", results=(make_source(),), status="ok"),
        chat_client=FakeChatClient(error=httpx.ConnectError("Ollama unavailable")),
        semantic_cache=cache_on_error,
        cache_enabled=True,
    )
    assert generation_failure.run("question").status == "error"
    assert cache_on_error.stores == []


def test_cache_write_failure_preserves_generated_answer_and_surfaces_warning() -> None:
    semantic_cache = FakeSemanticCache(store_error=CacheError("cache disk error"))
    workflow, _ = make_workflow(
        RetrievalResponse(query="question", results=(make_source(),), status="ok"),
        semantic_cache=semantic_cache,
        cache_enabled=True,
    )

    result = workflow.run("question")

    assert result.status == "answered"
    assert result.answer == "Apples are fruits [S1]."
    assert result.cache_warning == "cache disk error"
