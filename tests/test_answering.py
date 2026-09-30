from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from rag_app.answering import (
    NO_EVIDENCE_ANSWER,
    GenerationError,
    OllamaAnswerGenerator,
    build_grounded_messages,
)
from rag_app.config import Settings
from rag_app.vector_store import RetrievalResponse, RetrievalResult


def source(
    text: str = "An apple is a fruit.",
    *,
    filename: str = "food.pdf",
    page: int | None = 3,
) -> RetrievalResult:
    return RetrievalResult(
        text=text,
        distance=0.1,
        similarity=0.9,
        filename=filename,
        page=page,
        media_type="pdf",
        ingested_at=datetime.now(timezone.utc).isoformat(),
        chunk_index=1,
    )


class FakeStore:
    def __init__(self, response: RetrievalResponse) -> None:
        self.response = response
        self.queries: list[str] = []

    def search(self, query: str) -> RetrievalResponse:
        self.queries.append(query)
        return self.response


class FakeChatClient:
    def __init__(self, content: str = "An apple is a fruit [S1].", error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(message=SimpleNamespace(content=self.content))


def make_retrieval(*results: RetrievalResult, status: str = "ok") -> RetrievalResponse:
    return RetrievalResponse(query="What is an apple?", results=results, status=status)


def test_prompt_labels_sources_and_requires_evidence_based_citations() -> None:
    messages = build_grounded_messages("What is an apple?", [source()])

    assert messages[0]["role"] == "system"
    assert "only the supplied source excerpts" in messages[0]["content"]
    assert "Cite each factual claim" in messages[0]["content"]
    assert "[S1] food.pdf, page 3" in messages[1]["content"]
    assert "An apple is a fruit." in messages[1]["content"]


def test_answer_returns_generated_text_and_attributed_citations() -> None:
    store = FakeStore(make_retrieval(source()))
    client = FakeChatClient()
    generator = OllamaAnswerGenerator(
        Settings(_env_file=None, ollama_chat_model="test-chat"),
        store,  # type: ignore[arg-type]
        chat_client=client,  # type: ignore[arg-type]
    )

    answer = generator.answer(" What is an apple? ")

    assert answer.status == "answered"
    assert answer.question == "What is an apple?"
    assert answer.answer == "An apple is a fruit [S1]."
    assert answer.citations[0].reference == "S1"
    assert answer.citations[0].filename == "food.pdf"
    assert answer.citations[0].page == 3
    assert answer.citations[0].snippet == "An apple is a fruit."
    assert store.queries == ["What is an apple?"]
    assert client.calls[0]["model"] == "test-chat"
    assert client.calls[0]["stream"] is False


@pytest.mark.parametrize("status", ["empty_collection", "no_relevant_results"])
def test_no_retrieved_evidence_returns_clear_message_without_chat(status: str) -> None:
    store = FakeStore(make_retrieval(status=status))
    client = FakeChatClient()
    generator = OllamaAnswerGenerator(
        Settings(_env_file=None),
        store,  # type: ignore[arg-type]
        chat_client=client,  # type: ignore[arg-type]
    )

    answer = generator.answer("What is an apple?")

    assert answer.status == status
    assert answer.answer == NO_EVIDENCE_ANSWER
    assert answer.citations == ()
    assert client.calls == []


def test_citations_include_source_without_page_number_and_bound_snippet() -> None:
    long_source = source("evidence " * 80, filename="notes.txt", page=None)
    store = FakeStore(make_retrieval(long_source))
    generator = OllamaAnswerGenerator(
        Settings(_env_file=None),
        store,  # type: ignore[arg-type]
        chat_client=FakeChatClient(),  # type: ignore[arg-type]
    )

    answer = generator.answer("Find evidence")

    assert answer.citations[0].page is None
    assert answer.citations[0].filename == "notes.txt"
    assert len(answer.citations[0].snippet) <= 283
    assert answer.citations[0].snippet.endswith("...")


@pytest.mark.parametrize(
    ("client", "message"),
    [
        (FakeChatClient(content="  "), "empty answer"),
        (FakeChatClient(error=httpx.ConnectError("server down")), "server down"),
    ],
)
def test_generation_errors_are_explicit(client: FakeChatClient, message: str) -> None:
    generator = OllamaAnswerGenerator(
        Settings(_env_file=None),
        FakeStore(make_retrieval(source())),  # type: ignore[arg-type]
        chat_client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(GenerationError, match=message):
        generator.answer("What is an apple?")


def test_answer_rejects_blank_question() -> None:
    generator = OllamaAnswerGenerator(
        Settings(_env_file=None),
        FakeStore(make_retrieval()),  # type: ignore[arg-type]
        chat_client=FakeChatClient(),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="Question cannot be empty"):
        generator.answer("  ")
