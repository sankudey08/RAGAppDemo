"""LangGraph retrieval and grounded-answer workflow with token streaming."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import StreamWriter

from .answering import (
    NO_EVIDENCE_ANSWER,
    Citation,
    GenerationError,
    OllamaAnswerGenerator,
    _format_citations,
)
from .config import Settings
from .semantic_cache import CacheError, SemanticAnswerCache
from .vector_store import (
    DocumentVectorStore,
    EmbeddingError,
    RetrievalResult,
    StorageError,
)


class RAGState(TypedDict, total=False):
    """State passed between retrieval, conditional routing, and generation."""

    question: str
    retrieval_status: str
    sources: tuple[RetrievalResult, ...]
    answer: str
    citations: tuple[Citation, ...]
    status: str
    error: str
    cache_hit: bool
    cache_warning: str


@dataclass(frozen=True)
class WorkflowResult:
    """Final workflow state suitable for UI rendering or API responses."""

    question: str
    answer: str
    citations: tuple[Citation, ...]
    sources: tuple[RetrievalResult, ...]
    status: str
    error: str | None = None
    cache_hit: bool = False
    cache_warning: str | None = None


@dataclass(frozen=True)
class WorkflowEvent:
    """A streamed answer token or the final workflow result."""

    kind: str
    token: str = ""
    result: WorkflowResult | None = None


class RAGWorkflow:
    """Run retrieval then conditionally generate and stream a grounded answer."""

    def __init__(
        self,
        settings: Settings,
        vector_store: DocumentVectorStore,
        *,
        answer_generator: OllamaAnswerGenerator | None = None,
        semantic_cache: SemanticAnswerCache | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._answer_generator = answer_generator or OllamaAnswerGenerator(settings, vector_store)
        self._cache_enabled = settings.cache_enabled
        self._cache_initialization_error: str | None = None
        if semantic_cache is not None:
            self._semantic_cache = semantic_cache
        elif settings.cache_enabled:
            try:
                self._semantic_cache = SemanticAnswerCache(settings, vector_store)
            except CacheError as error:
                self._semantic_cache = None
                self._cache_initialization_error = str(error)
        else:
            self._semantic_cache = None

        graph = StateGraph(RAGState)
        graph.add_node("cache_lookup", self._cache_lookup)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate", self._generate)
        graph.add_node("no_evidence", self._no_evidence)
        graph.add_node("cache_store", self._cache_store)
        graph.set_entry_point("cache_lookup")
        graph.add_conditional_edges(
            "cache_lookup",
            self._route_after_cache,
            {
                "cached": END,
                "retrieve": "retrieve",
                "error": END,
            },
        )
        graph.add_conditional_edges(
            "retrieve",
            self._route_after_retrieval,
            {
                "generate": "generate",
                "no_evidence": "no_evidence",
                "error": END,
            },
        )
        graph.add_edge("generate", "cache_store")
        graph.add_edge("cache_store", END)
        graph.add_edge("no_evidence", END)
        self._graph = graph.compile()

    def stream(self, question: str) -> Iterator[WorkflowEvent]:
        """Yield token events followed by the complete answer and citations."""

        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question cannot be empty.")

        final_state: RAGState = {"question": normalized_question}
        for mode, payload in self._graph.stream(
            {"question": normalized_question},
            stream_mode=["custom", "updates"],
        ):
            if mode == "custom" and isinstance(payload, str):
                yield WorkflowEvent(kind="token", token=payload)
            elif mode == "updates" and isinstance(payload, dict):
                for node_state in payload.values():
                    if isinstance(node_state, dict):
                        final_state.update(node_state)

        yield WorkflowEvent(kind="complete", result=_result_from_state(final_state))

    def run(self, question: str) -> WorkflowResult:
        """Run the workflow to completion and return its final result."""

        result: WorkflowResult | None = None
        for event in self.stream(question):
            if event.kind == "complete":
                result = event.result
        if result is None:
            raise RuntimeError("RAG workflow completed without a final result.")
        return result

    def _retrieve(self, state: RAGState) -> RAGState:
        question = state["question"]
        try:
            retrieval = self._vector_store.search(question)
        except (EmbeddingError, StorageError, ValueError) as error:
            return {"status": "error", "error": str(error), "sources": (), "citations": ()}
        return {
            "retrieval_status": retrieval.status,
            "sources": retrieval.results,
            "citations": _format_citations(retrieval.results),
        }

    def _cache_lookup(self, state: RAGState) -> RAGState:
        if not self._cache_enabled:
            return {"cache_hit": False}
        if self._cache_initialization_error:
            return {
                "status": "error",
                "error": f"Could not initialize semantic cache: {self._cache_initialization_error}",
            }
        if self._semantic_cache is None:
            return {"status": "error", "error": "Semantic cache is enabled but unavailable."}

        try:
            cached = self._semantic_cache.lookup(state["question"])
        except (CacheError, ValueError) as error:
            return {"status": "error", "error": str(error), "cache_hit": False}
        if cached is None:
            return {"cache_hit": False}
        return {
            "answer": cached.answer,
            "citations": cached.citations,
            "sources": (),
            "status": "cache_hit",
            "cache_hit": True,
        }

    @staticmethod
    def _route_after_cache(state: RAGState) -> str:
        if state.get("status") == "error":
            return "error"
        return "cached" if state.get("cache_hit") else "retrieve"

    @staticmethod
    def _route_after_retrieval(state: RAGState) -> str:
        if state.get("status") == "error":
            return "error"
        if state.get("sources"):
            return "generate"
        return "no_evidence"

    def _generate(self, state: RAGState, writer: StreamWriter) -> RAGState:
        try:
            answer = self._answer_generator.generate_stream(
                state["question"],
                state["sources"],
                writer,
            )
        except GenerationError as error:
            return {"status": "error", "error": str(error)}
        return {"answer": answer, "status": "answered"}

    def _cache_store(self, state: RAGState) -> RAGState:
        if not self._cache_enabled or state.get("status") != "answered":
            return {}
        if self._semantic_cache is None:
            return {"cache_warning": "Answer was generated, but semantic cache is unavailable."}
        try:
            self._semantic_cache.store(state["question"], state["answer"], state["citations"])
        except (CacheError, ValueError) as error:
            return {"cache_warning": str(error)}
        return {}

    @staticmethod
    def _no_evidence(state: RAGState) -> RAGState:
        return {
            "answer": NO_EVIDENCE_ANSWER,
            "status": state.get("retrieval_status", "no_relevant_results"),
            "citations": (),
        }


def _result_from_state(state: RAGState) -> WorkflowResult:
    return WorkflowResult(
        question=state.get("question", ""),
        answer=state.get("answer", ""),
        citations=state.get("citations", ()),
        sources=state.get("sources", ()),
        status=state.get("status", "error"),
        error=state.get("error"),
        cache_hit=state.get("cache_hit", False),
        cache_warning=state.get("cache_warning"),
    )
