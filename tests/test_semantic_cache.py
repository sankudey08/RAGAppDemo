from datetime import datetime, timezone
from pathlib import Path

import pytest

from rag_app.answering import Citation
from rag_app.config import Settings
from rag_app.ingestion import ExtractedDocument, ExtractedPage, SourceMetadata
from rag_app.semantic_cache import SemanticAnswerCache
from rag_app.vector_store import DocumentVectorStore


def cache_embeddings(texts: list[str]) -> list[list[float]]:
    vectors = {
        "query alpha": [1.0, 0.0],
        "question about alpha": [0.995, 0.1],
        "different question": [0.0, 1.0],
        "the alpha fact": [0.8, 0.6],
    }
    return [vectors.get(text, [0.8, 0.6]) for text in texts]


def make_document(filename: str, text: str) -> ExtractedDocument:
    ingested_at = datetime.now(timezone.utc)
    return ExtractedDocument(
        filename=filename,
        ingested_at=ingested_at,
        pages=(
            ExtractedPage(
                text=text,
                source=SourceMetadata(
                    filename=filename,
                    page=1,
                    ingested_at=ingested_at,
                    media_type="txt",
                ),
            ),
        ),
    )


def setup_cache(
    tmp_path: Path,
) -> tuple[Settings, DocumentVectorStore, SemanticAnswerCache, str, str]:
    settings = Settings(
        _env_file=None,
        chroma_persist_directory=tmp_path / "chroma",
        cache_enabled=True,
        cache_similarity_threshold=0.9,
        chunk_size=100,
        chunk_overlap=0,
    )
    vector_store = DocumentVectorStore(settings, embedder=cache_embeddings)
    stored = vector_store.index_document(make_document("alpha.txt", "the alpha fact"))
    return (
        settings,
        vector_store,
        SemanticAnswerCache(settings, vector_store),
        stored.document_id,
        stored.ingested_at,
    )


def make_citation(document_id: str, ingested_at: str) -> Citation:
    return Citation(
        reference="S1",
        filename="alpha.txt",
        page=1,
        snippet="the alpha fact",
        similarity=0.91,
        document_id=document_id,
        ingested_at=ingested_at,
    )


def test_semantic_cache_returns_similar_answer_and_citations(tmp_path: Path) -> None:
    settings, vector_store, cache, document_id, ingested_at = setup_cache(tmp_path)
    cache.store("query alpha", "The alpha fact [S1].", (make_citation(document_id, ingested_at),))

    reopened_store = DocumentVectorStore(settings, embedder=cache_embeddings)
    reopened_cache = SemanticAnswerCache(settings, reopened_store)
    result = reopened_cache.lookup("question about alpha")

    assert result is not None
    assert result.query == "query alpha"
    assert result.answer == "The alpha fact [S1]."
    assert result.similarity >= 0.9
    assert result.citations[0].document_id == document_id
    assert result.citations[0].filename == "alpha.txt"


def test_cache_refuses_to_store_answer_after_source_changes_mid_generation(tmp_path: Path) -> None:
    _, vector_store, cache, document_id, ingested_at = setup_cache(tmp_path)
    citation = make_citation(document_id, ingested_at)
    vector_store.index_document(make_document("alpha.txt", "updated alpha fact"))

    with pytest.raises(ValueError, match="changed during answer generation"):
        cache.store("query alpha", "Stale answer [S1].", (citation,))
    assert cache._collection.count() == 0


def test_semantic_cache_misses_below_similarity_threshold(tmp_path: Path) -> None:
    _, _, cache, document_id, ingested_at = setup_cache(tmp_path)
    cache.store("query alpha", "The alpha fact [S1].", (make_citation(document_id, ingested_at),))

    assert cache.lookup("different question") is None


def test_cache_entry_invalidated_after_source_reingestion(tmp_path: Path) -> None:
    settings, vector_store, cache, document_id, ingested_at = setup_cache(tmp_path)
    cache.store("query alpha", "The alpha fact [S1].", (make_citation(document_id, ingested_at),))
    vector_store.index_document(make_document("alpha.txt", "updated alpha fact"))

    assert cache.lookup("query alpha") is None
    assert cache._collection.count() == 0


def test_cache_entry_invalidated_after_source_removal(tmp_path: Path) -> None:
    _, vector_store, cache, document_id, ingested_at = setup_cache(tmp_path)
    cache.store("query alpha", "The alpha fact [S1].", (make_citation(document_id, ingested_at),))
    vector_store.delete_document(document_id)

    assert cache.lookup("query alpha") is None
    assert cache._collection.count() == 0


def test_cache_entry_invalidated_when_chat_model_changes(tmp_path: Path) -> None:
    settings, vector_store, cache, document_id, ingested_at = setup_cache(tmp_path)
    cache.store("query alpha", "The alpha fact [S1].", (make_citation(document_id, ingested_at),))

    changed_settings = settings.model_copy(update={"ollama_chat_model": "different-model"})
    changed_cache = SemanticAnswerCache(changed_settings, vector_store)

    assert changed_cache.lookup("query alpha") is None


def test_cache_rejects_answer_without_source_provenance(tmp_path: Path) -> None:
    _, _, cache, _, _ = setup_cache(tmp_path)
    citation = make_citation("", "")

    with pytest.raises(ValueError, match="missing document provenance"):
        cache.store("query alpha", "answer", (citation,))
