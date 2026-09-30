from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_app.config import Settings
from rag_app.ingestion import ExtractedDocument, ExtractedPage, SourceMetadata, extract_document
from rag_app.vector_store import (
    DocumentVectorStore,
    EmbeddingError,
    StorageError,
    chunk_document,
    split_text,
)


def deterministic_embeddings(texts: list[str]) -> list[list[float]]:
    return [[float(len(text)), float(sum(text.encode("utf-8")) % 997), 1.0] for text in texts]


def semantic_test_embeddings(texts: list[str]) -> list[list[float]]:
    vectors = {
        "fruit query": [1.0, 0.0],
        "apple is a fruit": [1.0, 0.0],
        "banana is a fruit": [0.98, 0.2],
        "car engine repair": [0.0, 1.0],
        "unrelated query": [0.0, 1.0],
    }
    return [vectors[text] for text in texts]


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
                    page=2,
                    ingested_at=ingested_at,
                    media_type="txt",
                ),
            ),
        ),
    )


def make_settings(path: Path, **kwargs: object) -> Settings:
    return Settings(
        _env_file=None,
        chroma_persist_directory=path,
        chunk_size=kwargs.pop("chunk_size", 100),
        chunk_overlap=kwargs.pop("chunk_overlap", 20),
        **kwargs,
    )


def test_split_text_creates_overlapping_chunks_without_gaps() -> None:
    text = " ".join(f"word{index}" for index in range(50))

    chunks = split_text(text, chunk_size=40, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(len(chunk) <= 40 for chunk in chunks)
    original_words = set(text.split())
    chunk_words = {word for chunk in chunks for word in chunk.split()}
    assert original_words == chunk_words
    assert chunks[0].startswith("word0")
    assert chunks[-1].endswith("word49")


def test_chunk_document_retains_page_provenance() -> None:
    document = make_document("source.txt", "word " * 80)

    chunks = chunk_document(document, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk.filename == "source.txt" and chunk.page == 2 for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_index_survives_store_recreation(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "chroma")
    store = DocumentVectorStore(settings, embedder=deterministic_embeddings)
    stored = store.index_document(make_document("source.txt", "important source text " * 20))

    reopened = DocumentVectorStore(settings, embedder=deterministic_embeddings)
    documents = reopened.list_documents()

    assert len(documents) == 1
    assert documents[0].document_id == stored.document_id
    assert documents[0].filename == "source.txt"
    assert documents[0].chunk_count == stored.chunk_count
    result = reopened._collection.get(where={"document_id": stored.document_id}, include=["documents", "metadatas"])
    assert len(result["ids"]) == stored.chunk_count
    assert all(metadata["page"] == 2 for metadata in result["metadatas"])
    assert all("important source text" in text for text in result["documents"])


def test_reingestion_replaces_old_chunks_and_is_case_insensitive(tmp_path: Path) -> None:
    store = DocumentVectorStore(make_settings(tmp_path / "chroma"), embedder=deterministic_embeddings)
    first = store.index_document(make_document("Notes.txt", "long text " * 100))
    assert first.chunk_count > 1

    replacement = store.index_document(make_document("notes.TXT", "replacement"))

    assert replacement.document_id == first.document_id
    assert replacement.chunk_count == 1
    result = store._collection.get(where={"document_id": first.document_id}, include=["documents"])
    assert result["documents"] == ["replacement"]
    assert store.list_documents()[0].chunk_count == 1


def test_embedding_failure_does_not_replace_existing_document(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "chroma")
    store = DocumentVectorStore(settings, embedder=deterministic_embeddings)
    stored = store.index_document(make_document("source.txt", "prior version"))

    def failing_embedder(texts: list[str]) -> list[list[float]]:
        raise EmbeddingError("Ollama is unavailable")

    failing_store = DocumentVectorStore(settings, embedder=failing_embedder)
    with pytest.raises(EmbeddingError, match="Ollama is unavailable"):
        failing_store.index_document(make_document("source.txt", "replacement version"))

    result = store._collection.get(where={"document_id": stored.document_id}, include=["documents"])
    assert result["documents"] == ["prior version"]


def test_embedding_provider_must_return_one_vector_per_chunk(tmp_path: Path) -> None:
    store = DocumentVectorStore(make_settings(tmp_path / "chroma"), embedder=lambda texts: [])

    with pytest.raises(EmbeddingError, match="0 vectors"):
        store.index_document(make_document("source.txt", "some text"))


def test_delete_document_removes_all_chunks_and_is_idempotent(tmp_path: Path) -> None:
    store = DocumentVectorStore(make_settings(tmp_path / "chroma"), embedder=deterministic_embeddings)
    stored = store.index_document(make_document("source.txt", "long text " * 100))

    assert store.delete_document(stored.document_id)
    assert store.list_documents() == []
    assert not store.delete_document(stored.document_id)


def test_duplicate_contents_with_different_names_are_separate_documents(tmp_path: Path) -> None:
    store = DocumentVectorStore(make_settings(tmp_path / "chroma"), embedder=deterministic_embeddings)
    store.index_document(make_document("first.txt", "identical content"))
    store.index_document(make_document("second.txt", "identical content"))

    assert {document.filename for document in store.list_documents()} == {"first.txt", "second.txt"}


def test_chunking_empty_text_returns_no_chunks() -> None:
    document = extract_document("empty-near.txt", b" text ")
    empty_document = ExtractedDocument(
        filename=document.filename,
        ingested_at=document.ingested_at,
        pages=(
            ExtractedPage(
                text=" ",
                source=document.pages[0].source,
            ),
        ),
    )

    assert chunk_document(empty_document) == []


def test_settings_validate_chroma_collection_name() -> None:
    with pytest.raises(ValidationError):
        Settings(chroma_collection_name="invalid collection name")


def test_split_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        split_text("some text", chunk_size=10, chunk_overlap=10)


def test_search_returns_top_k_similarity_and_provenance(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path / "chroma",
        chunk_size=100,
        chunk_overlap=0,
        retrieval_top_k=2,
        retrieval_min_score=0.5,
    )
    store = DocumentVectorStore(settings, embedder=semantic_test_embeddings)
    store.index_document(make_document("apple.txt", "apple is a fruit"))
    store.index_document(make_document("banana.txt", "banana is a fruit"))
    store.index_document(make_document("car.txt", "car engine repair"))

    response = store.search("fruit query")

    assert response.status == "ok"
    assert [result.filename for result in response.results] == ["apple.txt", "banana.txt"]
    assert response.results[0].similarity == pytest.approx(1.0)
    assert 0 <= response.results[1].similarity < response.results[0].similarity
    assert response.results[0].distance == pytest.approx(0.0)
    assert response.results[0].page == 2
    assert response.results[0].chunk_index == 0


def test_search_empty_collection_is_explicit(tmp_path: Path) -> None:
    store = DocumentVectorStore(make_settings(tmp_path / "chroma"), embedder=semantic_test_embeddings)

    response = store.search("fruit query")

    assert response.status == "empty_collection"
    assert response.results == ()


def test_search_below_threshold_has_explicit_empty_results(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path / "chroma",
        chunk_size=100,
        chunk_overlap=0,
        retrieval_min_score=0.99,
    )
    store = DocumentVectorStore(settings, embedder=semantic_test_embeddings)
    store.index_document(make_document("car.txt", "car engine repair"))

    response = store.search("fruit query")

    assert response.status == "no_relevant_results"
    assert response.results == ()


def test_search_supports_existing_default_l2_collection(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "chroma", retrieval_min_score=0)
    store = DocumentVectorStore(settings, embedder=semantic_test_embeddings)
    assert store._distance_space == "cosine"

    # Existing Phase 3 collections have no explicit space metadata and therefore use Chroma's L2 default.
    import chromadb

    legacy_client = chromadb.PersistentClient(path=str(tmp_path / "legacy"))
    legacy_collection = legacy_client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"description": "Legacy collection"},
    )
    legacy_collection.upsert(
        ids=["legacy-chunk"],
        embeddings=[[1.0, 0.0]],
        documents=["apple is a fruit"],
        metadatas=[
            {
                "document_id": "legacy-document",
                "filename": "legacy.txt",
                "media_type": "txt",
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "chunk_index": 0,
                "page": 2,
            }
        ],
    )
    legacy_store = DocumentVectorStore(
        Settings(
            _env_file=None,
            chroma_persist_directory=tmp_path / "legacy",
            chroma_collection_name=settings.chroma_collection_name,
            retrieval_min_score=0,
        ),
        embedder=semantic_test_embeddings,
    )

    response = legacy_store.search("fruit query")

    assert legacy_store._distance_space == "l2"
    assert response.status == "ok"
    assert response.results[0].filename == "legacy.txt"
    assert response.results[0].similarity == pytest.approx(1.0)


def test_search_validates_query_and_settings(tmp_path: Path) -> None:
    store = DocumentVectorStore(make_settings(tmp_path / "chroma"), embedder=semantic_test_embeddings)

    with pytest.raises(ValueError, match="cannot be empty"):
        store.search("  ")
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        store.search("fruit query", top_k=0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        store.search("fruit query", min_similarity=1.1)
