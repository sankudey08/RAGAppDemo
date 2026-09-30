from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_app.config import Settings


def test_default_settings_define_models_and_upload_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.ollama_chat_model
    assert settings.ollama_embedding_model == "nomic-embed-text"
    assert settings.max_upload_size_mb == 25
    assert settings.retrieval_top_k == 5


def test_settings_reject_invalid_retrieval_score() -> None:
    with pytest.raises(ValidationError):
        Settings(retrieval_min_score=1.1)


def test_settings_reject_overlap_larger_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap must be smaller"):
        Settings(chunk_size=100, chunk_overlap=100)


def test_settings_reject_file_persistence_path(tmp_path: Path) -> None:
    persistence_file = tmp_path / "chroma"
    persistence_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValidationError):
        Settings(chroma_persist_directory=persistence_file)
