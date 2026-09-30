"""Startup checks and user-facing configuration errors."""

import logging

from .config import Settings

logger = logging.getLogger(__name__)


def prepare_runtime(settings: Settings) -> None:
    """Create local runtime directories and log the selected configuration."""

    settings.chroma_persist_directory.mkdir(parents=True, exist_ok=True)
    logger.info("Using Ollama at %s", settings.ollama_base_url)
    logger.info("Chat model: %s; embedding model: %s", settings.ollama_chat_model, settings.ollama_embedding_model)
    logger.info("ChromaDB persistence directory: %s", settings.chroma_persist_directory)
