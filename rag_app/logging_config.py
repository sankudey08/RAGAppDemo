"""Logging setup shared by the CLI and Streamlit entry points."""

import logging

from .config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure application logging without changing third-party loggers."""

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
