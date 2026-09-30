"""Command-line entry point for validating and preparing the application."""

import logging
import sys

from pydantic import ValidationError

from .config import Settings
from .logging_config import configure_logging
from .startup import prepare_runtime


def main() -> int:
    """Validate configuration and prepare the local runtime."""

    try:
        settings = Settings()
        configure_logging(settings)
        prepare_runtime(settings)
    except ValidationError as error:
        print(f"Configuration error:\n{error}", file=sys.stderr)
        return 2
    except OSError as error:
        logging.getLogger(__name__).error("Unable to prepare application runtime: %s", error)
        return 1

    print("RAG application configuration is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
