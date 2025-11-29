from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv


def load_dotenv_if_present(path: str | Path = ".env") -> None:
    """Load environment variables from a .env file if it exists."""
    dotenv_path = Path(path)
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)


def configure_logging(default_level: str = "INFO") -> None:
    """Configure root logging level from LOG_LEVEL env (default INFO)."""
    level_name = os.getenv("LOG_LEVEL", default_level).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level)
