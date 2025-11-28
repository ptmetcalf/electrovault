from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_dotenv_if_present(path: str | Path = ".env") -> None:
    """Load environment variables from a .env file if it exists."""
    dotenv_path = Path(path)
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)
