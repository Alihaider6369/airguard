"""
Application configuration.

All settings are read from environment variables (or a local ".env" file),
so the same code works on your laptop (SQLite) and on a server (PostgreSQL)
without any code changes.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder that contains "app/" and "scripts/".
BASE_DIR = Path(__file__).resolve().parent.parent

# Load variables from ".env" if the file exists (silently ignored otherwise).
load_dotenv(BASE_DIR / ".env")


class Settings:
    """Simple container for configuration values."""

    # Default: a SQLite file stored in the project root.
    # Using an absolute path means it works no matter where you run Python from.
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'airguard.db'}"
    )

    # When True, SQLAlchemy logs every SQL statement to the console.
    SQL_ECHO: bool = os.getenv("SQL_ECHO", "false").lower() == "true"

    # Which websites may call the API from a browser (CORS).
    # "*" = anyone (fine for development). In production, set this to your
    # frontend's address, e.g. CORS_ORIGINS=https://airguard.vercel.app
    CORS_ORIGINS: list = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]


# One shared instance imported by the rest of the app.
settings = Settings()
