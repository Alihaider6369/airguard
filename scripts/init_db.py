"""
Create all database tables.

Run from the project root:
    python -m scripts.init_db

Safe to run more than once: existing tables are left untouched.
"""

from sqlalchemy import inspect

from app import models  # noqa: F401  (importing registers the tables on Base)
from app.database import Base, engine


def main() -> None:
    # Creates any table that doesn't exist yet. Never drops or alters data.
    Base.metadata.create_all(bind=engine)

    # Print what exists so you can confirm it worked.
    tables = inspect(engine).get_table_names()
    print(f"Database: {engine.url}")
    print(f"Tables ready: {', '.join(sorted(tables))}")


if __name__ == "__main__":
    main()
