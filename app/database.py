"""
Database connection setup (SQLAlchemy engine, session factory, Base class).

Other modules import from here:
    - Base          -> parent class for all ORM models (see models.py)
    - SessionLocal  -> creates database sessions (used in scripts)
    - get_db        -> FastAPI dependency that yields a session per request
"""

from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

# SQLite needs one extra option because FastAPI may use the connection
# from different threads. Other databases (PostgreSQL) don't need it.
_is_sqlite = settings.DATABASE_URL.startswith("sqlite")
_connect_args = {"check_same_thread": False} if _is_sqlite else {}

# The engine is the actual connection "pool" to the database.
engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.SQL_ECHO,
    connect_args=_connect_args,
)

if _is_sqlite:
    # SQLite ignores FOREIGN KEY constraints unless this is switched on
    # for every new connection. Without it, broken references go unnoticed.
    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


# A "session" is one unit of work (a set of queries + a commit/rollback).
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """Parent class for every database model."""


def get_db() -> Iterator[Session]:
    """
    FastAPI dependency: gives each request its own session
    and always closes it afterwards.

    Usage in a route:
        def my_route(db: Session = Depends(get_db)): ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
