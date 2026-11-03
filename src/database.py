from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker
from config import settings

connect_args = {"check_same_thread": False}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args["timeout"] = 5

engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)

# P0: SQLite WAL Mode and Performance Tuning
if settings.DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute("PRAGMA busy_timeout=5000;")
        cursor.execute("PRAGMA cache_size=-10000;")
        cursor.execute("PRAGMA mmap_size=268435456;")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency injection helper to yield a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
