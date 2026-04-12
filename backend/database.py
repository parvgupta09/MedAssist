# backend/database.py
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from backend.config import DATABASE_URL

print(f"[INFO] Database URL configured (first 80 chars): {DATABASE_URL[:80]}...")

# Optimize connection pool for Neon (cloud DB with connection limits)
# Neon has aggressive connection timeout and limited connections per account
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_pre_ping=True,
    pool_recycle=60,  # Recycle connections every 60 seconds for Neon stability
    pool_size=3,  # Minimal pool for Neon's connection limits
    max_overflow=5,  # Allow limited overflow for surge traffic
    connect_args={
        "connect_timeout": 10,  # 10 second timeout for Neon connections
        "application_name": "medassist_backend",  # Help track connections in Neon console
    },
)

# Log pool events for debugging
@event.listens_for(engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    print(f"[INFO] Database connection established")

@event.listens_for(engine, "close")
def receive_close(dbapi_conn, connection_record):
    print(f"[INFO] Database connection closed")

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

def get_db():
    """FastAPI dependency — yields a DB session and closes it after."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# File Summary:
# Initializes SQLAlchemy engine/session/base using DATABASE_URL.
# Provides get_db dependency for request-scoped database sessions.