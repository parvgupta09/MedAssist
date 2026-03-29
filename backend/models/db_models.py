# backend/models/db_models.py
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.database import Base

class DiseaseMetadata(Base):
    """Stores full Pydantic-validated metadata for all 35 diseases."""
    __tablename__ = "disease_metadata"

    id          = Column(String,   primary_key=True)   # snake_case disease id
    name        = Column(String,   nullable=False, unique=True)
    body_system = Column(String,   nullable=False)
    category    = Column(String,   nullable=False)
    data        = Column(JSONB,    nullable=False)      # full metadata JSON
    created_at  = Column(DateTime, default=datetime.utcnow)


class ChatSession(Base):
    """One row per user conversation."""
    __tablename__ = "chat_sessions"

    id              = Column(String,   primary_key=True)  # UUID
    user_id         = Column(String,   nullable=False)    # FK to users table (friend's)
    started_at      = Column(DateTime, default=datetime.utcnow)
    ended_at        = Column(DateTime, nullable=True)
    status          = Column(String,   default="active")  # active / completed
    top_5_diseases  = Column(JSONB,    nullable=True)     # Stage 4 results
    final_diagnosis = Column(JSONB,    nullable=True)     # Stage 5 top 3 results
    user_summary    = Column(Text,     nullable=True)     # Stage 3 summary

    messages = relationship("ChatMessage", back_populates="session")


class ChatMessage(Base):
    """Every single message in every conversation."""
    __tablename__ = "chat_messages"

    id           = Column(Integer,  primary_key=True, autoincrement=True)
    session_id   = Column(String,   ForeignKey("chat_sessions.id"), nullable=False)
    role         = Column(String,   nullable=False)   # user / assistant
    message      = Column(Text,     nullable=False)
    stage        = Column(String,   nullable=True)    # stage_1/stage_2/stage_3/stage_4/stage_5
    timestamp    = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")