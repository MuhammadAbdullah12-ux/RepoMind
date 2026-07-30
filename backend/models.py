from datetime import datetime
from typing import Optional, Literal
from sqlmodel import SQLModel, Field, Relationship

# ==========================================
# Day 4-5 Task 2: Define SQLModel Schemas
# ==========================================

class DocumentRecord(SQLModel, table=True):
    """
    Source of truth metadata record for an ingested repository document (README, Issue, or PR).
    Stores content_hash to prevent redundant re-chunking and re-embedding.
    """
    doc_id: str = Field(primary_key=True, index=True)
    doc_type: str  # "readme", "issue", or "pr"
    repo: str = Field(index=True)
    title: str
    created_at: datetime
    updated_at: datetime
    url: str
    content_hash: str = Field(index=True)  # SHA-256 hash string of raw document text

class ChunkRecord(SQLModel, table=True):
    """
    Represents an individual text chunk derived from a DocumentRecord.
    Maintains a foreign key to DocumentRecord and a qdrant_point_id mapping to Qdrant.
    """
    chunk_id: str = Field(primary_key=True, index=True)
    doc_id: str = Field(foreign_key="documentrecord.doc_id", index=True)
    position: int  # 0, 1, 2... sequence index within parent document
    text: str
    qdrant_point_id: int  # Integer mapping ID to Qdrant point

class SyncLogRecord(SQLModel, table=True):
    """
    Tracks sync operation execution logs per repository.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    repo: str = Field(index=True)
    last_synced_at: datetime = Field(default_factory=datetime.utcnow)
    docs_added: int = 0
    docs_updated: int = 0
    docs_deleted: int = 0
