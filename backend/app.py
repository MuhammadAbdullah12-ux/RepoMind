import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import hashlib
import json
from contextlib import asynccontextmanager

from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, select

from backend.database import init_db, get_session
from backend.models import DocumentRecord, ChunkRecord, SyncLogRecord
from ingestion.pipeline import SyncPipeline
from ingestion.ask_pipeline import AskPipeline
from pydantic import BaseModel


# Lifespan context manager to handle DB initialization on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield
    from ingestion.qdrant_store import close_global_client
    close_global_client()

app = FastAPI(
    title="RepoMind RAG Backend",
    description="Relational Metadata & RAG Retrieval API Engine",
    version="0.1.0",
    lifespan=lifespan
)

# Ensure the static files directory exists and mount it
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def compute_content_hash(text: str) -> str:
    """
    Computes SHA-256 hash string for raw text content.
    Used for content mutation detection.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

@app.get("/")
def read_root():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return FileResponse("backend/static/index.html")


@app.post("/sync")
def sync_full_pipeline(
    repo: str = Query(default="fastapi/fastapi", description="GitHub Repository full name"),
    force: bool = Query(default=False, description="Force re-chunking and re-embedding")
):
    """
    Triggers the master RAG ingestion pipeline for a repository.
    Applies SHA-256 diffing, text chunking, embedding, Qdrant upserts, and SQLModel metadata updates.
    """
    # Auto-fix if user pasted a full GitHub URL
    repo = repo.replace("https://github.com/", "").replace("http://github.com/", "").strip("/")
    
    pipeline = SyncPipeline()
    try:
        summary = pipeline.sync_repository(repo=repo, force=force)
        return summary
    except Exception as e:
        print(f"[ERROR] Sync pipeline execution failed: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
    finally:
        pipeline.close()


@app.post("/repository")
def sync_repository_metadata(
    repo: str = Query(default="fastapi/fastapi", description="GitHub Repository full name"),
    session: Session = Depends(get_session)
):
    """
    Seed/Sync metadata for raw ingested files into the database.
    Calculates SHA-256 content_hash for mutation detection.
    """
    readme_path = "data/raw/readme.json"
    issues_path = "data/raw/issues.json"
    
    docs_added = 0
    docs_updated = 0
    
    # Process README if present
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_data = json.load(f)
            
        doc_id = f"readme-{repo}"
        content = readme_data.get("content", "")
        c_hash = compute_content_hash(content)
        
        existing_doc = session.get(DocumentRecord, doc_id)
        if not existing_doc:
            new_doc = DocumentRecord(
                doc_id=doc_id,
                doc_type="readme",
                repo=repo,
                title="README.md",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                url=f"https://github.com/{repo}/blob/main/README.md",
                content_hash=c_hash
            )
            session.add(new_doc)
            docs_added += 1
        elif existing_doc.content_hash != c_hash:
            existing_doc.content_hash = c_hash
            existing_doc.updated_at = datetime.utcnow()
            session.add(existing_doc)
            docs_updated += 1

    # Process Issues if present
    if os.path.exists(issues_path):
        with open(issues_path, "r", encoding="utf-8") as f:
            issues_data = json.load(f)
            
        for item in issues_data:
            number = item.get("number")
            is_pr = "pull_request" in item or item.get("html_url", "").find("/pull/") != -1
            doc_type = "pr" if is_pr else "issue"
            doc_id = f"{doc_type}-{number}"
            
            title = item.get("title", "")
            body = item.get("body", "") or ""
            content = f"Title: {title}\nDescription: {body}"
            c_hash = compute_content_hash(content)
            
            created_at_str = item.get("created_at")
            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00")) if created_at_str else datetime.utcnow()
            
            updated_at_str = item.get("updated_at")
            updated_at = datetime.fromisoformat(updated_at_str.replace("Z", "+00:00")) if updated_at_str else datetime.utcnow()
            
            existing_doc = session.get(DocumentRecord, doc_id)
            if not existing_doc:
                new_doc = DocumentRecord(
                    doc_id=doc_id,
                    doc_type=doc_type,
                    repo=repo,
                    title=title,
                    created_at=created_at,
                    updated_at=updated_at,
                    url=item.get("html_url", ""),
                    content_hash=c_hash
                )
                session.add(new_doc)
                docs_added += 1
            elif existing_doc.content_hash != c_hash:
                existing_doc.content_hash = c_hash
                existing_doc.updated_at = updated_at
                session.add(existing_doc)
                docs_updated += 1
                
    # Create sync log entry
    sync_log = SyncLogRecord(
        repo=repo,
        last_synced_at=datetime.utcnow(),
        docs_added=docs_added,
        docs_updated=docs_updated
    )
    session.add(sync_log)
    session.commit()
    
    return {
        "status": "success",
        "repo": repo,
        "docs_added": docs_added,
        "docs_updated": docs_updated,
        "timestamp": sync_log.last_synced_at
    }

@app.get("/documents")
def get_documents(
    repo: str = Query(default="fastapi/fastapi"),
    session: Session = Depends(get_session)
):
    """
    Retrieves all ingested metadata documents for a given repository.
    """
    statement = select(DocumentRecord).where(DocumentRecord.repo == repo)
    documents = session.exec(statement).all()
    return {"repo": repo, "count": len(documents), "documents": documents}

class AskRequest(BaseModel):
    query: str
    repo: str | None = None

@app.post("/ask")
def ask_question(request: AskRequest):
    """
    Queries the repository using the end-to-end RAG pipeline (search -> rerank -> generate).
    """
    repo_name = request.repo
    if repo_name:
        repo_name = repo_name.replace("https://github.com/", "").replace("http://github.com/", "").strip("/")

    try:
        pipeline = AskPipeline()
        response = pipeline.ask(request.query, repo=repo_name)
        return response
    except Exception as e:
        print(f"[ERROR] Ask pipeline error: {e}")
        return {
            "answer": f"Retrieval error: {str(e)}",
            "cited_chunk_ids": [],
            "source_chunks": []
        }
    finally:
        try:
            pipeline.close()
        except Exception:
            pass


