import os
import sys
import json
import hashlib
from datetime import datetime
from typing import Optional

from sqlmodel import Session, select
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from backend.database import init_db, engine
from backend.models import DocumentRecord, ChunkRecord, SyncLogRecord
from ingestion.embedder import TextEmbedder
from ingestion.splitter import split_readme, split_text_by_length
from github import Github, GithubException
from ingestion.fetcher import fetch_readme, fetch_issues


# Safe print helper for Windows consoles
def safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or 'utf-8'
        print(text.encode(encoding, errors='replace').decode(encoding))

class SyncPipeline:
    """
    Master Ingestion Orchestrator.
    Wires together raw document ingestion, SHA-256 mutation diffing,
    Markdown/length text splitters, TextEmbedder, Qdrant vector storage,
    and SQLModel relational metadata storage.
    """
    def __init__(
        self,
        qdrant_path: str = "data/qdrant_db",
        collection_name: str = "repomind_collection"
    ):
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        
        # 1. Ensure SQLModel tables exist
        init_db()
        
        # 2. Instantiate TextEmbedder
        self.embedder = TextEmbedder()
        
        # 3. Initialize local Qdrant engine
        from ingestion.qdrant_store import get_qdrant_client
        print(f"[RUNNING] Connecting to local Qdrant engine at '{self.qdrant_path}'...")
        self.qdrant = get_qdrant_client(self.qdrant_path)
        
        # 4. Setup collection if missing
        try:
            if not self.qdrant.collection_exists(self.collection_name):
                print(f"[RUNNING] Creating collection '{self.collection_name}' with 384 dimensions...")
                self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
        except Exception:
            pass
        print(f"[SUCCESS] Pipeline ready with Qdrant collection '{self.collection_name}'.")

    def close(self):
        # Global client is managed at the application process level, do not close here
        pass

    @staticmethod
    def compute_content_hash(text: str) -> str:
        """
        Computes SHA-256 hash string for raw text content.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def sync_repository(self, repo: str = "fastapi/fastapi", force: bool = False) -> dict:
        """
        Executes a full repository sync pass with differential update detection.
        """
        print(f"\n=========================================")
        print(f"Starting Master Sync Pipeline for: '{repo}'")
        print(f"=========================================")
        
        token = os.getenv("GITHUB_TOKEN")
        if token and token.strip():
            g = Github(token.strip())
        else:
            print("[WARNING] GITHUB_TOKEN is not set. Falling back to unauthenticated GitHub API access.")
            g = Github()

        # Connect to GitHub & fetch repository
        repo_obj = g.get_repo(repo)
        
        # Check GitHub API rate limits and back off if needed
        try:
            rate_limit = g.get_rate_limit()
            rate = rate_limit.rate
            print(f"[INFO] GitHub API Rate Limit: {rate.remaining}/{rate.limit} remaining (Resets at {rate.reset})")
            if rate.remaining < 5:
                import time
                reset_timestamp = rate.reset.timestamp()
                now_timestamp = datetime.now().timestamp()
                sleep_time = max(0.0, reset_timestamp - now_timestamp) + 5
                print(f"[WARNING] GitHub rate limit extremely low ({rate.remaining}). Sleeping for {sleep_time:.1f} seconds until reset...")
                time.sleep(sleep_time)
        except Exception as re:
            print(f"[WARNING] Could not retrieve GitHub rate limit info: {re}")

        docs_to_process = []
        
        # 1. Fetch README
        try:
            readme_doc = fetch_readme(repo_obj)
            docs_to_process.append({
                "doc_id": readme_doc.doc_id,
                "doc_type": readme_doc.doc_type,
                "title": readme_doc.title,
                "content": readme_doc.body,
                "url": readme_doc.url,
                "created_at": readme_doc.created_at,
                "updated_at": readme_doc.updated_at
            })
        except GithubException as ge:
            if ge.status == 403:
                print(f"[ERROR] GitHub rate limit exceeded or access denied when fetching README: {ge.data.get('message', '')}")
            else:
                print(f"[WARNING] GitHub error fetching README.md: {ge}")
        except Exception as e:
            print(f"[WARNING] Failed to fetch live README.md: {e}")

        # 2. Fetch Issues & PRs (limit to 5 on Vercel for fast execution within serverless timeouts)
        fetch_limit = 5 if os.getenv("VERCEL") else 30
        try:
            issue_docs = fetch_issues(repo_obj, limit=fetch_limit)
            for doc in issue_docs:
                docs_to_process.append({
                    "doc_id": doc.doc_id,
                    "doc_type": doc.doc_type,
                    "title": doc.title,
                    "content": doc.body,
                    "url": doc.url,
                    "created_at": doc.created_at,
                    "updated_at": doc.updated_at
                })
        except GithubException as ge:
            if ge.status == 403:
                print(f"[ERROR] GitHub rate limit exceeded or access denied when fetching issues/PRs: {ge.data.get('message', '')}")
            else:
                print(f"[WARNING] GitHub error fetching issues/PRs: {ge}")
        except Exception as e:
            print(f"[WARNING] Failed to fetch live issues/PRs: {e}")



        print(f"Loaded {len(docs_to_process)} raw documents to inspect.")
        
        docs_added = 0
        docs_updated = 0
        docs_skipped = 0
        total_chunks_created = 0
        
        with Session(engine) as session:
            for item in docs_to_process:
                doc_id = item["doc_id"]
                content = item["content"]
                c_hash = self.compute_content_hash(content)
                
                existing_doc = session.get(DocumentRecord, doc_id)
                
                # Check for unchanged document
                if existing_doc and existing_doc.content_hash == c_hash and not force:
                    docs_skipped += 1
                    continue
                    
                # Handle updated document (clean up stale chunks)
                if existing_doc and (existing_doc.content_hash != c_hash or force):
                    safe_print(f"  [MUTATION DETECTED] Updating document: {doc_id}")
                    # Find old chunk records
                    old_chunks = session.exec(select(ChunkRecord).where(ChunkRecord.doc_id == doc_id)).all()
                    if old_chunks:
                        point_ids = [c.qdrant_point_id for c in old_chunks]
                        try:
                            self.qdrant.delete(collection_name=self.collection_name, points_selector=point_ids)
                        except Exception:
                            pass
                        for c in old_chunks:
                            session.delete(c)
                    existing_doc.content_hash = c_hash
                    existing_doc.updated_at = item["updated_at"]
                    session.add(existing_doc)
                    docs_updated += 1
                else:
                    safe_print(f"  [NEW DOCUMENT] Adding document: {doc_id}")
                    new_doc = DocumentRecord(
                        doc_id=doc_id,
                        doc_type=item["doc_type"],
                        repo=repo,
                        title=item["title"],
                        created_at=item["created_at"],
                        updated_at=item["updated_at"],
                        url=item["url"],
                        content_hash=c_hash
                    )
                    session.add(new_doc)
                    docs_added += 1
                    
                # Chunk text
                if item["doc_type"] == "readme":
                    chunks_text = split_readme(content)
                else:
                    chunks_text = split_text_by_length(content)
                    
                if not chunks_text:
                    continue
                    
                # Embed chunks & build Qdrant points
                vectors = self.embedder.embed_chunks(chunks_text)
                points_buffer = []
                
                for pos, (chunk_str, vector) in enumerate(zip(chunks_text, vectors)):
                    chunk_id = f"{doc_id}-chunk-{pos}"
                    # Generate deterministic integer point ID for Qdrant
                    qdrant_point_id = abs(hash(chunk_id)) % (2**63 - 1)
                    
                    points_buffer.append(PointStruct(
                        id=qdrant_point_id,
                        vector=vector,
                        payload={
                            "doc_id": doc_id,
                            "chunk_id": chunk_id,
                            "doc_type": item["doc_type"],
                            "repo": repo,
                            "text": chunk_str,
                            "url": item["url"]
                        }
                    ))
                    
                    chunk_record = ChunkRecord(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        position=pos,
                        text=chunk_str,
                        qdrant_point_id=qdrant_point_id
                    )
                    session.add(chunk_record)
                    total_chunks_created += 1
                    
                # Upsert points to Qdrant
                if points_buffer:
                    self.qdrant.upsert(
                        collection_name=self.collection_name,
                        wait=True,
                        points=points_buffer
                    )
            # --- NEW DELETION LOGIC ---
            docs_deleted = 0
            fetched_doc_ids = {item["doc_id"] for item in docs_to_process}
            existing_docs = session.exec(select(DocumentRecord).where(DocumentRecord.repo == repo)).all()
            
            for doc in existing_docs:
                if doc.doc_id not in fetched_doc_ids:
                    safe_print(f"  [DELETED DOCUMENT] Removing obsolete document: {doc.doc_id}")
                    
                    # 1. Delete associated chunks from Qdrant
                    old_chunks = session.exec(select(ChunkRecord).where(ChunkRecord.doc_id == doc.doc_id)).all()
                    if old_chunks:
                        point_ids = [c.qdrant_point_id for c in old_chunks]
                        try:
                            self.qdrant.delete(collection_name=self.collection_name, points_selector=point_ids)
                        except Exception as e:
                            print(f"[WARNING] Failed to delete obsolete points from Qdrant: {e}")
                        
                        # 2. Delete ChunkRecords from SQLite
                        for c in old_chunks:
                            session.delete(c)
                            
                    # 3. Delete DocumentRecord from SQLite
                    session.delete(doc)
                    docs_deleted += 1
            # --------------------------

            # Record audit sync log
            sync_log = SyncLogRecord(
                repo=repo,
                last_synced_at=datetime.utcnow(),
                docs_added=docs_added,
                docs_updated=docs_updated,
                docs_deleted=docs_deleted
            )
            session.add(sync_log)
            session.commit()
            
        summary = {
            "status": "success",
            "repo": repo,
            "total_documents": len(docs_to_process),
            "docs_added": docs_added,
            "docs_updated": docs_updated,
            "docs_deleted": docs_deleted,
            "docs_skipped": docs_skipped,
            "chunks_created": total_chunks_created
        }
        
        print("\n-----------------------------------------")
        print("Master Ingestion Sync Completed Summary:")
        print(f"  - Total Documents Examined: {summary['total_documents']}")
        print(f"  - Documents Added:           {summary['docs_added']}")
        print(f"  - Documents Updated:         {summary['docs_updated']}")
        print(f"  - Documents Skipped:         {summary['docs_skipped']}")
        print(f"  - Documents Deleted:         {summary['docs_deleted']}")
        print(f"  - Chunks Created & Upserted: {summary['chunks_created']}")
        print("-----------------------------------------\n")
        
        return summary
